// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. E5M2EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// PyTorch bindings of E5M2EffMA (include/e5m2effma.h).  Every argument is checked here, before any
// pointer reaches the library: wrong dtype -> TypeError, wrong device / shape / layout / alignment ->
// ValueError, unsupported GPU -> NotImplementedError, library or CUDA failure -> RuntimeError with the
// library's description (e5ma_last_error).  Calls run on the current CUDA stream of the tensors' device.
#include <torch/extension.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <optional>
#include <string>
#include "e5m2effma.h"

using at::Tensor;
using OptT = std::optional<Tensor>;

namespace {

e5ma_dtype to_e5ma(at::ScalarType t, const char *what) {
  switch (t) {
    case at::kFloat8_e5m2: return E5MA_E5M2;
    case at::kFloat8_e4m3fn: return E5MA_E4M3;
    case at::kBFloat16: return E5MA_BF16;
    case at::kHalf: return E5MA_FP16;
    case at::kFloat: return E5MA_FP32;
    default:
      TORCH_CHECK_TYPE(false, what, ": unsupported dtype ", t,
                       " (expected torch.float8_e5m2, torch.float8_e4m3fn, torch.bfloat16, torch.float16 or torch.float32)");
  }
  return E5MA_FP32;
}

bool is_fp8(at::ScalarType t) { return t == at::kFloat8_e5m2 || t == at::kFloat8_e4m3fn; }

void check_dtype(const Tensor &t, const char *what, std::initializer_list<at::ScalarType> ok, const char *expected) {
  for (auto s : ok)
    if (t.scalar_type() == s) return;
  TORCH_CHECK_TYPE(false, what, " must be ", expected, ", got ", t.scalar_type());
}

// The device every tensor of a call must live on; also rejects GPUs the library does not cover.
void check_device(const Tensor &t, const at::Device &dev, const char *what) {
  TORCH_CHECK_VALUE(t.is_cuda(), what, " must be a CUDA tensor, got a tensor on ", t.device());
  TORCH_CHECK_VALUE(t.device() == dev, what, " is on ", t.device(), " but the operation runs on ", dev);
}
void check_gpu(const at::Device &dev) {
  const cudaDeviceProp *p = at::cuda::getDeviceProperties(dev.index());
  const int cc = p->major * 10 + p->minor;
  TORCH_CHECK_NOT_IMPLEMENTED(cc < 89, "E5M2EffMA targets GPUs without native FP8 (sm_50..sm_87); ", p->name,
                              " is sm_", cc, " and has native FP8: use torch._scaled_mm / Transformer Engine");
  TORCH_CHECK_NOT_IMPLEMENTED(cc >= 50, p->name, " is sm_", cc, "; E5M2EffMA needs sm_50 or newer");
}

// optional per-tensor scale / amax: float32, one element, on dev
const float *scalar_ptr(const OptT &t, const at::Device &dev, const char *what) {
  if (!t) return nullptr;
  check_device(*t, dev, what);
  TORCH_CHECK_TYPE(t->scalar_type() == at::kFloat, what, " must be torch.float32, got ", t->scalar_type());
  TORCH_CHECK_VALUE(t->numel() == 1, what, " must have exactly one element, got shape ", t->sizes());
  return t->data_ptr<float>();
}

// dequantization factor of a GEMM operand: one float32 (per tensor) or a vector of `outer` floats,
// one per row of a / column of b (row-wise scaling, as torch._scaled_mm).  holder keeps a contiguous
// copy alive when the vector was strided.
struct OperandScale { const float *ptr = nullptr; bool outer = false; Tensor holder; };
OperandScale operand_scale(const OptT &t, const at::Device &dev, const char *what, int64_t outer, const char *shapes) {
  OperandScale r;
  if (!t) return r;
  check_device(*t, dev, what);
  TORCH_CHECK_TYPE(t->scalar_type() == at::kFloat, what, " must be torch.float32, got ", t->scalar_type());
  if (t->numel() == 1) { r.holder = *t; r.ptr = t->data_ptr<float>(); return r; }
  TORCH_CHECK_VALUE(t->numel() == outer && t->dim() <= 2, what, " must have one element (per tensor) or shape ", shapes,
                    " (one factor per ", outer, " outer indices), got shape ", t->sizes());
  r.holder = t->contiguous().view(-1);
  r.ptr = r.holder.data_ptr<float>();
  r.outer = true;
  return r;
}

void check_status(e5ma_status s, const char *what) {
  if (s == E5MA_OK) return;
  const std::string msg = std::string(what) + " failed (" + e5ma_status_string(s) + "): " + e5ma_last_error();
  TORCH_CHECK_NOT_IMPLEMENTED(s != E5MA_NOT_SUPPORTED, msg);
  TORCH_CHECK_VALUE(s != E5MA_INVALID_VALUE, msg);
  TORCH_CHECK(false, msg);
}

void check_no_overlap(const Tensor &out, const Tensor &in, const char *out_name, const char *in_name) {
  TORCH_CHECK_VALUE(at::get_overlap_status(out, in) == at::MemOverlapStatus::No, out_name,
                    " must not share memory with ", in_name);
}

// Storage of a 2-D operand as seen by the column-major API: row-contiguous [rows, cols] is the
// column-major cols x rows matrix with ld = stride(0); column-contiguous is rows x cols, ld = stride(1).
struct Layout { bool ok, row_major; int64_t ld; };
Layout layout_of(const Tensor &t) {
  const int64_t r = t.size(0), c = t.size(1);
  if (t.stride(1) == 1 && (r <= 1 || t.stride(0) >= c)) return {true, true, r <= 1 ? c : t.stride(0)};
  if (t.stride(0) == 1 && (c <= 1 || t.stride(1) >= r)) return {true, false, c <= 1 ? r : t.stride(1)};
  return {false, false, 0};
}
// The library needs a 16-byte aligned operand, row- or column-contiguous, with a leading dimension that
// is a multiple of 16.  Anything else (a strided view, a slice with an offset, a row-major [K, N] with
// N % 16 != 0, ...) is copied once into the K-contiguous layout, which always qualifies since K % 16 == 0.
Tensor qualified(const Tensor &t, bool k_is_dim1, Layout &l) {
  l = layout_of(t);
  if (l.ok && l.ld % 16 == 0 && reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0) return t;
  const Tensor c = k_is_dim1 ? t.contiguous() : t.t().contiguous().t();
  l = layout_of(c);
  return c;
}

int as_int(int64_t v, const char *what) {
  TORCH_CHECK_VALUE(v <= INT32_MAX, what, " = ", v, " exceeds the 32-bit range of the library");
  return (int)v;
}

}  // namespace

// out = alpha * (scale_a * scale_b) * (a @ b) + bias + beta * out   (row-major, like torch.matmul)
//   a [M, K], b [K, N]: E5M2, E4M3, BF16 or FP32 in any pairing, K a multiple of 16 (operands the
//   library cannot take as they are get a K-contiguous copy, see qualified()).  out [M, N]
//   row-contiguous, any of the five dtypes.  With M <= 8, a row-contiguous and b column-contiguous FP8
//   (e.g. b = W.t()) the GEMV kernel runs.
Tensor mm(const Tensor &a, const Tensor &b, std::optional<at::ScalarType> out_dtype, const OptT &bias, OptT out,
          double alpha, double beta, const OptT &scale_a, const OptT &scale_b, const OptT &scale_out,
          const OptT &amax_out) {
  TORCH_CHECK_VALUE(a.dim() == 2 && b.dim() == 2, "a and b must be 2-D, got shapes ", a.sizes(), " and ", b.sizes());
  const auto ops = {at::kFloat8_e5m2, at::kFloat8_e4m3fn, at::kBFloat16, at::kFloat};
  const char *ops_s = "torch.float8_e5m2, torch.float8_e4m3fn, torch.bfloat16 or torch.float32";
  check_dtype(a, "a", ops, ops_s);
  check_dtype(b, "b", ops, ops_s);
  TORCH_CHECK_VALUE(a.is_cuda(), "a must be a CUDA tensor, got a tensor on ", a.device());
  const at::Device dev = a.device();
  check_device(b, dev, "b");
  check_gpu(dev);
  const int64_t M = a.size(0), K = a.size(1), N = b.size(1);
  TORCH_CHECK_VALUE(b.size(0) == K, "shapes do not match for a @ b: ", a.sizes(), " and ", b.sizes());
  TORCH_CHECK_VALUE(K % 16 == 0, "K = ", K, " must be a multiple of 16 (pad the reduction dimension with zeros)");
  TORCH_CHECK_VALUE(scale_a == std::nullopt || is_fp8(a.scalar_type()),
                    "scale_a is only meaningful for an FP8 a (it is the dequantization factor)");
  TORCH_CHECK_VALUE(scale_b == std::nullopt || is_fp8(b.scalar_type()),
                    "scale_b is only meaningful for an FP8 b (it is the dequantization factor)");

  if (beta != 0.0) {
    TORCH_CHECK_VALUE(out.has_value(), "beta != 0 accumulates into out, so out must be given");
  }
  if (out) {
    check_device(*out, dev, "out");
    to_e5ma(out->scalar_type(), "out");
    TORCH_CHECK_VALUE(!out_dtype || *out_dtype == out->scalar_type(), "out_dtype ", *out_dtype,
                      " does not match out.dtype ", out->scalar_type());
    TORCH_CHECK_VALUE(out->dim() == 2 && out->size(0) == M && out->size(1) == N, "out must have shape [", M, ", ", N,
                      "], got ", out->sizes());
    TORCH_CHECK_VALUE(out->stride(1) == 1 && (M <= 1 || out->stride(0) >= N),
                      "out must be row-contiguous (stride(1) == 1), got strides ", out->strides());
    TORCH_CHECK_VALUE(at::has_internal_overlap(*out) != at::MemOverlap::Yes, "out must not have internally overlapping memory");
    check_no_overlap(*out, a, "out", "a");
    check_no_overlap(*out, b, "out", "b");
  } else {
    out = at::empty({M, N}, a.options().dtype(out_dtype.value_or(at::kBFloat16)));
  }
  const e5ma_dtype dt = to_e5ma(out->scalar_type(), "out");
  TORCH_CHECK_VALUE(scale_out == std::nullopt || is_fp8(out->scalar_type()),
                    "scale_out is only meaningful for an FP8 output (it is the quantization factor)");
  if (bias) {
    check_device(*bias, dev, "bias");
    to_e5ma(bias->scalar_type(), "bias");
    TORCH_CHECK_VALUE(bias->dim() == 1 && bias->size(0) == N && bias->stride(0) == 1,
                      "bias must be a contiguous vector of length N = ", N, ", got shape ", bias->sizes());
  }
  if (M == 0 || N == 0) return *out;

  Layout la, lb;
  const Tensor qa = qualified(a, true, la), qb = qualified(b, false, lb);

  // column-major view: out^T (N x M) = b^T (N x K) * a^T (K x M)
  const e5ma_op transa = lb.row_major ? E5MA_OP_N : E5MA_OP_T, transb = la.row_major ? E5MA_OP_N : E5MA_OP_T;
  const c10::cuda::CUDAGuard guard(dev);
  // row-wise scale of a: per output row (M) = per column of the column-major D^T = per column of op(B);
  // scale of b: per output column (N) = per row of op(A)
  const OperandScale sa = operand_scale(scale_a, dev, "scale_a", M, "[M] or [M, 1]");
  const OperandScale sb = operand_scale(scale_b, dev, "scale_b", N, "[N] or [1, N]");
  const e5ma_tensor A = {qb.data_ptr(), to_e5ma(b.scalar_type(), "b"), sb.ptr, nullptr, nullptr,
                          sb.outer ? E5MA_SCALING_OUTER : E5MA_SCALING_TENSOR};
  const e5ma_tensor B = {qa.data_ptr(), to_e5ma(a.scalar_type(), "a"), sa.ptr, nullptr, nullptr,
                          sa.outer ? E5MA_SCALING_OUTER : E5MA_SCALING_TENSOR};
  const e5ma_tensor D = {out->data_ptr(), dt, nullptr, scalar_ptr(scale_out, dev, "scale_out"),
                          const_cast<float *>(scalar_ptr(amax_out, dev, "amax_out"))};
  e5ma_tensor BIAS = {};
  if (bias) BIAS = {bias->data_ptr(), to_e5ma(bias->scalar_type(), "bias"), nullptr, nullptr, nullptr};
  const int64_t ldd = M <= 1 ? N : out->stride(0);
  check_status(e5ma_gemm(transa, transb, as_int(N, "N"), as_int(M, "M"), as_int(K, "K"), (float)alpha, &A,
                         as_int(lb.ld, "ld(b)"), &B, as_int(la.ld, "ld(a)"), (float)beta, nullptr, 0, &D,
                         as_int(ldd, "ld(out)"), bias ? &BIAS : nullptr, at::cuda::getCurrentCUDAStream().stream()),
               "e5ma_gemm");
  return *out;
}

// dst = convert(src * scale_inv) (quantized with scale for an FP8 dst); amax gets max |value|.
Tensor cast(const Tensor &src, at::ScalarType dtype, const OptT &scale_inv, const OptT &scale, const OptT &amax) {
  TORCH_CHECK_VALUE(src.is_cuda(), "x must be a CUDA tensor, got a tensor on ", src.device());
  const at::Device dev = src.device();
  check_gpu(dev);
  const e5ma_dtype st = to_e5ma(src.scalar_type(), "x"), dt = to_e5ma(dtype, "dtype");
  TORCH_CHECK_VALUE(scale_inv == std::nullopt || is_fp8(src.scalar_type()), "scale_inv is only meaningful for an FP8 x");
  TORCH_CHECK_VALUE(scale == std::nullopt || is_fp8(dtype), "scale is only meaningful for an FP8 result");
  const Tensor x = src.contiguous();
  Tensor y = at::empty(x.sizes(), x.options().dtype(dtype));
  if (x.numel() == 0) return y;
  const c10::cuda::CUDAGuard guard(dev);
  const e5ma_tensor S = {x.data_ptr(), st, scalar_ptr(scale_inv, dev, "scale_inv"), nullptr, nullptr};
  const e5ma_tensor D = {y.data_ptr(), dt, nullptr, scalar_ptr(scale, dev, "scale"),
                          const_cast<float *>(scalar_ptr(amax, dev, "amax"))};
  check_status(e5ma_cast(&S, &D, (size_t)x.numel(), at::cuda::getCurrentCUDAStream().stream()), "e5ma_cast");
  return y;
}

// One in-place Adam / AdamW step (see include/e5m2effma.h for the exact order of operations).
void adam_step(Tensor param, const Tensor &grad, Tensor exp_avg, Tensor exp_avg_sq, double lr, double beta1,
               double beta2, double eps, double weight_decay, int64_t step, bool adamw, bool bias_correction,
               double grad_scale, OptT param_fp8, const OptT &fp8_scale, const OptT &fp8_amax) {
  TORCH_CHECK_VALUE(param.is_cuda(), "param must be a CUDA tensor, got a tensor on ", param.device());
  const at::Device dev = param.device();
  check_device(grad, dev, "grad");
  check_device(exp_avg, dev, "exp_avg");
  check_device(exp_avg_sq, dev, "exp_avg_sq");
  check_gpu(dev);
  check_dtype(param, "param", {at::kFloat, at::kBFloat16}, "torch.float32 or torch.bfloat16");
  check_dtype(grad, "grad", {at::kFloat, at::kBFloat16, at::kHalf}, "torch.float32, torch.bfloat16 or torch.float16");
  check_dtype(exp_avg, "exp_avg", {at::kFloat, at::kBFloat16}, "torch.float32 or torch.bfloat16");
  TORCH_CHECK_TYPE(exp_avg_sq.scalar_type() == exp_avg.scalar_type(), "exp_avg_sq must have the dtype of exp_avg (",
                   exp_avg.scalar_type(), "), got ", exp_avg_sq.scalar_type());
  const int64_t n = param.numel();
  using Named = std::pair<const Tensor *, const char *>;
  for (const Named &p : {Named{&grad, "grad"}, Named{&exp_avg, "exp_avg"}, Named{&exp_avg_sq, "exp_avg_sq"}})
    TORCH_CHECK_VALUE(p.first->numel() == n, p.second, " has ", p.first->numel(), " elements, param has ", n);
  for (const Named &p : {Named{&param, "param"}, Named{&grad, "grad"}, Named{&exp_avg, "exp_avg"}, Named{&exp_avg_sq, "exp_avg_sq"}})
    TORCH_CHECK_VALUE(p.first->is_contiguous(), p.second, " must be contiguous");
  TORCH_CHECK_VALUE(!grad.is_sparse(), "sparse gradients are not supported");
  TORCH_CHECK_VALUE(step >= 1 && step <= INT32_MAX, "step must be >= 1 (1-based), got ", step);
  TORCH_CHECK_VALUE(lr >= 0.0 && eps >= 0.0 && weight_decay >= 0.0, "lr, eps and weight_decay must be non-negative");
  TORCH_CHECK_VALUE(beta1 >= 0.0 && beta1 < 1.0 && beta2 >= 0.0 && beta2 < 1.0, "betas must be in [0, 1), got (",
                    beta1, ", ", beta2, ")");
  check_no_overlap(exp_avg, exp_avg_sq, "exp_avg", "exp_avg_sq");
  check_no_overlap(param, exp_avg, "param", "exp_avg");
  check_no_overlap(param, exp_avg_sq, "param", "exp_avg_sq");
  if (param_fp8) {
    check_device(*param_fp8, dev, "param_fp8");
    TORCH_CHECK_TYPE(is_fp8(param_fp8->scalar_type()), "param_fp8 must be torch.float8_e5m2 or torch.float8_e4m3fn, got ",
                     param_fp8->scalar_type());
    TORCH_CHECK_VALUE(param_fp8->numel() == n && param_fp8->is_contiguous(),
                      "param_fp8 must be contiguous with the element count of param (", n, ")");
  } else {
    TORCH_CHECK_VALUE(fp8_scale == std::nullopt && fp8_amax == std::nullopt, "fp8_scale / fp8_amax need param_fp8");
  }
  if (n == 0) return;
  const c10::cuda::CUDAGuard guard(dev);
  const e5ma_adam_config cfg = {(float)lr, (float)beta1, (float)beta2, (float)eps, (float)weight_decay, (int)step,
                                 adamw ? 1 : 0, bias_correction ? 1 : 0, (float)grad_scale};
  const e5ma_tensor P = {param.data_ptr(), to_e5ma(param.scalar_type(), "param"), nullptr, nullptr, nullptr};
  const e5ma_tensor G = {grad.data_ptr(), to_e5ma(grad.scalar_type(), "grad"), nullptr, nullptr, nullptr};
  const e5ma_tensor Mo = {exp_avg.data_ptr(), to_e5ma(exp_avg.scalar_type(), "exp_avg"), nullptr, nullptr, nullptr};
  const e5ma_tensor V = {exp_avg_sq.data_ptr(), to_e5ma(exp_avg_sq.scalar_type(), "exp_avg_sq"), nullptr, nullptr, nullptr};
  e5ma_tensor Q = {};
  if (param_fp8)
    Q = {param_fp8->data_ptr(), to_e5ma(param_fp8->scalar_type(), "param_fp8"), nullptr,
         scalar_ptr(fp8_scale, dev, "fp8_scale"), const_cast<float *>(scalar_ptr(fp8_amax, dev, "fp8_amax"))};
  check_status(e5ma_adam(&cfg, (size_t)n, &P, &G, &Mo, &V, param_fp8 ? &Q : nullptr,
                         at::cuda::getCurrentCUDAStream().stream()),
               "e5ma_adam");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "E5M2EffMA: FP8 (E5M2 / E4M3) GEMM / GEMV / cast / Adam for GPUs without native FP8 (sm_50..sm_87)";
  m.attr("__version__") = E5M2EFFMA_VERSION;
  m.def("mm", &mm, py::arg("a"), py::arg("b"), py::arg("out_dtype") = py::none(), py::arg("bias") = py::none(),
        py::arg("out") = py::none(), py::arg("alpha") = 1.0, py::arg("beta") = 0.0, py::arg("scale_a") = py::none(),
        py::arg("scale_b") = py::none(), py::arg("scale_out") = py::none(), py::arg("amax_out") = py::none());
  m.def("cast", &cast, py::arg("x"), py::arg("dtype"), py::arg("scale_inv") = py::none(),
        py::arg("scale") = py::none(), py::arg("amax") = py::none());
  m.def("adam_step", &adam_step, py::arg("param"), py::arg("grad"), py::arg("exp_avg"), py::arg("exp_avg_sq"),
        py::arg("lr"), py::arg("beta1"), py::arg("beta2"), py::arg("eps"), py::arg("weight_decay"), py::arg("step"),
        py::arg("adamw"), py::arg("bias_correction"), py::arg("grad_scale"), py::arg("param_fp8") = py::none(),
        py::arg("fp8_scale") = py::none(), py::arg("fp8_amax") = py::none());
}
