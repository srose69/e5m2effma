// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. E5M2EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// E5M2EffMA core (host side): e5ma_cast, e5ma_status_string and the choice of the kernel table
// for the current device.
#include <stdarg.h>
#include <stdio.h>
#include "e5m2effma.cuh"

namespace e5ma {

static thread_local char g_error[512];

void clear_error() { g_error[0] = 0; }
e5ma_status fail(e5ma_status s, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(g_error, sizeof g_error, fmt, ap);
  va_end(ap);
  return s;
}
e5ma_status cuda_status(cudaError_t e, const char *what) {
  if (e == cudaSuccess) return E5MA_OK;
  return fail(E5MA_CUDA_ERROR, "%s: CUDA error %d (%s): %s", what, (int)e, cudaGetErrorName(e), cudaGetErrorString(e));
}

namespace kern {

// Compiled with the same architectures as the kernels: tells which SASS the device actually runs.
__global__ void probe_kernel() {}

const DeviceInfo &device_info() {
  static DeviceInfo cache[64] = {};
  int dev = 0;
  cudaGetDevice(&dev);
  DeviceInfo &d = cache[dev & 63];
  if (!d.table) {
    int ma = 0, mi = 0;
    cudaDeviceGetAttribute(&ma, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&mi, cudaDevAttrComputeCapabilityMinor, dev);
    cudaDeviceGetAttribute(&d.sms, cudaDevAttrMultiProcessorCount, dev);
    d.cc = ma * 10 + mi;
    // A tuned table only applies when the device runs SASS built for exactly its compute capability:
    // other architectures carry kernel bodies for their own table and the defaults only.
    cudaFuncAttributes fa;
    const bool loads = cudaFuncGetAttributes(&fa, probe_kernel) == cudaSuccess;
    if (!loads) cudaGetLastError();  // "no kernel image" is reported by check_device, not left sticky
    d.supported = loads && d.cc >= 50 && d.cc < 89;
    d.table = &kTables[loads && fa.binaryVersion == d.cc ? table_index(d.cc) : 0];
  }
  return d;
}

__global__ void cast_kernel(int tin, int tout, const void *__restrict__ src, void *__restrict__ dst,
                            size_t count, const float *sinv_p, const float *scale_p, float *amax) {
  const float sinv = scale_or_one(sinv_p), qs = scale_or_one(scale_p);
  float mx = 0.f;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += (size_t)gridDim.x * blockDim.x) {
    const float v = load_any(tin, src, i, sinv);
    mx = fmaxf(mx, fabsf(v));
    store_any(tout, dst, i, v, qs);
  }
  if (amax) amax_commit(amax, mx);
}

}  // namespace kern

e5ma_status check_device() {
  int dev = -1;
  const cudaError_t e = cudaGetDevice(&dev);
  if (e != cudaSuccess) return cuda_status(e, "cudaGetDevice");
  const kern::DeviceInfo &d = kern::device_info();
  if (d.supported) return E5MA_OK;
  if (d.cc >= 89)
    return fail(E5MA_NOT_SUPPORTED, "device %d is sm_%d, which has native FP8: use cuBLASLt / Transformer Engine "
                "(E5M2EffMA targets sm_50..sm_87)", dev, d.cc);
  if (d.cc < 50) return fail(E5MA_NOT_SUPPORTED, "device %d is sm_%d; E5M2EffMA needs sm_50 or newer", dev, d.cc);
  return fail(E5MA_NOT_SUPPORTED, "device %d is sm_%d, but this build of the library has no code for it: "
              "rebuild with ARCHS including %d", dev, d.cc, d.cc);
}

}  // namespace e5ma

extern "C" const char *e5ma_last_error(void) { return e5ma::g_error; }

extern "C" e5ma_status e5ma_cast(const e5ma_tensor *src, const e5ma_tensor *dst, size_t count, cudaStream_t stream) {
  using e5ma::fail;
  e5ma::clear_error();
  const auto valid = [](e5ma_dtype t) { return t >= E5MA_E4M3 && t <= E5MA_E5M2; };
  if (!src || !dst || !src->data || !dst->data) return fail(E5MA_INVALID_VALUE, "e5ma_cast: src / dst or their data is NULL");
  if (!valid(src->dtype) || !valid(dst->dtype))
    return fail(E5MA_INVALID_VALUE, "e5ma_cast: invalid dtype (src %d, dst %d)", (int)src->dtype, (int)dst->dtype);
  if (src->scaling != E5MA_SCALING_TENSOR || dst->scaling != E5MA_SCALING_TENSOR)
    return fail(E5MA_INVALID_VALUE, "e5ma_cast: per-tensor scaling only");
  if (count == 0) return E5MA_OK;
  if (e5ma_status st = e5ma::check_device()) return st;
  const int grid = (int)((count + 255) / 256 < 8192 ? (count + 255) / 256 : 8192);
  e5ma::kern::cast_kernel<<<grid, 256, 0, stream>>>(src->dtype, dst->dtype, src->data, dst->data, count,
                                                    e5ma::is_fp8(src->dtype) ? src->scale_inv : nullptr,
                                                    e5ma::is_fp8(dst->dtype) ? dst->scale : nullptr, dst->amax);
  return e5ma::cuda_status(cudaGetLastError(), "e5ma_cast: kernel launch");
}

extern "C" const char *e5ma_status_string(e5ma_status s) {
  switch (s) {
    case E5MA_OK: return "E5MA_OK";
    case E5MA_INVALID_VALUE: return "E5MA_INVALID_VALUE";
    case E5MA_NOT_SUPPORTED: return "E5MA_NOT_SUPPORTED";
    case E5MA_CUDA_ERROR: return "E5MA_CUDA_ERROR";
  }
  return "unknown e5ma_status";
}
