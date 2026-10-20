# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. E5M2EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
# E5M2EffMA: FP8 (E5M2 / E4M3) GEMM / GEMV / cast / Adam for GPUs without native FP8 (sm_50 ... sm_87).
#   make tune       benchmark the kernel candidates on GPU $(GPU) and record the winners for its
#                   compute capability in src/e5m2effma_tuned.h (other architectures keep defaults)
#   make lib        libe5m2effma.a for every pre-FP8 architecture (SASS only, no PTX: never JIT-ed
#                   onto sm_89+, which has native FP8);  ARCHS=native builds only the local one
#   make test       bit-exact API tests + exhaustive core validation (GPU $(GPU))
#   make bench      GEMM / GEMV / Adam throughput vs cuBLAS and memory bandwidth
#                   (bench/run_bench.sh: controlled run with board settings -> benches/<gpu>.md)
#   make install    header, static library and pkg-config file under PREFIX (default /usr/local)
# Python / PyTorch bindings: binds/python (pip install).
# Builds run in parallel by default: JOBS = min(cores, half of MemAvailable / JOB_MB), override with
# `make JOBS=n` (a plain -j on the command line is overridden).  Multi-architecture objects also
# compile their architectures in parallel (nvcc --threads) when there are more slots than objects.
CUDA_HOME ?= /usr/local/cuda
NVCC  ?= $(CUDA_HOME)/bin/nvcc
# host compiler for nvcc; g++-12 if present (CUDA 12.0-12.3 do not accept GCC 13), else nvcc's default
CCBIN ?= $(shell command -v g++-12 2>/dev/null)
GPU   ?= 0
RUN    = CUDA_VISIBLE_DEVICES=$(GPU)
DETECTED := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i $(GPU) 2>/dev/null | head -1 | tr -d '.')
TEST_ARCH ?= $(if $(DETECTED),$(DETECTED),61)
VERSION := $(shell sed -n 's/^\#define E5M2EFFMA_VERSION "\(.*\)"/\1/p' include/e5m2effma.h)
ALL_ARCHS = 50 52 53 60 61 62 70 72 75 80 86 87
ARCHS ?= $(ALL_ARCHS)
ifeq ($(ARCHS),native)
  override ARCHS := $(TEST_ARCH)
endif
FLAGS  = $(if $(CCBIN),-ccbin $(CCBIN)) -O3 -Xptxas -O3 -std=c++14 -Xcompiler -ffp-contract=off,-fPIC
GENCODE = $(foreach a,$(ARCHS),-gencode arch=compute_$(a),code=sm_$(a))

# parallelism within half of the available RAM (JOB_MB: peak of one nvcc process, ~110 MB measured)
JOB_MB ?= 512
NPROC  := $(shell nproc 2>/dev/null || echo 4)
MEM_MB := $(shell awk '/^MemAvailable:/ {print int($$2 / 1024)}' /proc/meminfo 2>/dev/null)
JOBS   ?= $(shell n=$$(( $(or $(MEM_MB),4096) / 2 / $(JOB_MB) )); [ $$n -gt $(NPROC) ] && n=$(NPROC); [ $$n -lt 1 ] && n=1; echo $$n)
MAKEFLAGS += -j$(JOBS)

CORE_H = include/e5m2effma.h src/e5m2effma.cuh src/e5m2effma_tuned.h
GEMM_H = $(CORE_H) src/gemm_e5m2effma.cuh
GEMV_H = $(CORE_H) src/gemv_e5m2effma.cuh

# kernel configurations the library instantiates: defaults + those used by tuned entries
TUNED_IDS = $(shell sed -n 's/^\#define E5MA_$(1)_USED(X)//p' src/e5m2effma_tuned.h | grep -o '[0-9]\+')
GEMM_IDS  = 0 1 $(call TUNED_IDS,GEMM)
GEMV_IDS  = 0 $(call TUNED_IDS,GEMV)
TYPES     = 4 0 1 3
FP8_TYPES = 4 0
# objects of similar compile cost: GEMM per config x A dtype x B dtype, GEMV per config x A dtype x
# x dtype (dtype codes: E5M2 4, E4M3 0, BF16 1, FP32 3)
INST = $(foreach i,$(GEMV_IDS),$(foreach a,$(FP8_TYPES),$(foreach t,$(TYPES),inst_gemv_$(i)_$(a)_$(t)))) \
       $(foreach i,$(GEMM_IDS),$(foreach a,$(TYPES),$(foreach b,$(TYPES),inst_gemm_$(i)_$(a)_$(b))))
OBJS = e5m2effma gemm_e5m2effma gemv_e5m2effma adamw_e5m2effma $(INST)
NVCC_THREADS := $(shell t=$$(( $(JOBS) / $(words $(OBJS)) )); [ $$t -lt 1 ] && t=1; a=$(words $(ARCHS)); [ $$t -gt $$a ] && t=$$a; echo $$t)
word_ = $(word $(1),$(subst _, ,$(2)))

all: lib test-bin bench-bin

# ---- library: all pre-FP8 architectures (out/obj) and the local one (out/native) ---------------
define LIBRARY  # $(1) = object dir, $(2) = arch flags
$(1)/e5m2effma.o $(1)/adamw_e5m2effma.o: $(1)/%.o: src/%.cu $(CORE_H) | $(1)
	$(NVCC) $(FLAGS) $(2) -c -o $$@ $$<
$(1)/gemm_e5m2effma.o: src/gemm_e5m2effma.cu $(GEMM_H) src/gemv_e5m2effma.cuh | $(1)
	$(NVCC) $(FLAGS) $(2) -c -o $$@ $$<
$(1)/gemv_e5m2effma.o: src/gemv_e5m2effma.cu $(GEMV_H) | $(1)
	$(NVCC) $(FLAGS) $(2) -c -o $$@ $$<
$(1)/inst_gemm_%.o: src/gemm_e5m2effma_inst.cu $(GEMM_H) | $(1)
	$(NVCC) $(FLAGS) $(2) -DE5MA_ID=$$(call word_,1,$$*) -DE5MA_TA=$$(call word_,2,$$*) -DE5MA_TB=$$(call word_,3,$$*) -c -o $$@ $$<
$(1)/inst_gemv_%.o: src/gemv_e5m2effma_inst.cu $(GEMV_H) | $(1)
	$(NVCC) $(FLAGS) $(2) -DE5MA_ID=$$(call word_,1,$$*) -DE5MA_TA=$$(call word_,2,$$*) -DE5MA_TX=$$(call word_,3,$$*) -c -o $$@ $$<
$(1)/libe5m2effma.a: $(foreach o,$(OBJS),$(1)/$(o).o)
	rm -f $$@ && ar rcs $$@ $$^
$(1):
	mkdir -p $$@
endef
# multi-architecture objects (binds/python builds its own copy for the architectures it targets)
OBJDIR ?= out/obj
$(eval $(call LIBRARY,$(OBJDIR),$(GENCODE) --threads $(NVCC_THREADS)))
$(eval $(call LIBRARY,out/native,-arch=sm_$(TEST_ARCH)))
out/libe5m2effma.a: $(OBJDIR)/libe5m2effma.a
	cp $< $@
lib: out/libe5m2effma.a

# ---- tuning, tests, benchmarks (local architecture) -------------------------------------------
NATIVE = out/native/libe5m2effma.a
NATIVE_FLAGS = $(FLAGS) -arch=sm_$(TEST_ARCH)
out/tune: tune/tune.cu $(GEMM_H) $(GEMV_H) | out/native
	$(NVCC) $(NATIVE_FLAGS) -o $@ $<
tune: out/tune                   ## writes src/e5m2effma_tuned.h for this GPU; rebuild the library afterwards
	$(RUN) ./out/tune src/e5m2effma_tuned.h

TESTS   = test_gemm test_gemv test_adamw test_e5m2effma
BENCHES = bench_gemm bench_gemv bench_adamw
$(addprefix out/,$(TESTS)): out/%: tests/%.cu tests/test_util.h tests/e5m2effma_ref.h $(NATIVE)
	$(NVCC) $(NATIVE_FLAGS) -o $@ $< $(NATIVE)
out/test_e5m2effma_neg: tests/test_e5m2effma.cu tests/test_util.h tests/e5m2effma_ref.h $(NATIVE)
	$(NVCC) $(NATIVE_FLAGS) -DE5MA_NEGATIVE_CONTROLS -o $@ $< $(NATIVE)
out/libref.so: tests/ref_capi.cpp tests/e5m2effma_ref.h | out/native
	g++ -O2 -shared -fPIC -o $@ $<
$(addprefix out/,$(BENCHES)): out/%: bench/%.cu bench/bench_util.h include/e5m2effma.h $(NATIVE)
	$(NVCC) $(NATIVE_FLAGS) -o $@ $< $(NATIVE) -lcublas -lnvidia-ml

test-bin: $(addprefix out/,$(TESTS)) out/test_e5m2effma_neg out/libref.so
bench-bin: $(addprefix out/,$(BENCHES))

test: test-bin                   ## API bit-exact, 6 x 2^32 exhaustive, negative controls, reference
	$(RUN) ./out/test_gemm
	$(RUN) ./out/test_gemv
	$(RUN) ./out/test_adamw
	$(RUN) ./out/test_e5m2effma
	$(RUN) ./out/test_e5m2effma_neg
	python3 tests/check_ref.py out/libref.so
bench: bench-bin
	$(RUN) ./out/bench_gemm
	$(RUN) ./out/bench_gemv
	$(RUN) ./out/bench_adamw

# ---- install (C / C++ projects: pkg-config --cflags --libs e5m2effma) -------------------------
PREFIX ?= /usr/local
install: $(OBJDIR)/libe5m2effma.a
	install -d $(DESTDIR)$(PREFIX)/include $(DESTDIR)$(PREFIX)/lib/pkgconfig
	install -m 644 include/e5m2effma.h $(DESTDIR)$(PREFIX)/include/
	install -m 644 $(OBJDIR)/libe5m2effma.a $(DESTDIR)$(PREFIX)/lib/
	printf 'prefix=%s\nincludedir=$${prefix}/include\nlibdir=$${prefix}/lib\n\nName: e5m2effma\nDescription: FP8 (E5M2 / E4M3) GEMM / GEMV / Adam for GPUs without native FP8\nVersion: $(VERSION)\nCflags: -I$${includedir} -I$(CUDA_HOME)/include\nLibs: -L$${libdir} -le5m2effma -L$(CUDA_HOME)/lib64 -lcudart -lstdc++\n' \
	       '$(PREFIX)' > $(DESTDIR)$(PREFIX)/lib/pkgconfig/e5m2effma.pc

clean:
	rm -rf out/obj out/native out/libe5m2effma.a out/tune out/libref.so \
	       $(addprefix out/,$(TESTS) $(BENCHES)) out/test_e5m2effma_neg

.PHONY: all lib tune test-bin bench-bin test bench install clean
