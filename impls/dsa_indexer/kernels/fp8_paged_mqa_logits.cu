// Single-kernel translation unit: DeepGEMM SM100 paged FP8 MQA logits.
//
// Upstream: https://github.com/deepseek-ai/DeepGEMM at
// 78b69000794d0937b47ae3387eff7663410264d1 (MIT, see ../LICENSE; CUTLASS is
// BSD-3-Clause, see ../CUTLASS_LICENSE). The template arguments are those
// emitted by csrc/jit_kernels/impls/sm100_mqa_logits.hpp
// sm100_paged_mqa_logits() for fp8_paged_mqa_logits with next_n = 1,
// 64 heads, head_dim 128, page_kv 64, FP8 (non-MX) scales, 2D context
// lengths, non-varlen, 3 Q / 5 KV stages, split_kv 256, 16 splits per chunk,
// 128 + 256 threads, FP32 logits and FP32 weights.
// Compiled by ../compiler.py with nvcc -cubin; this file has exactly one
// explicit kernel instantiation, so the cubin holds exactly one kernel.
#include <deep_gemm/impls/sm100_mqa_logits.cuh>

// Same shared-memory check as DeepGEMM's JIT source (get_mqa_logits_smem_size).
static_assert(sizeof(deep_gemm::layout::MQALogitsSharedStorage<
                  64, 128, false, 2, 256, 3, 5, 3, cutlass::float_e4m3_t, float>) == 220160,
              "Incorrect MQA logits shared-memory size");

void* instantiate() {
  return reinterpret_cast<void*>(
      &deep_gemm::sm100_paged_mqa_logits<1, 64, 128, 64, false, true, false, 3, 5, 256, 16,
                                         128, 256, cutlass::float_e4m3_t, float, float>);
}
