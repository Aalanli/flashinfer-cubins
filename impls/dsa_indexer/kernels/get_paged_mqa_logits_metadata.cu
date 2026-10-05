// Single-kernel translation unit: DeepGEMM SM100 paged MQA logits scheduler.
//
// Upstream: https://github.com/deepseek-ai/DeepGEMM at
// 78b69000794d0937b47ae3387eff7663410264d1 (MIT, see ../LICENSE). The template
// arguments match csrc/jit_kernels/impls/sm100_mqa_logits.hpp
// sm100_paged_mqa_logits_metadata() for the DSA indexer: next_n = 1,
// 2D context lengths, non-varlen, split_kv = 256 and num_sms = 148
// (B200 SM count, baked in as the scheduling-partition count).
// Compiled by ../compiler.py with nvcc -cubin; this file has exactly one
// explicit kernel instantiation, so the cubin holds exactly one kernel.
#include <deep_gemm/scheduler/sm100_paged_mqa_logits.cuh>

void* instantiate() {
  return reinterpret_cast<void*>(
      &deep_gemm::sched::sm100_paged_mqa_logits_metadata<1, true, false, 256, 148>);
}
