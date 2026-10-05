// SPDX-License-Identifier: Apache-2.0
// Plain-type interface between the two translation units of bmm_probe:
// bmm_probe.cu (TrtllmGenBatchedGemmRunner, the interposed CUDA API, the
// KernelParams rebuild) and bmm_moe_probe.cu (FlashInfer's MoE runners and
// launcher tile selection). Upstream compiles csrc/trtllm_batched_gemm_runner.cu
// and csrc/trtllm_fused_moe_runner.cu separately too: their two cudaUtils.h
// cannot share a translation unit.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace moe_bridge {

// One MoE problem for one batched-GEMM config (option values as integers).
struct Query {
  int config;
  bool fc1;                  // routed config (PermuteGemm1) or not (Gemm2)
  uint32_t dtype_act;        // the config's mDtypeB (hidden states)
  uint32_t dtype_weights;    // the config's mDtypeA
  bool deepseek, fused_act, shuffled;
  int act_type, eltwise_act_type, layout, tile;
  std::string pair_act;      // FC2: the paired FC1 activation; "-": the config's own
  int top_k, hidden, inter, experts, tokens;
};

struct Answer {
  std::string launcher;
  std::vector<int32_t> ladder, selected;
  int64_t activation = -1;  // MoE ActivationType of the runners built
  bool passing = false, valid = false, moe_valid = false, is_default = false;
  int m = 0, n = 0, k = 0, max_num_ctas = 0;  // TrtllmGenBatchedGemmRunner::run's problem
  size_t workspace = 0;
};

// Kernel-role buffers (nullptr = not passed).
struct Pointers {
  void *a, *sf_a, *b, *sf_b, *c, *sf_c, *scale_c, *scale_gate, *bias, *alpha, *beta,
      *clamp_limit, *route_map, *per_token_sf_b, *total_num_padded_tokens,
      *cta_idx_xy_to_batch_idx, *cta_idx_xy_to_mn_limit, *num_non_exiting_ctas, *workspace;
};

// Validity, tile selection and the runner-level problem (throws on upstream
// errors).
void query(Query const& q, Answer& a);

// PermuteGemm1::Runner::run or Gemm2::Runner::run (throws on upstream errors).
void run(Query const& q, Pointers const& p);

}  // namespace moe_bridge
