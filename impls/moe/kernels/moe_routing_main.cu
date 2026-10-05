// SPDX-License-Identifier: Apache-2.0
// One kernel: FlashInfer v0.2.10 DeepSeek routingMainKernel as launched by
// routingDeepSeek::run for this package (mDtypeExpW = Bfloat16, groups,
// forceFloatInput, mUsePdl = false): KernelParams<float, bf16, true, false>.
#include "moe_routing_deepseek.cuh"

namespace moe::dev::routing::routingDeepSeek {
using MoeRoutingParams = KernelParams<float, __nv_bfloat16, true, false>;
template __global__ void routingMainKernel<MoeRoutingParams>(MoeRoutingParams);
}  // namespace moe::dev::routing::routingDeepSeek
