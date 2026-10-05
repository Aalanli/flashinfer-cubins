// SPDX-License-Identifier: Apache-2.0
// One kernel: routingIndicesCoopKernel (cooperative launch, 128 CTAs,
// 1025..262144 tokens), KernelParams<float, bf16, true, false>. SM90+ only.
#include "moe_routing_deepseek.cuh"

namespace moe::dev::routing::routingDeepSeek {
using MoeRoutingParams = KernelParams<float, __nv_bfloat16, true, false>;
template __global__ void routingIndicesCoopKernel<MoeRoutingParams>(MoeRoutingParams);
}  // namespace moe::dev::routing::routingDeepSeek
