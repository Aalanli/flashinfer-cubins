// SPDX-License-Identifier: Apache-2.0
// One kernel: finalizeKernel (finalize::run when numBlocksX * numTokens <
// 1184), BF16 output and BF16 expert weights, mUsePdl = false.
#include "moe_finalize.cuh"

namespace moe::dev::finalize {
using MoeFinalizeParams = KernelParams<cutlass::bfloat16_t, cutlass::bfloat16_t, false>;
template __global__ void finalizeKernel<MoeFinalizeParams>(MoeFinalizeParams);
}  // namespace moe::dev::finalize
