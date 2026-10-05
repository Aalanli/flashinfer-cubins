// SPDX-License-Identifier: Apache-2.0
// One kernel: finalizeKernelVecLoad (finalize::run otherwise), BF16 output
// and BF16 expert weights, mUsePdl = false.
#include "moe_finalize.cuh"

namespace moe::dev::finalize {
using MoeFinalizeParams = KernelParams<cutlass::bfloat16_t, cutlass::bfloat16_t, false>;
template __global__ void finalizeKernelVecLoad<MoeFinalizeParams>(MoeFinalizeParams);
}  // namespace moe::dev::finalize
