// SPDX-License-Identifier: Apache-2.0
// One kernel: activationDeepSeekKernel (SwiGLU + per-128 FP8 requantization)
// as activation::run launches it for E4m3 / DeepSeek FP8, mUsePdl = false.
#include "moe_activation.cuh"

namespace moe::dev::activation {
using MoeActivationParams = KernelParams<cutlass::float_e4m3_t, false>;
template __global__ void activationDeepSeekKernel<MoeActivationParams>(MoeActivationParams);
}  // namespace moe::dev::activation
