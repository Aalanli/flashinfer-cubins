// SPDX-License-Identifier: Apache-2.0
// Kernel templates copied verbatim from FlashInfer v0.2.10
// (7c79b41b8efd512eb40ec9bd56c9e6cda328d12e), Copyright (c) 2022-2025 NVIDIA
// CORPORATION, Apache-2.0 (see ../LICENSE). Each "VERBATIM" block is the given
// line range of the named upstream file; impls/moe/compiler.py checks every
// block byte for byte before compiling. Only the host-side launchers (run())
// are left out, so that a translation unit instantiates one kernel only.
#pragma once
// VERBATIM BEGIN csrc/trtllm_fused_moe_dev_kernel.cu:17-52
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include <cub/cub.cuh>
#include <cuda/functional>
#include <cuda/std/functional>
#include <cuda/std/type_traits>

#include "flashinfer/trtllm/fused_moe/DevKernel.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function for array conversion
template <class T, class U>
__host__ __device__ constexpr static U arrayConvert(T const& input) {
  cutlass::NumericArrayConverter<typename U::Element, typename T::Element, U::kElements> converter;
  return converter(input);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace moe::dev {

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace activation {

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tg = batchedGemm::trtllm::gen;

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float silu(float x) { return x / (1.0f + expf(-x)); }
// VERBATIM END
// VERBATIM BEGIN csrc/trtllm_fused_moe_dev_kernel.cu:95-161
template <typename KernelParams>
__global__ void activationDeepSeekKernel(KernelParams params) {
  using Type = typename KernelParams::Type;
  using BlockReduce = cub::BlockReduce<float, 128>;

  __shared__ float s_scaleOut;
  __shared__ typename BlockReduce::TempStorage temp_storage;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  // immediately trigger the secondary kernel when using PDL, then wait on primary
  if constexpr (KernelParams::UsePdl) {
    cudaTriggerProgrammaticLaunchCompletion();
    cudaGridDependencySynchronize();
  }
#endif
  // Loop over tokens
  for (int tokenIdx = blockIdx.z; tokenIdx < params.numTokens; tokenIdx += gridDim.z) {
    // Look over experts per token
    for (int k = blockIdx.y; k < params.topK; k += gridDim.y) {
      int const expandedIdx = tokenIdx * params.topK + k;
      int const permutedIdx = params.expandedIdxToPermutedIdx[expandedIdx];

      // Needed for expert parallelism
      if (permutedIdx == -1) continue;

      // Loop over hidden dim
      for (int hiddenIdx = threadIdx.x + blockDim.x * blockIdx.x; hiddenIdx < params.innerDim / 2;
           hiddenIdx += blockDim.x * gridDim.x) {
        int const baseIdx = permutedIdx * params.innerDim + hiddenIdx;

        int const totalNumPaddedTokens = params.totalNumPaddedTokens[0];

        int const scale1_idx = permutedIdx + totalNumPaddedTokens * (hiddenIdx / 128);
        int const scale2_idx =
            permutedIdx + totalNumPaddedTokens * ((hiddenIdx / 128) + (params.innerDim / 2 / 128));
        float const scale1 = params.inDqSfsPtr[scale1_idx];
        float const scale2 = params.inDqSfsPtr[scale2_idx];

        float x1 = scale1 * (float)params.inPtr[baseIdx];
        float x2 = scale2 * (float)params.inPtr[baseIdx + params.innerDim / 2];

        float act = silu(x2);
        float out = act * x1;

        // The largest (finite) value that can be represented using E4m3.
        float constexpr E4m3MaxVal{448.f};

        // Compute the absolute max
#if CUDA_VERSION >= 12090
        float aMax = BlockReduce(temp_storage).Reduce(fabsf(out), cuda::maximum<>{});
#else
        float aMax = BlockReduce(temp_storage).Reduce(fabsf(out), cub::Max{});
#endif
        if (threadIdx.x == 0) {
          s_scaleOut = aMax / E4m3MaxVal;
          int const scaleOut_idx = permutedIdx + totalNumPaddedTokens * (hiddenIdx / 128);
          params.outDqSfsPtr[scaleOut_idx] = aMax / E4m3MaxVal;
        }
        __syncthreads();
        float const scaleOut = s_scaleOut;
        __syncthreads();
        int const outIdx = permutedIdx * (params.innerDim / 2) + hiddenIdx;
        params.outPtr[outIdx] = (Type)(out / scaleOut);
      }
    }
  }
}
// VERBATIM END
// VERBATIM BEGIN csrc/trtllm_fused_moe_dev_kernel.cu:188-188
}  // namespace activation
// VERBATIM END
// VERBATIM BEGIN csrc/trtllm_fused_moe_dev_kernel.cu:662-662
}  // namespace moe::dev
// VERBATIM END
