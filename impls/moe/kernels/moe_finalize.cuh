// SPDX-License-Identifier: Apache-2.0
// Kernel templates copied verbatim from FlashInfer v0.2.10
// (7c79b41b8efd512eb40ec9bd56c9e6cda328d12e), Copyright (c) 2022-2025 NVIDIA
// CORPORATION, Apache-2.0 (see ../LICENSE). Each "VERBATIM" block is the given
// line range of the named upstream file; impls/moe/compiler.py checks every
// block byte for byte before compiling. Only the host-side launchers (run())
// are left out, so that a translation unit instantiates one kernel only.
#pragma once
// VERBATIM BEGIN csrc/trtllm_fused_moe_dev_kernel.cu:17-40
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
// VERBATIM END
// VERBATIM BEGIN csrc/trtllm_fused_moe_dev_kernel.cu:430-555
namespace finalize {

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tg = batchedGemm::trtllm::gen;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void finalizeKernel(KernelParams params) {
  using Type = typename KernelParams::Type;
  using TypeExpW = typename KernelParams::TypeExpW;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  // wait on primary kernel when using PDL
  if constexpr (KernelParams::UsePdl) {
    cudaGridDependencySynchronize();
  }
#endif

  for (int tokenIdx = blockIdx.y; tokenIdx < params.numTokens; tokenIdx += gridDim.y) {
    // Loop over hidden dim
    for (int hiddenIdx = threadIdx.x + blockDim.x * blockIdx.x; hiddenIdx < params.hiddenDim;
         hiddenIdx += blockDim.x * gridDim.x) {
      // Accumulate chunk of token into registers
      float data = 0.0F;

      // Write to topK places
      for (int k = 0; k < params.topK; k++) {
        int const expandedIdx = tokenIdx * params.topK + k;
        int const permutedIdx = params.expandedIdxToPermutedIdx[expandedIdx];

        if (permutedIdx == -1) {
          continue;
        }

        if (params.expertWeightsPtr != nullptr) {
          TypeExpW const scale = params.expertWeightsPtr[expandedIdx];
          data +=
              float{scale} * float{params.inPtr[permutedIdx * params.hiddenDimPadded + hiddenIdx]};
        } else {
          data += float{params.inPtr[permutedIdx * params.hiddenDimPadded + hiddenIdx]};
        }
      }

      params.outPtr[tokenIdx * params.hiddenDim + hiddenIdx] = static_cast<Type>(data);
    }
  }
}

constexpr static int FINALIZE_THREADS_PER_BLOCK = 256;

__device__ float4 vectorizedLoadPtx(float4 const* ptr) {
  float4 ret;
  asm volatile("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
               : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w)
               : "l"(ptr));
  return ret;
}

// Final kernel to unpermute and scale
// This kernel unpermutes the original data, does the k-way reduction and performs the final skip
// connection.

template <typename KernelParams>
__global__ void finalizeKernelVecLoad(KernelParams params) {
  using Type = typename KernelParams::Type;
  using TypeExpW = typename KernelParams::TypeExpW;

  int const hiddenDimPaddedBits = params.hiddenDimPadded * cutlass::sizeof_bits<Type>::value;
  int const hiddenDimBits = params.hiddenDim * cutlass::sizeof_bits<Type>::value;
  assert(hiddenDimPaddedBits % 128 == 0);
  assert(hiddenDimBits % 128 == 0);

  // Load 128-bits per thread, according to the smallest data type we read/write
  constexpr int64_t FINALIZE_ELEM_PER_THREAD = 128 / cutlass::sizeof_bits<Type>::value;
  using InputElem = cutlass::Array<Type, FINALIZE_ELEM_PER_THREAD>;
  using OutputElem = cutlass::Array<Type, FINALIZE_ELEM_PER_THREAD>;
  using ComputeElem = cutlass::Array<float, FINALIZE_ELEM_PER_THREAD>;

  int64_t const tokenIdx = blockIdx.x;
  int64_t const startOffset = threadIdx.x;
  int64_t const stride = FINALIZE_THREADS_PER_BLOCK;
  int64_t const numElemsInPaddedCol = params.hiddenDimPadded / FINALIZE_ELEM_PER_THREAD;
  int64_t const numElemsInCol = params.hiddenDim / FINALIZE_ELEM_PER_THREAD;

  auto const offset = tokenIdx * params.hiddenDim;
  Type* outputPtr = params.outPtr + offset;
  auto* outElemPtr = reinterpret_cast<OutputElem*>(outputPtr);
  auto const* inElemPtr = reinterpret_cast<InputElem const*>(params.inPtr);

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  // wait on primary kernel when using PDL
  if constexpr (KernelParams::UsePdl) {
    cudaGridDependencySynchronize();
  }
#endif

  for (int elemIndex = startOffset; elemIndex < numElemsInCol; elemIndex += stride) {
    ComputeElem threadOutput;
    threadOutput.fill(0);
    for (int k = 0; k < params.topK; ++k) {
      int const expandedIdx = tokenIdx * params.topK + k;
      int const permutedIdx = params.expandedIdxToPermutedIdx[expandedIdx];
      if (permutedIdx == -1) {
        continue;
      }

      float const scale = (params.expertWeightsPtr != nullptr)
                              ? static_cast<float>(params.expertWeightsPtr[expandedIdx])
                              : 1.f;

      auto const* inputPermutedPtr = inElemPtr + permutedIdx * numElemsInPaddedCol;

      float4 input =
          vectorizedLoadPtx(reinterpret_cast<float4 const*>(&inputPermutedPtr[elemIndex]));
      InputElem inputPermutedElem = *reinterpret_cast<InputElem const*>(&input);
      ComputeElem expertResult = arrayConvert<InputElem, ComputeElem>(inputPermutedElem);

      threadOutput = threadOutput + scale * expertResult;
    }

    OutputElem outputElem = arrayConvert<ComputeElem, OutputElem>(threadOutput);
    outElemPtr[elemIndex] = outputElem;
  }
}
// VERBATIM END
// VERBATIM BEGIN csrc/trtllm_fused_moe_dev_kernel.cu:658-658
}  // namespace finalize
// VERBATIM END
// VERBATIM BEGIN csrc/trtllm_fused_moe_dev_kernel.cu:662-662
}  // namespace moe::dev
// VERBATIM END
