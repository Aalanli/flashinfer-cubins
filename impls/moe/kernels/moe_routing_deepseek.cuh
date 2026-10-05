// SPDX-License-Identifier: Apache-2.0
// Kernel templates copied verbatim from FlashInfer v0.2.10
// (7c79b41b8efd512eb40ec9bd56c9e6cda328d12e), Copyright (c) 2022-2025 NVIDIA
// CORPORATION, Apache-2.0 (see ../LICENSE). Each "VERBATIM" block is the given
// line range of the named upstream file; impls/moe/compiler.py checks every
// block byte for byte before compiling. Only the host-side launchers (run())
// are left out, so that a translation unit instantiates one kernel only.
#pragma once
// VERBATIM BEGIN csrc/trtllm_fused_moe_routing_deepseek.cu:17-407
#include "flashinfer/trtllm/fused_moe/RoutingKernel.cuh"

namespace moe::dev::routing {

namespace routingDeepSeek {

////////////////////////////////////////////////////////////////////////////////////////////////////

static constexpr int NumThreads = 256;
static constexpr int NumWarps = NumThreads / WarpSize;
static constexpr int NumTopGroupScores = 2;
static constexpr int MaxNumTopExperts = 8;
static constexpr int MaxNumTopGroups = 4;

template <typename KernelParams>
__global__ void routingMainKernel(KernelParams params) {
  // declare types
  using OutputT = typename KernelParams::OutputT;
  using InputT = typename KernelParams::InputT;

  // declare shared memory structure
  // number of experts is bounded by number of threads
  __shared__ float __attribute((aligned(128))) smemScoreSigmoid[NumThreads];
  __shared__ float __attribute((aligned(128))) smemScoreBias[NumThreads];
  // number of expert groups is bounded by number of warps
  __shared__ float __attribute((aligned(128))) smemGroupScores[NumWarps];

  // needed for warp reduce
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WarpSize>(block);
  // for the final reduction of weight norm, only some lanes need to participate
  int32_t laneIdx = threadIdx.x % WarpSize;
  int32_t warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WarpSize, 0);
  // warps outside the range of expert groups do not participate
  if constexpr (KernelParams::UseGroups) {
    if (warpIdx >= params.mNumExpertGroups) {
      return;
    }
  }

  // note that for invalid scores, we simply use a negative value:
  // they work well even with the compacted format used in topK, and
  // sigmoid / bias activated scores cannot be negative
  static constexpr float invalidScoreFloat = -1.F;
  const OutputT invalidScore = OutputT{invalidScoreFloat};

  // load bias already; each warp represents one expert group
  auto threadExpert = threadIdx.x;
  bool expertSelected = threadExpert < params.mNumExperts;
  if constexpr (KernelParams::UseGroups) {
    threadExpert = warpIdx * params.mNumExpertsPerGroup + laneIdx;
    expertSelected = laneIdx < params.mNumExpertsPerGroup;
  }
  auto scoreIdx = int64_t{blockIdx.x} * int64_t{params.mNumExperts} + threadExpert;
  auto biasVal = expertSelected ? params.mPtrRoutingBias[threadExpert] : invalidScore;

  // initialize the mPtrExpertCounts
  if (params.mPtrExpertCounts) {
    int32_t globalThreadIdx = blockIdx.x * NumThreads + threadIdx.x;
    int32_t globalThreadStride = gridDim.x * NumThreads;
    int32_t expertCountsNum = 2 * params.mNumExperts;
    initArr(globalThreadIdx, expertCountsNum, globalThreadStride, params.mPtrExpertCounts, 0);
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  // trigger the secondary kernel when using PDL, then wait on primary
  if constexpr (KernelParams::UsePdl) {
    cudaTriggerProgrammaticLaunchCompletion();
    cudaGridDependencySynchronize();
  }
#endif

  // get our assigned thread score; each warp represents one expert group
  float score =
      expertSelected ? static_cast<float>(params.mPtrScores[scoreIdx]) : invalidScoreFloat;
  // get the sigmoid score
  // note that for invalid values, we simply use a negative value:
  // sigmoig scores are always strictly positive
  auto scoreSigmoid = sigmoid_accurate(score);
  // write the sigmoid score to shared for later use
  if (expertSelected) {
    smemScoreSigmoid[threadExpert] = scoreSigmoid;
  }
  // get the score with bias
  // note that with invalid values, because sigmoid is < 1 and bias is -1,
  // we must get a negative value, which is smaller than any valid value
  auto scoreBias = float{scoreSigmoid + float{biasVal}};

  if (expertSelected) {
    smemScoreBias[threadExpert] = scoreBias;
  }

  // registers for top group score reduction
  float topExpGroupScores[NumTopGroupScores];
  [[maybe_unused]] int32_t topExpGroupIdx[NumTopGroupScores];
  float topGroups[MaxNumTopGroups];  // bound of params.mNumLimitedGroups
  int32_t topGroupIdx[MaxNumTopGroups];
  float expertScoreGroup[MaxNumTopGroups];
  int32_t expertIdxGroup[MaxNumTopGroups];
  float topScores[MaxNumTopExperts];  // bound of params.mTopK
  int32_t topExperts[MaxNumTopExperts];

  if constexpr (KernelParams::UseGroups) {
    topk::reduceTopK(warp, topExpGroupScores, topExpGroupIdx, scoreBias, threadExpert,
                     /* minValue */ invalidScoreFloat);

    // get the final group score and write it to shared
    if (cute::elect_one_sync()) {
      auto groupScore = topExpGroupScores[0] + topExpGroupScores[1];
      smemGroupScores[warpIdx] = groupScore;
    }
  }

  // make group scores available to all warps
  __syncthreads();

  auto localExpertExtent = params.mNumLocalExperts << params.mLocalExpertsStrideLog2;
  if (warpIdx == 0) {
    // a single warp performs the selection of top groups, and goes on to select the final experts
    if constexpr (KernelParams::UseGroups) {
      float groupScore =
          laneIdx < params.mNumExpertGroups ? smemGroupScores[laneIdx] : invalidScoreFloat;

      topk::reduceTopK(warp, topGroups, topGroupIdx, groupScore, laneIdx,
                       /* minValue */ invalidScoreFloat);

      // final expert selection: get relevant indexes and scores from shared

#pragma unroll
      for (int ii = 0; ii < MaxNumTopGroups; ++ii) {  // bound of params.mNumLimitedGroups
        auto groupIdx = topGroupIdx[ii];
        expertIdxGroup[ii] = groupIdx * params.mNumExpertsPerGroup + laneIdx;
        // note: expertSelected implies laneIdx < params.mNumExpertsPerGroup.
        // we have params.mNumExpertsPerGroup == params.mNumExperts / params.mNumExpertGroups,
        // thus groupIdx <= params.mNumExpertGroups - 1 =>
        // groupIdx * params.mNumExpertsPerGroup <= params.mNumExperts - params.mNumExpertsPerGroup
        // => expertIdxGroup[ii] < params.mNumExperts <= NumThreads,
        // so the access is safe here
        expertScoreGroup[ii] = groupIdx < params.mNumExpertGroups && expertSelected
                                   ? smemScoreBias[expertIdxGroup[ii]]
                                   : invalidScoreFloat;
      }
    } else {
      // without groups, each thread just takes `MaxNumTopGroups` experts

#pragma unroll
      for (int ii = 0; ii < MaxNumTopGroups; ++ii) {
        auto expertIdx = ii * WarpSize + laneIdx;
        expertIdxGroup[ii] = expertIdx;
        expertScoreGroup[ii] =
            expertIdx < params.mNumExperts ? smemScoreBias[expertIdx] : invalidScoreFloat;
      }
    }

    topk::reduceTopK(warp, topScores, topExperts, expertScoreGroup, expertIdxGroup,
                     /* minValue */ invalidScoreFloat, params.mTopK);

    // determine our lane's expert index and write to output
    int32_t expertIdx = 0;
#pragma unroll
    for (int ii = 0; ii < params.mTopK; ++ii) {  // bound of params.mTopK
      expertIdx = laneIdx == ii ? topExperts[ii] : expertIdx;
    }
    // determine whether our expert is local to this GPU
    auto localExpertIdx = expertIdx - params.mLocalExpertsStartIdx;
    auto isLocalExpert = localExpertIdx >= 0 && localExpertIdx < localExpertExtent &&
                         (localExpertIdx & params.mLocalExpertsStrideLog2) == 0;

    float scoreNorm = laneIdx < params.mTopK ? smemScoreSigmoid[expertIdx] : 0.F;
    auto redNorm = cg::reduce(warp, scoreNorm, cg::plus<float>{});
    auto finalScore = OutputT{scoreNorm * params.mRouteScale / redNorm};

    // write expert idx out already
    auto idxTopK = blockIdx.x * params.mTopK + laneIdx;
    if (laneIdx < params.mTopK && params.mPtrExpertIdx != nullptr) {
      PackedScoreIdx<OutputT> packedScore{static_cast<OutputT>(finalScore),
                                          static_cast<int16_t>(expertIdx)};
      params.mPtrExpertIdx[idxTopK] = packedScore;
    }

    if (laneIdx < params.mTopK && params.mPtrExpertWeights != nullptr) {
      params.mPtrExpertWeights[idxTopK] = finalScore;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
__global__ void __cluster_dims__(NumBlocksPerCluster, 1, 1) __launch_bounds__(NumThreads)
    routingIndicesClusterKernel(KernelParams params) {
  using OutputT = typename KernelParams::OutputT;

  int32_t const warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WarpSize, 0);
  int32_t const clusterBlockRank = blockIdx.x;

  //@todo: try to move it into routingPermutation
  // then wait on primary grid
  if constexpr (KernelParams::UsePdl) {
    cudaGridDependencySynchronize();
  }

  routingPermutation<KernelParams, OutputT, NumThreads, NumWarps, MaxNumTopExperts,
                     /*LoadExpertIdxFromGlobal=*/true>(params, nullptr, warpIdx, clusterBlockRank);
}
#else
__global__ void routingIndicesClusterKernel(KernelParams params) {
  assert(false && "routingIndicesClusterKernel is only supported on SM90+ architectures");
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
__global__ void __launch_bounds__(NumThreads) routingIndicesCoopKernel(KernelParams params) {
  // number of experts is bounded by number of threads
  __shared__ int32_t __attribute((aligned(128))) smemExpertCount[NumThreads];
  __shared__ int32_t __attribute((aligned(128))) smemExpertOffset[NumThreads];
  // needed for the exclusive sum of token offsets
  using Scan = cub::BlockScan<int32_t, NumThreads, cub::BLOCK_SCAN_WARP_SCANS>;
  __shared__ typename Scan::TempStorage tempStorage;
  // 64 elements -> 128+ registers. Above that we may start to see spilling to local memory.
  static constexpr int MaxExpandedIdxPerThread = 64;

  // Initialize grid.
  cg::grid_group grid = cg::this_grid();
  // Note: the following is more efficient than grid.block_index() because we don't use y and z.
  int32_t const gridBlockIdx = blockIdx.x;
  int32_t const gridThreadIdx = NumThreads * gridBlockIdx + threadIdx.x;
  int32_t const numBlocks = gridDim.x;
  int32_t const numThreadsPerGrid = numBlocks * NumThreads;

  int32_t const warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WarpSize, 0);

  auto expandedIdxSize = params.mNumTokens * params.mTopK;

  // pre-fill the counts with 0
  smemExpertCount[threadIdx.x] = 0;
  __syncthreads();

  // then wait on primary grid
  if constexpr (KernelParams::UsePdl) {
    cudaGridDependencySynchronize();
  }

  // each thread keeps has some number of "expanded indexes" assigned to it
  // for each of these, we keep the associated expert and offset within expert in registers
  int32_t expertIndexes[MaxExpandedIdxPerThread];
  int32_t expertOffsets[MaxExpandedIdxPerThread];
  auto localExpertExtent = params.mNumLocalExperts << params.mLocalExpertsStrideLog2;
  // In order to avoid a serialization LDG-ATOMS-LDG-ATOMS-..., we skip multiple iterations at a
  // time, and branch between a fast path without bound checks and a slow path with bound checks.
  int constexpr IterStride = 4;
  static_assert(MaxExpandedIdxPerThread % IterStride == 0);

  // Define a lambda to avoid code duplication in both branches.
  auto loopBody = [&](int ii, int expandedIdx) {
    int32_t expertIdx = params.mPtrExpertIdx[expandedIdx].idx;
    expertIndexes[ii] = expertIdx;
    // check whether this expert is local to our GPU at all and ignore if not
    auto localExpertIdx = expertIdx - params.mLocalExpertsStartIdx;
    auto isLocalExpert = localExpertIdx >= 0 && localExpertIdx < localExpertExtent &&
                         (localExpertIdx & params.mLocalExpertsStrideLog2) == 0;
    expertOffsets[ii] = isLocalExpert ? atomicAdd(smemExpertCount + expertIdx, 1) : 0;
  };

#pragma unroll
  for (int32_t ii0 = 0; ii0 < MaxExpandedIdxPerThread; ii0 += IterStride) {
    // Whether it's safe to do multiple iterations without bound checks.
    bool const takeFastPath = (ii0 + IterStride) * numThreadsPerGrid <= expandedIdxSize;
    if (takeFastPath) {
#pragma unroll
      for (int32_t jj = 0; jj < IterStride; jj++) {
        int const ii = ii0 + jj;
        auto expandedIdx = static_cast<int32_t>(gridThreadIdx) + ii * numThreadsPerGrid;
        loopBody(ii, expandedIdx);
      }
    } else {
      bool doBreak = false;
#pragma unroll
      for (int32_t jj = 0; jj < IterStride; jj++) {
        int const ii = ii0 + jj;
        auto expandedIdx = static_cast<int32_t>(gridThreadIdx) + ii * numThreadsPerGrid;
        if (expandedIdx >= expandedIdxSize) {
          doBreak = true;
          break;
        }
        loopBody(ii, expandedIdx);
      }
      if (doBreak) {
        break;
      }
    }
  }

  // Make histogram (token counts per expert) available to all threads in the block.
  __syncthreads();

  //
  // Each thread now represents one expert
  //

  // Add the local bin count to the common bin count and get a per-CTA offset.
  int32_t const localExpertCount = smemExpertCount[threadIdx.x];

  int32_t blockExpertOffset = 0;
  if (threadIdx.x < params.mNumExperts) {
    blockExpertOffset = atomicAdd(&params.mPtrExpertCounts[threadIdx.x], localExpertCount);
  }

  // Sync to wait for completion of the histogram reduction.
  grid.sync();

  // Get total count for this expert.
  int32_t count = (threadIdx.x < params.mNumExperts) ? params.mPtrExpertCounts[threadIdx.x] : 0;

  // Note: the scan is redundant in all CTAs, but doing it in only 1 CTA would be worse for latency.

  // Compute the runtime config for projections
  // Whether or not an expert is local is taken into account when smemExpertCount is computed
  // so we do not need to take it into account here.
  const int32_t numCta = divUpLog2<int32_t>(count, params.mPaddingLog2);
  int32_t ctaOffset;
  int32_t numNonExitingCtas;
  Scan(tempStorage).ExclusiveSum(numCta, ctaOffset, numNonExitingCtas);

  for (int32_t cta = gridBlockIdx; cta < numCta; cta += numBlocks) {
    const int32_t localExpertIdx =
        (threadIdx.x - params.mLocalExpertsStartIdx) >> params.mLocalExpertsStrideLog2;
    params.mPtrCtaIdxXyToBatchIdx[ctaOffset + cta] = localExpertIdx;
    params.mPtrCtaIdxXyToMnLimit[ctaOffset + cta] =
        min(mulLog2<int32_t>(ctaOffset + cta + 1, params.mPaddingLog2),
            mulLog2<int32_t>(ctaOffset, params.mPaddingLog2) + count);
  }

  // get the padded offset associated with this expert
  const int32_t offset = mulLog2<int32_t>(ctaOffset, params.mPaddingLog2);
  const int32_t permutedIdxSize = mulLog2<int32_t>(numNonExitingCtas, params.mPaddingLog2);

  // write out padded count
  if (gridBlockIdx == 0 && warpIdx == NumWarps - 1 && cute::elect_one_sync()) {
    params.mPtrPermutedIdxSize[0] = permutedIdxSize;
    params.mPtrNumNonExitingCtas[0] = numNonExitingCtas;
  }

  // write expert offsets to shared
  smemExpertOffset[threadIdx.x] = offset + blockExpertOffset;

  // make expert offsets available to all threads
  __syncthreads();

  // trigger the secondary kernel when using PDL
  // We can't do it earlier because FC1 depends on the mPtrCtaIdxXyToBatchIdx,
  // mPtrCtaIdxXyToMnLimit, mPtrNumNonExitingCtas and mPtrTotalNumPaddedTokens
  // TODO: this is not sufficient to ensure visibility in the next kernel!
  if constexpr (KernelParams::UsePdl) {
    cudaTriggerProgrammaticLaunchCompletion();
  }

// each thread has the same "expanded indexes" assigned to it as above
// at this point, we know the final offsets of experts and the offsets within
// experts, which allows writing the final index values
#pragma unroll
  for (int32_t ii = 0; ii < MaxExpandedIdxPerThread; ++ii) {
    auto expandedIdx = static_cast<int32_t>(gridThreadIdx) + ii * numThreadsPerGrid;
    if (expandedIdx >= expandedIdxSize) {
      break;
    }
    auto expertIdx = expertIndexes[ii];
    // check whether this expert is local to our GPU at all
    auto localExpertIdx = static_cast<int32_t>(expertIdx) - params.mLocalExpertsStartIdx;
    auto isLocalExpert = localExpertIdx >= 0 && localExpertIdx < localExpertExtent &&
                         (localExpertIdx & params.mLocalExpertsStrideLog2) == 0;
    auto tokenIdx = expandedIdx / params.mTopK;
    auto permutedIdx =
        isLocalExpert ? int32_t{smemExpertOffset[expertIdx]} + expertOffsets[ii] : int32_t{-1};
    if (params.mPtrExpandedIdxToPermutedIdx != nullptr) {
      params.mPtrExpandedIdxToPermutedIdx[expandedIdx] = permutedIdx;
    }
    if (params.mPtrPermutedIdxToTokenIdx != nullptr && isLocalExpert) {
      params.mPtrPermutedIdxToTokenIdx[permutedIdx] = tokenIdx;
    }
  }
}
#else
__global__ void routingIndicesCoopKernel(KernelParams params) {
  assert(false && "routingIndicesCoopKernel is only supported on SM90+ architectures");
}
#endif
// VERBATIM END
// VERBATIM BEGIN csrc/trtllm_fused_moe_routing_deepseek.cu:533-534
}  // namespace routingDeepSeek
}  // namespace moe::dev::routing
// VERBATIM END
