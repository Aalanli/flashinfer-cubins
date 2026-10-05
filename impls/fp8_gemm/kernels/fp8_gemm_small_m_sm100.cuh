/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Kernel type of FlashInfer's CutlassGroupwiseScaledGEMMSM100LowLatency as
// FlashInfer's own dispatcher (csrc/gemm_groupwise_sm100.cu,
// CutlassGemmGroupwiseScaledSM100) instantiates it for m <= 32:
// <ScaleGranularityM=1, ScaleGranularityN=128, ScaleGranularityK=128,
// ScaleMajorK=true, MmaSM=1 (unused by this kernel), DTypeIn=float_e4m3_t,
// DTypeOut=bfloat16_t>.
//
// The kernel computes the transposed problem D^T[n, m] = B[n, k] @ A[m, k]^T:
// upstream swaps (m, n), (A, B), (SFA, SFB) and the M/N scale granularities
// before building the kernel arguments, and stores D column-major (n, m),
// which is the original row-major (m, n) memory.
//
// The type definitions below are copied verbatim (modulo the template
// parameters being fixed) from
//   flashinfer@aa7c67f2b876b89be34c7a70ac022369a56e60d5
//   include/flashinfer/gemm/gemm_groupwise_sm100.cuh
// where they are local to the host function template. Only the kernel type is
// needed here: the host-side argument setup lives in Python
// (harness/workloads/fp8_gemm.py) and the probe (fp8_gemm_probe.cu).
#pragma once

#include <type_traits>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/util/packed_stride.hpp"

namespace fp8_gemm_small_m {

using namespace cute;

// Granularities of the original (unswapped) problem, as dispatched.
constexpr int ScaleGranularityM = 1;
constexpr int ScaleGranularityN = 128;
constexpr int ScaleGranularityK = 128;
constexpr bool ScaleMajorK = true;
using DTypeIn = cutlass::float_e4m3_t;
using DTypeOut = cutlass::bfloat16_t;

// Do the swap here as well
using ScaleConfig = std::conditional_t<
    ScaleMajorK,
    cutlass::detail::Sm100BlockwiseScaleConfig<ScaleGranularityN, ScaleGranularityM,
                                               ScaleGranularityK, UMMA::Major::K, UMMA::Major::K>,
    cutlass::detail::Sm100BlockwiseScaleConfig<ScaleGranularityN, ScaleGranularityM,
                                               ScaleGranularityK, UMMA::Major::MN,
                                               UMMA::Major::MN>>;

using ElementA = DTypeIn;
using LayoutA = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

using ElementB = DTypeIn;
using LayoutB = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

using ElementC = DTypeOut;
using LayoutC = cutlass::layout::ColumnMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

using ElementD = ElementC;
using LayoutD = LayoutC;
constexpr int AlignmentD = AlignmentC;

// MMA type
using ElementAccumulator = float;  // Element Accumulator will also be our scale factor type
using ElementCompute = float;

using MmaTileShape_MNK = Shape<cute::Int<128>, _16, _128>;
using ClusterShape_MNK = Shape<int, int, _1>;

using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto, float, float, void, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD, cutlass::epilogue::TmaWarpSpecialized1Sm,
    cutlass::epilogue::fusion::LinearCombination<ElementD, float, void, float>>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    /*ArchTag=*/cutlass::arch::Sm100, /*OpClass=*/cutlass::arch::OpClassTensorOp, ElementA,
    /*GemmLayoutA=*/cute::tuple<LayoutA, LayoutSFA>, AlignmentA, ElementB,
    /*GemmLayoutB=*/cute::tuple<LayoutB, LayoutSFB>, AlignmentB, ElementAccumulator,
    /*TileShapeMNK=*/MmaTileShape_MNK, ClusterShape_MNK,
    /*StageCountType=*/
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename CollectiveEpilogue::SharedStorage))>,
    /*KernelScheduleType=*/cutlass::gemm::KernelTmaWarpSpecializedBlockwise1SmSm100>::
    CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    /*ProblemShapeOrThreadblockMma_=*/Shape<int, int, int, int>, CollectiveMainloop,
    CollectiveEpilogue, void>;

}  // namespace fp8_gemm_small_m
