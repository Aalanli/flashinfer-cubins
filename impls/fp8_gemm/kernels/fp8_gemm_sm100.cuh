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
// Kernel type of FlashInfer's CutlassGroupwiseScaledGEMMSM100 as the fp8_gemm
// package instantiates it: <ScaleGranularityM=1, ScaleGranularityN=128,
// ScaleGranularityK=128, ScaleMajorK=true, MmaSM=1, DTypeIn=float_e4m3_t,
// DTypeOut=bfloat16_t>.
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

namespace fp8_gemm {

using namespace cute;

constexpr int ScaleGranularityM = 1;
constexpr int ScaleGranularityN = 128;
constexpr int ScaleGranularityK = 128;
constexpr bool ScaleMajorK = true;
constexpr int MmaSM = 1;
using DTypeIn = cutlass::float_e4m3_t;
using DTypeOut = cutlass::bfloat16_t;

using ElementA = DTypeIn;
using LayoutA = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

using ElementB = DTypeIn;
using LayoutB = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

using ElementC = DTypeOut;
using LayoutC = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

using ElementD = ElementC;
using LayoutD = LayoutC;
constexpr int AlignmentD = AlignmentC;

using ElementAccumulator = float;
using ElementCompute = float;

using MmaTileShape_MNK = Shape<cute::Int<MmaSM * 128>, _128, _128>;
using ClusterShape_MNK = Shape<cute::Int<MmaSM>, _1, _1>;

using ScaleConfig = std::conditional_t<
    ScaleMajorK,
    cutlass::detail::Sm100BlockwiseScaleConfig<ScaleGranularityM, ScaleGranularityN,
                                               ScaleGranularityK, UMMA::Major::K, UMMA::Major::K>,
    cutlass::detail::Sm100BlockwiseScaleConfig<ScaleGranularityM, ScaleGranularityN,
                                               ScaleGranularityK, UMMA::Major::MN,
                                               UMMA::Major::MN>>;

using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute, ElementC,
    LayoutC, AlignmentC, ElementD, LayoutC, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, ElementA,
    cute::tuple<LayoutA, LayoutSFA>, AlignmentA, ElementB, cute::tuple<LayoutB, LayoutSFB>,
    AlignmentB, ElementAccumulator, MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelScheduleSm100Blockwise>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue,
    void>;  // Default to ClusterLaunchControl (CLC) based tile scheduler

}  // namespace fp8_gemm
