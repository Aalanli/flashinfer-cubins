/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Kernel type of the nvfp4_group_gemm workload: one grouped (ptr-array)
// CUTLASS SM100 NVFP4 GEMM launch for all groups.
//
// The configuration is FlashInfer's single-GEMM NVFP4 kernel as the previous
// package instantiated it,
//   flashinfer@aa7c67f2b876b89be34c7a70ac022369a56e60d5
//   include/flashinfer/gemm/fp4_gemm_template_sm100.h
//   INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(half, 128, 128, 256, 1, 1, 1, _1SM)
// (type aliases of DeviceGemmFp4GemmSm100_half_128_128_256_1_1_1_1SM copied
// with the template parameters fixed), turned into CUTLASS's grouped GEMM the
// way CUTLASS's own example does it,
//   cutlass@b46b16d003484063bca4ed365e44095c4c6ed633
//   examples/75_blackwell_grouped_gemm/75_blackwell_grouped_gemm_block_scaled.cu
// i.e. with exactly these changes:
//   * problem shape GroupProblemShape<Shape<int,int,int>> (one (M,N,K) per group)
//     instead of Shape<int,int,int,int>;
//   * pointer layouts (LayoutA*, LayoutB*, LayoutC*) so that A/B/SFA/SFB/D
//     pointers, strides and scale-factor layouts are per-group device arrays;
//   * the ptr-array schedules KernelPtrArrayTmaWarpSpecialized1SmNvf4Sm100 and
//     PtrArrayTmaWarpSpecialized1Sm instead of KernelTmaWarpSpecialized1SmNvf4Sm100
//     and TmaWarpSpecialized1Sm (GemmUniversal then selects CUTLASS's SM100
//     group tile scheduler, replacing FlashInfer's PersistentScheduler tag);
//   * a static 1x1x1 cluster (the value FlashInfer passed at run time for its
//     dynamic Shape<int,int,_1> cluster), so no cluster launch attribute or
//     fallback cluster is involved;
//   * no Sm10x11xOnly wrapper (the cubin is built for sm_100a only).
// Unchanged: FP4 E2M1 A (row major) and B (column major), UE4M3 scale factors
// per 16 elements, FP32 accumulation, 128x128x256 MMA tile, 1-SM MMA,
// automatic stage count with the epilogue carve-out, automatic epilogue tile,
// sourceless (void C) LinearCombination epilogue storing FP16 D.
#pragma once

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/util/packed_stride.hpp"

namespace nvfp4_group_gemm {

using namespace cute;

constexpr int CTA_M = 128, CTA_N = 128, CTA_K = 256;

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;  // (M, N, K) per group
using OutElementType = cutlass::half_t;
using CTAShape = Shape<Int<CTA_M>, Int<CTA_N>, Int<CTA_K>>;
using ClusterShape = Shape<_1, _1, _1>;
using ElementType = cutlass::float_e2m1_t;
using Arch = cutlass::arch::Sm100;

using ElementA = ElementType;
using LayoutA = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementType>::value;

using ElementB = ElementType;
using LayoutB = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementType>::value;

using ElementC = void;
using LayoutC = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<OutElementType>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<OutElementType>::value;

using SFType = cutlass::float_ue4m3_t;
using ElementCompute = float;
using ElementAccumulator = float;
using OperatorClass = cutlass::arch::OpClassTensorOp;
using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
using MainloopSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmNvf4Sm100;
using MmaTileShape = Shape<Int<CTA_M>, Int<CTA_N>, Int<CTA_K>>;
using FusionOperation =
    cutlass::epilogue::fusion::LinearCombination<OutElementType, float, void, float>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    Arch, OperatorClass, MmaTileShape, ClusterShape, EpilogueTileType, ElementAccumulator,
    ElementCompute, ElementC, LayoutC*, AlignmentC, OutElementType, LayoutC*, AlignmentD,
    EpilogueSchedule, FusionOperation>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    Arch, cutlass::arch::OpClassBlockScaledTensorOp, cute::tuple<ElementA, SFType>, LayoutA*,
    AlignmentA, cute::tuple<ElementB, SFType>, LayoutB*, AlignmentB, ElementAccumulator,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename CollectiveEpilogue::SharedStorage))>,
    MainloopSchedule>::CollectiveOp;

using GemmKernel =
    cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;

}  // namespace nvfp4_group_gemm
