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
// Kernel type of FlashInfer's NVFP4 CUTLASS GEMM as the nvfp4_gemm package
// instantiates it: INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(half, 128, 128, 256,
// 1, 1, 1, _1SM), i.e. E2M1 A/B with UE4M3 16-element block scales, FP16 D,
// 128x128x256 MMA tile, dynamic cluster (launched as 1x1x1), CLC tile
// scheduler (cutlass::gemm::PersistentScheduler).
//
// The definitions below are copied verbatim from
//   flashinfer@aa7c67f2b876b89be34c7a70ac022369a56e60d5
//   include/flashinfer/gemm/fp4_gemm_template_sm100.h
// (SMTypeAdapter<_1SM>, kGenericFp4OutputAlignmentBits,
// GenericFp4FusionOperation and the DeviceGemmFp4GemmSm100_##... struct the
// macro defines, with its macro parameters substituted), keeping the
// namespace and struct name so the kernel symbol is the one of the previous
// package. Only `flashinfer::cutlass_dtype<half>::type` is spelled out as its
// value `cutlass::half_t` (flashinfer/cutlass_utils.cuh), and the
// GemmUniversalAdapter alias, host-side launcher and argument preparation are
// omitted: the host-side setup lives in Python (harness/workloads/nvfp4_gemm.py)
// and the compile-time probe (nvfp4_gemm_probe.cu).
#pragma once

#include <type_traits>

#include "cute/tensor.hpp"
#include "cutlass/arch/arch.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/numeric_types.h"
#include "flashinfer/arch_condition.h"

namespace flashinfer {
namespace gemm {
using namespace cute;

struct _1SM {};

template <typename T>
struct SMTypeAdapter {};

template <>
struct SMTypeAdapter<_1SM> {
  static int const Scale = 1;
  using AtomThrShape = cute::Shape<_1, _1, _1>;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized1Sm;
  using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized1SmNvf4Sm100;
};

inline constexpr int kGenericFp4OutputAlignmentBits = 128;

template <class ElementOutput, class ElementCompute, class ElementSource, class ElementScalar>
using GenericFp4FusionOperation =
    cutlass::epilogue::fusion::LinearCombination<ElementOutput, ElementCompute, ElementSource,
                                                 ElementScalar>;

// INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(half, 128, 128, 256, 1, 1, 1, _1SM)
struct DeviceGemmFp4GemmSm100_half_128_128_256_1_1_1_1SM {
  using OutElementType = cutlass::half_t;  // flashinfer::cutlass_dtype<half>::type
  using CTAShape = cute::Shape<cute::Int<128>, cute::Int<128>, cute::Int<256>>;
  /*using ClusterShape = cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>;*/
  using ClusterShape = cute::Shape<int, int, _1>;
  using ElementType = cutlass::float_e2m1_t;
  using Arch = cutlass::arch::Sm100;
  /* // Input A */
  using ElementA = ElementType;
  using LayoutA = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementType>::value;
  /* // Input B */
  using ElementB = ElementType;
  using LayoutB = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementType>::value;
  /* // Input C */
  using ElementC = void;
  using LayoutC = cutlass::layout::RowMajor;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<OutElementType>::value;
  static constexpr int AlignmentD =
      kGenericFp4OutputAlignmentBits / cutlass::sizeof_bits<OutElementType>::value;

  using SFType = cutlass::float_ue4m3_t;
  using ElementCompute = float;
  using ElementAccumulator = float;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using EpilogueTileType = std::conditional_t<128 == 128 && 128 == 256 && 256 == 256,
                                              cute::Shape<cute::_128, cute::_64>,
                                              cutlass::epilogue::collective::EpilogueTileAuto>;
  using EpilogueSchedule = SMTypeAdapter<_1SM>::EpilogueSchedule;
  using MainloopSchedule = SMTypeAdapter<_1SM>::MainloopSchedule;
  using MmaTileShape = cute::Shape<cute::Int<128 * SMTypeAdapter<_1SM>::Scale>, cute::Int<128>,
                                   cute::Int<256>>;
  using FusionOperation = GenericFp4FusionOperation<OutElementType, float, void, float>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      Arch, OperatorClass, MmaTileShape, ClusterShape, EpilogueTileType, ElementAccumulator,
      ElementCompute, ElementC, LayoutC, AlignmentC, OutElementType, LayoutC, AlignmentD,
      EpilogueSchedule, FusionOperation>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      Arch, cutlass::arch::OpClassBlockScaledTensorOp, cute::tuple<ElementA, SFType>, LayoutA,
      AlignmentA, cute::tuple<ElementB, SFType>, LayoutB, AlignmentB, ElementAccumulator,
      MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainloopSchedule>::CollectiveOp;

  template <typename Base>
  struct Sm10x11xOnly : Base {
    using typename Base::Params;
    CUTLASS_DEVICE
    void operator()(Params const& params, char* smem_buf) {
      if constexpr (flashinfer::arch::is_major_v<10> || flashinfer::arch::is_major_v<11>) {
        this->Base::operator()(params, smem_buf);
      } else {
        if (cute::thread0()) {
          printf("%s : This kernel shall only run on SM10x and SM11x devices.\n",
                 __PRETTY_FUNCTION__);
          __trap();
        }
      }
    }
  };
  using GemmKernel =
      Sm10x11xOnly<cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>,
                                                        CollectiveMainloop, CollectiveEpilogue,
                                                        cutlass::gemm::PersistentScheduler>>;
};

}  // namespace gemm
}  // namespace flashinfer

namespace nvfp4_gemm {
using Device = flashinfer::gemm::DeviceGemmFp4GemmSm100_half_128_128_256_1_1_1_1SM;
using GemmKernel = Device::GemmKernel;
}  // namespace nvfp4_gemm
