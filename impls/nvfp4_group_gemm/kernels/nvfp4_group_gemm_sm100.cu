// SPDX-License-Identifier: Apache-2.0
// Single-kernel translation unit: exactly one explicit instantiation of
// cutlass::device_kernel for the grouped NVFP4 GemmKernel and no host launch
// code. Compiled with `nvcc -cubin` into
// impls/nvfp4_group_gemm/cubins/<arch>/nvfp4_group_gemm.cubin.
#include "nvfp4_group_gemm_sm100.cuh"

template __global__ void cutlass::device_kernel<nvfp4_group_gemm::GemmKernel>(
    CUTLASS_GRID_CONSTANT nvfp4_group_gemm::GemmKernel::Params const);
