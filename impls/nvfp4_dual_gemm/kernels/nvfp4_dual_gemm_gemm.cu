// SPDX-License-Identifier: Apache-2.0
// Single-kernel translation unit: exactly one explicit instantiation of
// cutlass::device_kernel for the FP32-output NVFP4 GemmKernel and no host
// launch code. Compiled with `nvcc -cubin` into
// impls/nvfp4_dual_gemm/cubins/<arch>/nvfp4_dual_gemm_gemm.cubin.
#include "nvfp4_dual_gemm_sm100.cuh"

template __global__ void cutlass::device_kernel<nvfp4_dual_gemm::GemmKernel>(
    CUTLASS_GRID_CONSTANT nvfp4_dual_gemm::GemmKernel::Params const);
