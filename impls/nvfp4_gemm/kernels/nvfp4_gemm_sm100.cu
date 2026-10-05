// SPDX-License-Identifier: Apache-2.0
// Single-kernel translation unit: exactly one explicit instantiation of
// cutlass::device_kernel for the nvfp4_gemm GemmKernel (FlashInfer's
// DeviceGemmFp4GemmSm100_half_128_128_256_1_1_1_1SM) and no host launch code.
// Compiled with `nvcc -cubin` into impls/nvfp4_gemm/cubins/<arch>/nvfp4_gemm.cubin.
#include "nvfp4_gemm_sm100.cuh"

template __global__ void cutlass::device_kernel<nvfp4_gemm::GemmKernel>(
    CUTLASS_GRID_CONSTANT nvfp4_gemm::GemmKernel::Params const);
