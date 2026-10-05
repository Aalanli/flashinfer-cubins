// SPDX-License-Identifier: Apache-2.0
// Single-kernel translation unit: exactly one explicit instantiation of
// cutlass::device_kernel for the fp8_gemm_small_m GemmKernel and no host launch
// code. Compiled with `nvcc -cubin` into
// impls/fp8_gemm/cubins/<arch>/fp8_gemm_small_m.cubin.
#include "fp8_gemm_small_m_sm100.cuh"

template __global__ void cutlass::device_kernel<fp8_gemm_small_m::GemmKernel>(
    CUTLASS_GRID_CONSTANT fp8_gemm_small_m::GemmKernel::Params const);
