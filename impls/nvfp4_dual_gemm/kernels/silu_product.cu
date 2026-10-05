// SPDX-License-Identifier: Apache-2.0
// Single-kernel translation unit: the previous package's silu_product glue
// kernel (impls/templates/nvfp4.cu), verbatim. out = half(silu(a) * b) over
// `size` contiguous FP32 elements.
#include <cuda_fp16.h>

__global__ void silu_product(const float* a, const float* b, half* output, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) output[i] = __float2half(a[i] / (1.f + expf(-a[i])) * b[i]);
}
