// DSA attention, sm_86 stage 3/3: normalize rows whose sparse index list is empty.
//
// Project-local adapter kernel (moved verbatim from impls/templates/dsa_attention.cu);
// not upstream FlashInfer code. The FlashInfer decode kernel leaves undefined values
// for zero-length requests; this writes output 0 and base-2 LSE -inf for them in place.
// Launch: grid = batch, block = 256. Shapes are fixed at 16 heads x 512 latent dims.
#include <cuda_bf16.h>

__global__ void fix_empty(const int* indptr, __nv_bfloat16* output, float* lse) {
  int t = blockIdx.x;
  if (indptr[t] != indptr[t + 1]) return;
  for (int i = threadIdx.x; i < 16 * 512; i += blockDim.x) output[t * 16 * 512 + i] = __float2bfloat16(0.f);
  if (threadIdx.x < 16) lse[t * 16 + threadIdx.x] = -__int_as_float(0x7f800000);
}
