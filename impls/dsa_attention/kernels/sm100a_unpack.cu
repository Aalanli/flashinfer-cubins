// DSA attention, sm_100a stage 3/3: compact the 16 live heads out of the padded 128-head
// CUTLASS MLA output, convert natural-log LSE to base 2 and normalize empty rows.
//
// Project-local adapter kernel (moved verbatim from impls/templates/dsa_attention_sm100a.cu);
// not upstream code. Uses cutlass::bfloat16_t (CUTLASS, BSD-3-Clause, see ../CUTLASS_LICENSE)
// only as the element type. Launch: grid = batch, block = 256.
#include <cutlass/numeric_types.h>

using DsaElement = cutlass::bfloat16_t;

__global__ void dsa_unpack(const DsaElement* padded, const float* padded_lse,
                          const int* ids, DsaElement* out, float* lse) {
  int row = blockIdx.x;
  __shared__ int nonempty;
  if (threadIdx.x == 0) {
    nonempty = 0;
    for (int i=0; i<2048; ++i) nonempty |= ids[row*2048+i] >= 0;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < 16*512; i += blockDim.x)
    out[row*16*512+i] = nonempty ? padded[row*128*512+i] : DsaElement(0.f);
  if (threadIdx.x < 16)
    lse[row*16+threadIdx.x] = nonempty ? padded_lse[row*128+threadIdx.x]*1.4426950408889634f : -__int_as_float(0x7f800000);
}
