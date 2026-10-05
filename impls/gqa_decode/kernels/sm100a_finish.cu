// GQA paged decode, sm_100a stage 4/4: copy the FMHA output and normalize empty requests.
//
// Project-local adapter kernel (moved verbatim from impls/templates/gqa_decode_sm100a.cu); not
// upstream code. Uses cutlass::bfloat16_t (CUTLASS, BSD-3-Clause, see ../CUTLASS_LICENSE) only
// as the element type. `padded` is the FMHA output (row b of request b); `lse` is updated in
// place (base-2 -inf for empty requests, otherwise left as the FMHA wrote it).
// Launch: grid = batch, block = 256.
#include <cutlass/numeric_types.h>

using GqaElement = cutlass::bfloat16_t;

__global__ void gqa_finish(const GqaElement* padded, const int* indptr, GqaElement* out, float* lse) {
  int b = blockIdx.x;
  bool empty = indptr[b] == indptr[b+1];
  for (int i=threadIdx.x; i<32*128; i+=blockDim.x)
    out[b*32*128+i] = empty ? GqaElement(0.f) : padded[b*32*128+i];
  if (empty && threadIdx.x < 32) lse[b*32+threadIdx.x] = -__int_as_float(0x7f800000);
}
