// GQA paged decode, sm_100a stage 2/4: gather page-size-1 K/V into ragged contiguous rows.
//
// Project-local adapter kernel (moved verbatim from impls/templates/gqa_decode_sm100a.cu); not
// upstream code. Uses cutlass::bfloat16_t (CUTLASS, BSD-3-Clause, see ../CUTLASS_LICENSE) only
// as the element type. Request b's pages land at rows [kv[b], kv[b] + max(1, len)) of the
// packed [rows, 4 KV heads, 128] K and V; an empty request gets one zero row.
// Launch: grid = batch, block = 256.
#include <cutlass/numeric_types.h>

using GqaElement = cutlass::bfloat16_t;

__global__ void gqa_gather(const GqaElement* k, const GqaElement* v, const int* indptr,
    const int* indices, const int* kv, GqaElement* packed_k, GqaElement* packed_v) {
  int b = blockIdx.x, len = indptr[b+1]-indptr[b];
  for (int64_t i=threadIdx.x; i<int64_t(max(1,len))*512; i+=blockDim.x) {
    int64_t src = len ? int64_t(indices[indptr[b]+i/512])*512+i%512 : 0;
    int64_t dst = int64_t(kv[b])*512+i;
    packed_k[dst] = len ? k[src] : GqaElement(0.f);
    packed_v[dst] = len ? v[src] : GqaElement(0.f);
  }
}
