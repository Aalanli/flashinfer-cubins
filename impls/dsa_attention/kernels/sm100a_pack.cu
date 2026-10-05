// DSA attention, sm_100a stage 1/3: pad 16 query heads to the 128-head CUTLASS MLA tile
// and compact sparse top-k token IDs into a page-size-1 page table.
//
// Project-local adapter kernel (moved verbatim from impls/templates/dsa_attention_sm100a.cu);
// not upstream code. Uses cutlass::bfloat16_t (CUTLASS, BSD-3-Clause, see ../CUTLASS_LICENSE)
// only as the element type. Launch: grid = batch, block = 256.
#include <cutlass/numeric_types.h>

using DsaElement = cutlass::bfloat16_t;

__global__ void dsa_pack(const DsaElement* q, const DsaElement* qp, const int* ids,
                        DsaElement* padded_q, DsaElement* padded_qp, int* table, int* lengths) {
  int row = blockIdx.x;
  for (int i = threadIdx.x; i < 128*512; i += blockDim.x)
    padded_q[row*128*512+i] = i < 16*512 ? q[row*16*512+i] : DsaElement(0.f);
  for (int i = threadIdx.x; i < 128*64; i += blockDim.x)
    padded_qp[row*128*64+i] = i < 16*64 ? qp[row*16*64+i] : DsaElement(0.f);
  if (threadIdx.x == 0) {
    int n = 0;
    for (int i = 0; i < 2048; ++i) if (ids[row*2048+i] >= 0) table[row*2048+n++] = ids[row*2048+i];
    // A dummy valid token keeps the upstream empty softmax path well-defined.
    lengths[row] = n ? n : 1;
    for (int i = n; i < 2048; ++i) table[row*2048+i] = 0;
  }
}
