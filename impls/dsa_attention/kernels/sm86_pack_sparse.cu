// DSA attention, sm_86 stage 1/3: pack sparse top-k token IDs into FlashInfer CSR pages.
//
// Project-local adapter kernel (moved verbatim from impls/templates/dsa_attention.cu);
// not upstream FlashInfer code. Each token ID is reinterpreted as a size-1 page.
// The kernel is deliberately serial (<<<1, 1>>>): it is metadata preparation only.
//
// `mem` layout (int32 elements, capacity = 1024 query tokens):
//   [0, capacity + 1)           indptr          (batch + 1 written)
//   [capacity + 1, +capacity)   last_page_len   (batch written)
//   [.., +capacity)             request_indices (batch written)
//   [.., +capacity)             kv_tile_indices (batch written)
//   [.., +1)                    kv_chunk_size   (= 2048)
//   [.., +capacity * 2048)      indices         (indptr[batch] written)
constexpr int capacity = 1024;

__global__ void pack_sparse(const int* in, int* mem, int batch) {
  // Metadata preparation only; all attention arithmetic remains upstream code.
  int* indptr = mem;
  int* last = mem + capacity + 1;
  int* requests = last + capacity;
  int* tiles = requests + capacity;
  int* chunk = tiles + capacity;
  int* indices = chunk + 1;
  int offset = 0;
  indptr[0] = 0; *chunk = 2048;
  for (int t = 0; t < batch; ++t) {
    const int start = offset;
    for (int j = 0; j < 2048; ++j) {
      int index = in[t * 2048 + j];
      if (index >= 0) indices[offset++] = index;
    }
    indptr[t + 1] = offset;
    last[t] = offset > start ? 1 : 0;
    requests[t] = t; tiles[t] = 0;
  }
}
