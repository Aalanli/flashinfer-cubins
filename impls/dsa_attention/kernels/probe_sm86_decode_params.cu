// Host-only probe for the sm_86 FlashInfer MLA decode kernel parameter struct.
//
// Builds flashinfer::BatchDecodeParamsMLA exactly as the previous native host code
// did (impls/templates/dsa_attention.cu) for a sample batch size, using fake pointers,
// and prints sizeof/offsets/raw bytes as JSON. Needs no GPU.
// Usage: probe <batch> <sm_scale>
#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/attention/decode.cuh>

#include "probe_common.h"

int main(int argc, char** argv) {
  using namespace flashinfer;
  using dsa_probe::field;
  using dsa_probe::sentinel;
  using T = __nv_bfloat16;
  using Params = BatchDecodeParamsMLA<T, T, T, int>;
  if (argc != 3) return 2;
  const int batch = std::atoi(argv[1]);
  const float sm_scale = std::strtof(argv[2], nullptr);

  // Pointer roles (sentinel index): 0 q_nope, 1 q_pe, 2 ckv, 3 kpe, 4 o, 5 lse,
  // 6 indices, 7 indptr, 8 last_page_len, 9 request_indices, 10 kv_tile_indices,
  // 11 kv_chunk_size.
  alignas(Params) unsigned char storage[sizeof(Params)];
  std::memset(storage, 0, sizeof(storage));
  paged_kv_mla_t<T, int> pages(1, 512, 64, batch, (T*)sentinel(2), (T*)sentinel(3),
                               (int*)sentinel(6), (int*)sentinel(7), (int*)sentinel(8));
  auto* p = new (storage) Params((T*)sentinel(0), (T*)sentinel(1), nullptr, pages,
                                 (T*)sentinel(4), (float*)sentinel(5), 16, -1, 0.f, sm_scale,
                                 1.f, 1e4f);
  p->padded_batch_size = batch;
  p->request_indices = (int*)sentinel(9);
  p->kv_tile_indices = (int*)sentinel(10);
  p->kv_chunk_size_ptr = (int*)sentinel(11);

  auto& q = *p;
  field("q_nope", "ptr", q, q.q_nope);
  field("q_pe", "ptr", q, q.q_pe);
  field("o", "ptr", q, q.o);
  field("lse", "ptr", q, q.lse);
  field("sm_scale", "f32", q, q.sm_scale);
  field("q_rope_offset", "ptr", q, q.q_rope_offset);
  field("paged_kv.page_size", "bytes", q, q.paged_kv.page_size);
  field("paged_kv.head_dim_ckv", "u32", q, q.paged_kv.head_dim_ckv);
  field("paged_kv.head_dim_kpe", "u32", q, q.paged_kv.head_dim_kpe);
  field("paged_kv.batch_size", "u32", q, q.paged_kv.batch_size);
  field("paged_kv.stride_page_ckv", "u32", q, q.paged_kv.stride_page_ckv);
  field("paged_kv.stride_page_kpe", "u32", q, q.paged_kv.stride_page_kpe);
  field("paged_kv.stride_n_ckv", "u32", q, q.paged_kv.stride_n_ckv);
  field("paged_kv.stride_n_kpe", "u32", q, q.paged_kv.stride_n_kpe);
  field("paged_kv.ckv_data", "ptr", q, q.paged_kv.ckv_data);
  field("paged_kv.kpe_data", "ptr", q, q.paged_kv.kpe_data);
  field("paged_kv.indices", "ptr", q, q.paged_kv.indices);
  field("paged_kv.indptr", "ptr", q, q.paged_kv.indptr);
  field("paged_kv.last_page_len", "ptr", q, q.paged_kv.last_page_len);
  field("paged_kv.rope_pos_offset", "ptr", q, q.paged_kv.rope_pos_offset);
  field("padded_batch_size", "u32", q, q.padded_batch_size);
  field("num_qo_heads", "u32", q, q.num_qo_heads);
  field("window_left", "i32", q, q.window_left);
  field("logits_soft_cap", "f32", q, q.logits_soft_cap);
  field("rope_rcp_scale", "f32", q, q.rope_rcp_scale);
  field("rope_rcp_theta", "f32", q, q.rope_rcp_theta);
  field("request_indices", "ptr", q, q.request_indices);
  field("kv_tile_indices", "ptr", q, q.kv_tile_indices);
  field("o_indptr", "ptr", q, q.o_indptr);
  field("kv_chunk_size_ptr", "ptr", q, q.kv_chunk_size_ptr);
  field("block_valid_mask", "ptr", q, q.block_valid_mask);
  field("partition_kv", "bool", q, q.partition_kv);

  std::printf("{\n  \"param_size\": %zu,\n", sizeof(Params));
  // Dynamic shared memory requested by the previous host launch.
  constexpr int smem = 2 * 8 * (512 + 64) * sizeof(T) + 256 * sizeof(size_t) * 2;
  std::printf("  \"constants\": {\"shared_mem\": %d, \"block\": [32, 8, 1], "
              "\"page_size_fastdiv_1\": \"%s\"},\n",
              smem, dsa_probe::hex(&q.paged_kv.page_size, sizeof(q.paged_kv.page_size)).c_str());
  dsa_probe::print_fields();
  std::printf(",\n  \"example\": {\"batch\": %d, \"sm_scale\": %.9g, \"bytes\": \"%s\"}\n}\n",
              batch, sm_scale, dsa_probe::hex(storage, sizeof(storage)).c_str());
  return 0;
}
