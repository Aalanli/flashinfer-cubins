// Host-only probe for the sm_86 FlashInfer decode kernel parameter struct
// (flashinfer::BatchDecodeParams<bf16, bf16, bf16, int>, passed by value).
//
// Builds the struct exactly as the previous native host code did
// (impls/templates/gqa_decode.cu) for a sample batch size and scale, with fake pointers, and
// prints sizeof/offsets/raw bytes plus the launch constants as JSON. Needs no GPU.
// Usage: probe <batch> <sm_scale>
#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/attention/decode.cuh>

#include "probe_common.h"

int main(int argc, char** argv) {
  using namespace flashinfer;
  using gqa_probe::scalar;
  using gqa_probe::sentinel;
  using T = __nv_bfloat16;
  using Params = BatchDecodeParams<T, T, T, int>;
  if (argc != 3) return 2;
  const int batch = std::atoi(argv[1]);
  const float sm_scale = std::strtof(argv[2], nullptr);
  constexpr uint32_t num_qo_heads = 32, num_kv_heads = 4, head_dim = 128, page_size = 1;
  constexpr int q_stride_n = num_qo_heads * head_dim, q_stride_h = head_dim;

  // Pointer roles (sentinel index): 0 q, 1 k_cache, 2 v_cache, 3 kv_indices, 4 kv_indptr,
  // 5 last_page_len, 6 o, 7 lse, 8 request_indices, 9 kv_tile_indices, 10 kv_chunk_size.
  alignas(Params) unsigned char storage[sizeof(Params)];
  std::memset(storage, 0, sizeof(storage));
  paged_kv_t<T, int> pages(num_kv_heads, page_size, head_dim, batch, QKVLayout::kNHD,
                           (T*)sentinel(1), (T*)sentinel(2), (int*)sentinel(3), (int*)sentinel(4),
                           (int*)sentinel(5));
  auto* p = new (storage) Params((T*)sentinel(0), nullptr, pages, (T*)sentinel(6),
                                 (float*)sentinel(7), nullptr, num_qo_heads, q_stride_n,
                                 q_stride_h, -1, 0.f, sm_scale, 1.f, 1e4f);
  p->padded_batch_size = batch;
  p->request_indices = (int*)sentinel(8);
  p->kv_tile_indices = (int*)sentinel(9);
  p->kv_chunk_size_ptr = (int*)sentinel(10);

  auto& q = *p;
  auto& kv = q.paged_kv;
  scalar("q", q, q.q);
  scalar("q_rope_offset", q, q.q_rope_offset);
  gqa_probe::field("paged_kv.page_size", "bytes", q, kv.page_size);
  scalar("paged_kv.num_heads", q, kv.num_heads);
  scalar("paged_kv.head_dim", q, kv.head_dim);
  scalar("paged_kv.batch_size", q, kv.batch_size);
  scalar("paged_kv.stride_page", q, kv.stride_page);
  scalar("paged_kv.stride_n", q, kv.stride_n);
  scalar("paged_kv.stride_h", q, kv.stride_h);
  scalar("paged_kv.v_stride_page", q, kv.v_stride_page);
  scalar("paged_kv.v_stride_n", q, kv.v_stride_n);
  scalar("paged_kv.v_stride_h", q, kv.v_stride_h);
  scalar("paged_kv.k_data", q, kv.k_data);
  scalar("paged_kv.v_data", q, kv.v_data);
  scalar("paged_kv.indices", q, kv.indices);
  scalar("paged_kv.indptr", q, kv.indptr);
  scalar("paged_kv.last_page_len", q, kv.last_page_len);
  scalar("paged_kv.rope_pos_offset", q, kv.rope_pos_offset);
  scalar("o", q, q.o);
  scalar("lse", q, q.lse);
  scalar("maybe_alibi_slopes", q, q.maybe_alibi_slopes);
  scalar("padded_batch_size", q, q.padded_batch_size);
  scalar("num_qo_heads", q, q.num_qo_heads);
  scalar("q_stride_n", q, q.q_stride_n);
  scalar("q_stride_h", q, q.q_stride_h);
  scalar("window_left", q, q.window_left);
  scalar("logits_soft_cap", q, q.logits_soft_cap);
  scalar("sm_scale", q, q.sm_scale);
  scalar("rope_rcp_scale", q, q.rope_rcp_scale);
  scalar("rope_rcp_theta", q, q.rope_rcp_theta);
  scalar("request_indices", q, q.request_indices);
  scalar("kv_tile_indices", q, q.kv_tile_indices);
  scalar("o_indptr", q, q.o_indptr);
  scalar("kv_chunk_size_ptr", q, q.kv_chunk_size_ptr);
  scalar("block_valid_mask", q, q.block_valid_mask);
  scalar("partition_kv", q, q.partition_kv);

  // Launch configuration of the previous host code (and of upstream
  // BatchDecodeWithPagedKVCacheDispatched for this configuration).
  constexpr uint32_t num_stages_smem = 2, tile_size_per_bdx = 1, bdx = 16, bdy = 8, bdz = 1;
  constexpr size_t smem =
      2 * num_stages_smem * tile_size_per_bdx * bdy * bdz * head_dim * sizeof(T) +
      std::max(tile_size_per_bdx * bdx * bdy * bdz * sizeof(T*), 2 * bdy * bdz * sizeof(float));
  static_assert(smem == 2 * 2 * 8 * 128 * sizeof(T) + 128 * sizeof(T*),
                "dynamic smem differs from the previous launcher");
  std::printf("{\n  \"param_size\": %zu,\n", sizeof(Params));
  std::printf("  \"constants\": {\"shared_mem\": %zu, \"block\": [%u, %u, %u], "
              "\"grid_y\": %u, \"page_size_fastdiv\": \"%s\"},\n",
              smem, bdx, bdy, bdz, num_kv_heads,
              gqa_probe::hex(&kv.page_size, sizeof(kv.page_size)).c_str());
  gqa_probe::print_fields();
  std::printf(",\n  \"example\": {\"batch\": %d, \"sm_scale\": %.9g, \"bytes\": \"%s\"}\n}\n",
              batch, sm_scale, gqa_probe::hex(storage, sizeof(storage)).c_str());
  return 0;
}
