// Host-only probe for the FA2 batch-decode and merge-states kernels of the FlashInfer 0.7.0
// sm80 JIT cache.
//
// Compiled at the compile stage against the batch_decode_config.inc rendered by
// impls/fmha/fa2_build.py from csrc/batch_decode_customize_config.jinja (the exact Params of the
// JIT-cache modules). Needs no GPU. Prints JSON with
//
//  * "params": sizeof and every field (offset, size, kind) of the decode Params;
//  * "examples": Params built by the upstream host code of csrc/batch_decode.cu
//    (BatchDecodeWithPagedKVCacheRun, then the partition_kv / tmp_v substitution of
//    BatchDecodeWithPagedKVCacheDispatched) for sample shapes with fake pointers: non-split,
//    split-KV and CUDA-graph plans;
//  * "plans": DecodePlanImpl results (the split decision of
//    BatchDecodeWithPagedKVCacheWorkEstimationDispatched for a given occupancy-limited grid,
//    PartitionPagedKVCacheBinarySearchMinNumPagePerBatch, DecodeSplitKVIndptr, the padded
//    batch and the block_valid_mask) for sample page counts;
//  * "dispatch": per dtype pair / head dim / GQA group size, the template arguments and the
//    dynamic shared memory BatchDecodeWithPagedKVCacheDispatched (decode.cuh) launches on
//    sm_86 (compute capability 8.x -> NUM_STAGES_SMEM = 2);
//  * "merge": per dtype / head dim, the template arguments and dynamic shared memory of
//    VariableLengthMergeStates (cascade.cuh).
#include "fa2_probe_common.h"
// clang-format off
#include "batch_decode_config.inc"
#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/cascade.cuh>
#include <flashinfer/attention/scheduler.cuh>
// clang-format on

using fa2_probe::scalar;
using fa2_probe::sentinel;

void params_fields(Params const& p) {
  auto const& kv = p.paged_kv;
  scalar("q", p, p.q);
  fa2_probe::field("paged_kv.page_size", "fastdiv", p, kv.page_size);
  scalar("paged_kv.num_heads", p, kv.num_heads);
  scalar("paged_kv.head_dim", p, kv.head_dim);
  scalar("paged_kv.batch_size", p, kv.batch_size);
  scalar("paged_kv.stride_page", p, kv.stride_page);
  scalar("paged_kv.stride_n", p, kv.stride_n);
  scalar("paged_kv.stride_h", p, kv.stride_h);
  scalar("paged_kv.v_stride_page", p, kv.v_stride_page);
  scalar("paged_kv.v_stride_n", p, kv.v_stride_n);
  scalar("paged_kv.v_stride_h", p, kv.v_stride_h);
  scalar("paged_kv.k_data", p, kv.k_data);
  scalar("paged_kv.v_data", p, kv.v_data);
  scalar("paged_kv.indices", p, kv.indices);
  scalar("paged_kv.indptr", p, kv.indptr);
  scalar("paged_kv.last_page_len", p, kv.last_page_len);
  scalar("paged_kv.rope_pos_offset", p, kv.rope_pos_offset);
  scalar("o", p, p.o);
  scalar("lse", p, p.lse);
  scalar("maybe_alibi_slopes", p, p.maybe_alibi_slopes);
  scalar("logits_soft_cap", p, p.logits_soft_cap);
  scalar("sm_scale", p, p.sm_scale);
  scalar("rope_rcp_scale", p, p.rope_rcp_scale);
  scalar("rope_rcp_theta", p, p.rope_rcp_theta);
  scalar("padded_batch_size", p, p.padded_batch_size);
  scalar("num_qo_heads", p, p.num_qo_heads);
  scalar("q_stride_n", p, p.q_stride_n);
  scalar("q_stride_h", p, p.q_stride_h);
  scalar("window_left", p, p.window_left);
  scalar("enable_pdl", p, p.enable_pdl);
  scalar("request_indices", p, p.request_indices);
  scalar("kv_tile_indices", p, p.kv_tile_indices);
  scalar("o_indptr", p, p.o_indptr);
  scalar("kv_chunk_size_ptr", p, p.kv_chunk_size_ptr);
  scalar("block_valid_mask", p, p.block_valid_mask);
  scalar("partition_kv", p, p.partition_kv);
}

enum Role {
  kQ, kK, kV, kKVIndices, kKVIndptr, kLastPageLen, kO, kLse, kRequestIndices, kKvTileIndices,
  kOIndptr, kKvChunkSize, kBlockValidMask
};

struct Example {
  int batch, num_qo_heads, num_kv_heads, page_size, head_dim, kv_layout, window_left;
  double logits_soft_cap, sm_scale;
  int padded_batch_size = 0;  // 0: batch
  bool split = false, cuda_graph = false, no_lse = false;
};

// csrc/batch_decode.cu BatchDecodeWithPagedKVCacheRun with a non-split plan; k/v caches are
// contiguous [num_pages, page_size, H, D] (NHD) or [num_pages, H, page_size, D] (HND).
// Upstream leaves Params::enable_pdl unset (the kernel never reads it); the probe zero-fills
// the struct first, so it is 0 here.
void example(Example const& e) {
  alignas(Params) unsigned char storage[sizeof(Params)];
  std::memset(storage, 0, sizeof(storage));
  auto& params_ref = *new (storage) Params;
  const QKVLayout kv_layout = static_cast<QKVLayout>(e.kv_layout);
  const int64_t H = e.num_kv_heads, P = e.page_size, D = e.head_dim;
  std::vector<int64_t> k_strides = kv_layout == QKVLayout::kHND
                                       ? std::vector<int64_t>{H * P * D, P * D, D, 1}
                                       : std::vector<int64_t>{P * H * D, H * D, D, 1};
  std::vector<int64_t> v_strides = k_strides;
  const auto q_stride_n = int64_t(e.num_qo_heads) * D;
  const auto q_stride_h = D;
  paged_kv_t<DTypeKV, IdType> paged_kv(
      e.num_kv_heads, e.page_size, HEAD_DIM_QK, e.batch, kv_layout,
      static_cast<DTypeKV*>(sentinel(kK)), static_cast<DTypeKV*>(sentinel(kV)), k_strides.data(),
      v_strides.data(), static_cast<IdType*>(sentinel(kKVIndices)),
      static_cast<IdType*>(sentinel(kKVIndptr)), static_cast<IdType*>(sentinel(kLastPageLen)));
  Params& params = params_ref;
  params.q = static_cast<DTypeQ*>(sentinel(kQ));
  params.paged_kv = paged_kv;
  params.o = static_cast<DTypeO*>(sentinel(kO));
  params.lse = e.no_lse ? nullptr : static_cast<float*>(sentinel(kLse));
  params.padded_batch_size = 0;
  params.num_qo_heads = e.num_qo_heads;
  params.q_stride_n = q_stride_n;
  params.q_stride_h = q_stride_h;
  params.window_left = e.window_left;
  params.request_indices = nullptr;
  params.kv_tile_indices = nullptr;
  params.o_indptr = nullptr;
  params.kv_chunk_size_ptr = nullptr;
  params.block_valid_mask = nullptr;
  params.partition_kv = false;
  // ADDITIONAL_PARAMS_SETTER
  params.maybe_alibi_slopes = nullptr;
  params.logits_soft_cap = e.logits_soft_cap;
  params.sm_scale = e.sm_scale;
  params.rope_rcp_scale = 1.0;
  params.rope_rcp_theta = 1.0 / 1e4;
  params.request_indices = static_cast<IdType*>(sentinel(kRequestIndices));
  params.kv_tile_indices = static_cast<IdType*>(sentinel(kKvTileIndices));
  params.o_indptr = static_cast<IdType*>(sentinel(kOIndptr));
  params.kv_chunk_size_ptr = static_cast<IdType*>(sentinel(kKvChunkSize));
  if (e.split && e.cuda_graph) {
    params.block_valid_mask = static_cast<bool*>(sentinel(kBlockValidMask));
  }
  params.padded_batch_size = e.padded_batch_size ? e.padded_batch_size : e.batch;
  params.partition_kv = false;
  if (e.split) {  // Dispatched: partition_kv = true; o = tmp_v; lse = tmp_s (same sentinels)
    params.partition_kv = true;
    params.o = static_cast<DTypeO*>(sentinel(kO));
    params.lse = static_cast<float*>(sentinel(kLse));
  }
  std::printf("{\"args\": {\"batch\": %d, \"num_qo_heads\": %d, \"num_kv_heads\": %d, "
              "\"page_size\": %d, \"head_dim\": %d, \"kv_layout\": %d, \"window_left\": %d, "
              "\"logits_soft_cap\": %.17g, \"sm_scale\": %.17g, \"padded_batch_size\": %d, "
              "\"split\": %s, \"cuda_graph\": %s, \"no_lse\": %s}, \"bytes\": \"%s\"}",
              e.batch, e.num_qo_heads, e.num_kv_heads, e.page_size, e.head_dim, e.kv_layout,
              e.window_left, e.logits_soft_cap, e.sm_scale,
              e.padded_batch_size ? e.padded_batch_size : e.batch, e.split ? "true" : "false",
              e.cuda_graph ? "true" : "false", e.no_lse ? "true" : "false",
              fa2_probe::hex(storage, sizeof(storage)).c_str());
}

// DecodePlanImpl for page counts and an occupancy-limited grid (the host-only probe has no
// cudaOccupancyMaxActiveBlocksPerMultiprocessor): the branch of
// BatchDecodeWithPagedKVCacheWorkEstimationDispatched below the occupancy query, verbatim.
struct DecodePlanExample {
  std::vector<int32_t> num_pages;
  uint32_t num_kv_heads, page_size, max_grid_size;
  bool enable_cuda_graph;
};

void decode_plan_example(DecodePlanExample const& p) {
  using IdType = int32_t;
  const uint32_t batch_size = p.num_pages.size(), gdy = p.num_kv_heads;
  const uint32_t max_grid_size = p.max_grid_size, page_size = p.page_size;
  const bool enable_cuda_graph = p.enable_cuda_graph;
  std::vector<IdType> kv_indptr_h{0};
  for (auto n : p.num_pages) kv_indptr_h.push_back(kv_indptr_h.back() + n);
  bool split_kv;
  uint32_t max_num_pages_per_batch, new_batch_size;
  if (batch_size * gdy >= max_grid_size) {
    split_kv = false;
    max_num_pages_per_batch = 1;
    for (uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
      max_num_pages_per_batch = std::max<uint32_t>(
          max_num_pages_per_batch, kv_indptr_h[batch_idx + 1] - kv_indptr_h[batch_idx]);
    }
    new_batch_size = batch_size;
  } else {
    std::vector<IdType> num_pages(batch_size);
    for (uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
      num_pages[batch_idx] = kv_indptr_h[batch_idx + 1] - kv_indptr_h[batch_idx];
    }
    std::tie(max_num_pages_per_batch, new_batch_size) =
        PartitionPagedKVCacheBinarySearchMinNumPagePerBatch(max_grid_size, gdy, num_pages,
                                                            std::max(128 / page_size, 1U));
    if (new_batch_size == batch_size && !enable_cuda_graph) {
      split_kv = false;
    } else {
      split_kv = true;
    }
  }
  // DecodePlanImpl
  const size_t padded_batch_size =
      (enable_cuda_graph) ? (split_kv ? max_grid_size / gdy : batch_size) : new_batch_size;
  auto [request_indices, kv_tile_indices, o_indptr] =
      DecodeSplitKVIndptr(kv_indptr_h.data(), batch_size, max_num_pages_per_batch);
  std::vector<int32_t> block_valid_mask;  // passed by the Run function to CUDA-graph launches
  if (split_kv && enable_cuda_graph) {
    for (uint32_t i = 0; i < padded_batch_size; ++i) block_valid_mask.push_back(i < new_batch_size);
  }
  std::printf("{\"num_pages\": ");
  fa2_probe::print_vec(p.num_pages);
  std::printf(", \"num_kv_heads\": %u, \"page_size\": %u, \"max_grid_size\": %u, "
              "\"cuda_graph\": %s, \"split_kv\": %s, \"new_batch_size\": %u, "
              "\"padded_batch_size\": %zu, \"kv_chunk_size\": %u, \"request_indices\": ",
              p.num_kv_heads, page_size, max_grid_size, enable_cuda_graph ? "true" : "false",
              split_kv ? "true" : "false", new_batch_size, padded_batch_size,
              max_num_pages_per_batch * page_size);
  fa2_probe::print_vec(request_indices);
  std::printf(", \"kv_tile_indices\": ");
  fa2_probe::print_vec(kv_tile_indices);
  std::printf(", \"o_indptr\": ");
  fa2_probe::print_vec(o_indptr);
  std::printf(", \"block_valid_mask\": ");
  fa2_probe::print_vec(block_valid_mask);
  std::printf("}");
}

bool first_entry = true;
void sep() {
  std::printf(first_entry ? "\n    " : ",\n    ");
  first_entry = false;
}

// BatchDecodeWithPagedKVCacheDispatched for one GROUP_SIZE (verbatim launch arithmetic).
template <class DTypeKV_, uint32_t HEAD_DIM, uint32_t GROUP_SIZE>
void decode_dispatch(const char* dtype_q, const char* dtype_kv) {
  using DTypeKV = DTypeKV_;
  constexpr uint32_t vec_size = std::max(16UL / sizeof(DTypeKV), HEAD_DIM / 32UL);
  auto compute_capacity = GetCudaComputeCapability();
  constexpr uint32_t bdx = HEAD_DIM / vec_size;
  static_assert(bdx <= 32);
  constexpr uint32_t bdy = GROUP_SIZE;
  constexpr uint32_t num_threads = std::max(128U, bdx * bdy);
  constexpr uint32_t bdz = num_threads / (bdx * bdy);
  constexpr uint32_t tile_size_per_bdx = GROUP_SIZE == 1 ? (sizeof(DTypeKV) == 1 ? 2U : 4U) : 1U;
  DISPATCH_COMPUTE_CAP_DECODE_NUM_STAGES_SMEM(compute_capacity, NUM_STAGES_SMEM, {
    const uint32_t smem_size =
        std::max(2 * NUM_STAGES_SMEM * tile_size_per_bdx * bdy * bdz * HEAD_DIM * sizeof(DTypeKV),
                 bdz * bdy * HEAD_DIM * sizeof(float)) +
        std::max(tile_size_per_bdx * num_threads * sizeof(DTypeKV*),
                 2 * bdy * bdz * sizeof(float));
    sep();
    std::printf("{\"dtype_q\": \"%s\", \"dtype_kv\": \"%s\", \"head_dim\": %u, \"group_size\": "
                "%u, \"num_stages_smem\": %u, \"tile_size_per_bdx\": %u, \"vec_size\": %u, "
                "\"bdx\": %u, \"bdy\": %u, \"bdz\": %u, \"shared_mem\": %u}",
                dtype_q, dtype_kv, HEAD_DIM, GROUP_SIZE, NUM_STAGES_SMEM, tile_size_per_bdx,
                vec_size, bdx, bdy, bdz, smem_size);
  });
}

template <class DTypeKV, uint32_t HEAD_DIM>
void decode_groups(const char* q, const char* kv) {
  decode_dispatch<DTypeKV, HEAD_DIM, 1>(q, kv);
  decode_dispatch<DTypeKV, HEAD_DIM, 2>(q, kv);
  decode_dispatch<DTypeKV, HEAD_DIM, 3>(q, kv);
  decode_dispatch<DTypeKV, HEAD_DIM, 4>(q, kv);
  decode_dispatch<DTypeKV, HEAD_DIM, 6>(q, kv);
  decode_dispatch<DTypeKV, HEAD_DIM, 8>(q, kv);
}

template <class DTypeKV>
void decode_all(const char* q, const char* kv) {
  decode_groups<DTypeKV, 64>(q, kv);
  decode_groups<DTypeKV, 128>(q, kv);
  decode_groups<DTypeKV, 256>(q, kv);
}

// VariableLengthMergeStates (cascade.cuh) launch arithmetic for one head dim.
template <class DTypeIn, uint32_t HEAD_DIM>
void merge_dispatch(const char* dtype) {
  constexpr uint32_t vec_size = std::max<uint32_t>(16U / sizeof(DTypeIn), HEAD_DIM / 32U);
  constexpr uint32_t bdx = HEAD_DIM / vec_size;
  constexpr uint32_t num_threads = 128;
  constexpr uint32_t bdy = num_threads / bdx;
  constexpr uint32_t num_smem_stages = 4;
  const uint32_t head_dim = HEAD_DIM;
  uint32_t smem_size =
      num_smem_stages * bdy * head_dim * sizeof(DTypeIn) + num_threads * sizeof(float);
  sep();
  std::printf("{\"dtype\": \"%s\", \"head_dim\": %u, \"vec_size\": %u, \"bdx\": %u, \"bdy\": %u, "
              "\"num_smem_stages\": %u, \"num_threads\": %u, \"shared_mem\": %u}",
              dtype, HEAD_DIM, vec_size, bdx, bdy, num_smem_stages, num_threads, smem_size);
}

int main() {
  std::printf("{\n  \"params\": {\"param_size\": %zu, \"fields\": ", sizeof(Params));
  {
    Params p;
    params_fields(p);
  }
  fa2_probe::print_fields();
  std::printf("},\n  \"examples\": [");
  const Example examples[] = {
      {3, 32, 8, 16, 128, 0, -1, 0.0, 0.08838834764831845},
      {7, 12, 4, 5, 128, 1, 37, 30.0, 0.125},
      {1, 6, 1, 1, 128, 0, -1, 50.0, 0.5},
      {300, 8, 8, 3, 128, 0, 15, 0.0, 1.0},
      // padded batch, split, cuda graph, no lse
      {5, 16, 4, 16, 128, 0, -1, 0.0, 0.08838834764831845, 23, true, false, false},
      {3, 8, 1, 1, 128, 1, 100, 4.0, 0.125, 82, true, true, false},
      {12, 32, 4, 8, 128, 0, -1, 0.0, 0.0625, 12, false, true, true},
      {2, 4, 4, 5, 128, 1, -1, 0.0, 0.25, 0, false, false, true},
  };
  bool first = true;
  for (auto const& e : examples) {
    std::printf(first ? "\n    " : ",\n    ");
    first = false;
    example(e);
  }
  std::printf("\n  ],\n  \"plans\": [");
  const DecodePlanExample plans[] = {
      {{1, 2, 3}, 4, 16, 164, false},
      {{40, 3, 0, 200}, 2, 1, 164, false},
      {{40, 3, 0, 200}, 2, 1, 164, true},
      {{7, 7, 7, 7, 7, 7}, 8, 16, 82, false},
      {{7, 7, 7, 7, 7, 7}, 8, 16, 82, true},
      {{300, 1, 5}, 1, 5, 246, false},
      {{2048, 9, 33, 1}, 4, 1, 164, false},
      {{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, 32, 16, 164, false},
  };
  first = true;
  for (auto const& p : plans) {
    std::printf(first ? "\n    " : ",\n    ");
    first = false;
    decode_plan_example(p);
  }
  std::printf("\n  ],\n  \"dispatch\": [");
  decode_all<nv_bfloat16>("bf16", "bf16");
  decode_all<__nv_fp8_e4m3>("bf16", "e4m3");
  decode_all<__nv_fp8_e4m3>("f16", "e4m3");
  decode_all<half>("f16", "f16");
  std::printf("\n  ],\n  \"merge\": [");
  first_entry = true;
  merge_dispatch<nv_bfloat16, 64>("bf16");
  merge_dispatch<nv_bfloat16, 128>("bf16");
  merge_dispatch<nv_bfloat16, 256>("bf16");
  merge_dispatch<nv_bfloat16, 512>("bf16");
  merge_dispatch<half, 64>("f16");
  merge_dispatch<half, 128>("f16");
  merge_dispatch<half, 256>("f16");
  merge_dispatch<half, 512>("f16");
  std::printf("\n  ]\n}\n");
  return 0;
}
