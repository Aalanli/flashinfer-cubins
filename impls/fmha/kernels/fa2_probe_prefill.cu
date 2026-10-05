// Host-only probe for the FA2 batch-prefill kernels of the FlashInfer 0.7.0 sm80 JIT cache.
//
// Compiled at the compile stage against the batch_prefill_config.inc that
// impls/fmha/fa2_build.py renders from the upstream JIT templates
// (csrc/batch_prefill_customize_config.jinja + flashinfer/jit/attention/utils.py), i.e. the
// exact PagedParams / RaggedParams definitions of the JIT-cache modules. Needs no GPU: the
// device queries are answered by the sm_86 model in fa2_probe_common.h. Prints JSON with
//
//  * "paged"/"ragged": sizeof and every field (offset, size, kind) of PagedParams/RaggedParams;
//  * "examples": structs built by the upstream host code of csrc/batch_prefill_paged.cuh and
//    csrc/batch_prefill.cu (Run functions, then the partition_kv / tmp_v substitution of
//    BatchPrefillWith{Paged,Ragged}KVCacheDispatched) for sample shapes with fake pointers:
//    non-split, split-KV and CUDA-graph split-KV plans;
//  * "plans": PrefillSplitQOKVIndptr (scheduler.cuh) results for sample indptrs and plan
//    options (disable_split_kv, the default binary-searched split, fixed_split_size, sliding
//    window, CUDA graph with and without uniform_q_len), plus the block_valid_mask
//    PrefillPlanImpl writes;
//  * "dispatch": for every dtype pair / head dim / CTA_TILE_Q / launcher, the NUM_MMA_KV the
//    upstream dispatcher selects on sm_86, the warp layout and sizeof(shared storage) (the
//    dynamic shared memory of the launch). The selection mirrors
//    BatchPrefillWith{Paged,Ragged}KVCacheDispatched (prefill.cuh) line by line and evaluates
//    the upstream KernelTraits.
//
// With -DFA2_PROBE_SINK the probe is built against the AttentionSink module's config
// (gen_batch_prefill_attention_sink_module: additional params ``float* sink; double sm_scale``,
// independent paged K/V strides) and reports that module's Params and examples.
#include "fa2_probe_common.h"
// clang-format off
#include "batch_prefill_config.inc"
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/scheduler.cuh>
// clang-format on

using fa2_probe::scalar;
using fa2_probe::sentinel;

#ifdef FA2_PROBE_SINK
// AttentionSink modules (gen_batch_prefill_attention_sink_module): additional
// params are the per-head sink logits and sm_scale only.
template <class P>
void common_tail_fields(P const& p) {
  scalar("sink", p, p.sink);
  scalar("sm_scale", p, p.sm_scale);
}
#else
template <class P>
void common_tail_fields(P const& p) {
  scalar("maybe_custom_mask", p, p.maybe_custom_mask);
  scalar("maybe_mask_indptr", p, p.maybe_mask_indptr);
  scalar("maybe_alibi_slopes", p, p.maybe_alibi_slopes);
  scalar("maybe_prefix_len_ptr", p, p.maybe_prefix_len_ptr);
  scalar("maybe_token_pos_in_items_ptr", p, p.maybe_token_pos_in_items_ptr);
  scalar("maybe_max_item_len_ptr", p, p.maybe_max_item_len_ptr);
  scalar("maybe_k_cache_sf", p, p.maybe_k_cache_sf);
  scalar("maybe_v_cache_sf", p, p.maybe_v_cache_sf);
  scalar("logits_soft_cap", p, p.logits_soft_cap);
  scalar("sm_scale", p, p.sm_scale);
  scalar("rope_rcp_scale", p, p.rope_rcp_scale);
  scalar("rope_rcp_theta", p, p.rope_rcp_theta);
  scalar("token_pos_in_items_len", p, p.token_pos_in_items_len);
}
#endif

void paged_fields(PagedParams const& p) {
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
  scalar("q_indptr", p, p.q_indptr);
  scalar("o", p, p.o);
  scalar("lse", p, p.lse);
  fa2_probe::field("group_size", "fastdiv", p, p.group_size);
  common_tail_fields(p);
  scalar("num_qo_heads", p, p.num_qo_heads);
  scalar("q_stride_n", p, p.q_stride_n);
  scalar("q_stride_h", p, p.q_stride_h);
  scalar("k_sf_stride_page", p, p.k_sf_stride_page);
  scalar("k_sf_stride_n", p, p.k_sf_stride_n);
  scalar("k_sf_stride_h", p, p.k_sf_stride_h);
  scalar("v_sf_stride_page", p, p.v_sf_stride_page);
  scalar("v_sf_stride_n", p, p.v_sf_stride_n);
  scalar("v_sf_stride_h", p, p.v_sf_stride_h);
  scalar("window_left", p, p.window_left);
  scalar("request_indices", p, p.request_indices);
  scalar("qo_tile_indices", p, p.qo_tile_indices);
  scalar("kv_tile_indices", p, p.kv_tile_indices);
  scalar("merge_indptr", p, p.merge_indptr);
  scalar("o_indptr", p, p.o_indptr);
  scalar("block_valid_mask", p, p.block_valid_mask);
  scalar("kv_chunk_size_ptr", p, p.kv_chunk_size_ptr);
  scalar("max_total_num_rows", p, p.max_total_num_rows);
  scalar("total_num_rows", p, p.total_num_rows);
  scalar("padded_batch_size", p, p.padded_batch_size);
  scalar("partition_kv", p, p.partition_kv);
}

void ragged_fields(RaggedParams const& p) {
  scalar("q", p, p.q);
  scalar("k", p, p.k);
  scalar("v", p, p.v);
  scalar("q_indptr", p, p.q_indptr);
  scalar("kv_indptr", p, p.kv_indptr);
  scalar("o", p, p.o);
  scalar("lse", p, p.lse);
  fa2_probe::field("group_size", "fastdiv", p, p.group_size);
  common_tail_fields(p);
  scalar("num_qo_heads", p, p.num_qo_heads);
  scalar("num_kv_heads", p, p.num_kv_heads);
  scalar("q_stride_n", p, p.q_stride_n);
  scalar("q_stride_h", p, p.q_stride_h);
  scalar("k_stride_n", p, p.k_stride_n);
  scalar("k_stride_h", p, p.k_stride_h);
  scalar("v_stride_n", p, p.v_stride_n);
  scalar("v_stride_h", p, p.v_stride_h);
  scalar("k_sf_stride_page", p, p.k_sf_stride_page);
  scalar("k_sf_stride_n", p, p.k_sf_stride_n);
  scalar("k_sf_stride_h", p, p.k_sf_stride_h);
  scalar("v_sf_stride_page", p, p.v_sf_stride_page);
  scalar("v_sf_stride_n", p, p.v_sf_stride_n);
  scalar("v_sf_stride_h", p, p.v_sf_stride_h);
  scalar("window_left", p, p.window_left);
  scalar("request_indices", p, p.request_indices);
  scalar("qo_tile_indices", p, p.qo_tile_indices);
  scalar("kv_tile_indices", p, p.kv_tile_indices);
  scalar("merge_indptr", p, p.merge_indptr);
  scalar("o_indptr", p, p.o_indptr);
  scalar("kv_chunk_size_ptr", p, p.kv_chunk_size_ptr);
  scalar("block_valid_mask", p, p.block_valid_mask);
  scalar("max_total_num_rows", p, p.max_total_num_rows);
  scalar("total_num_rows", p, p.total_num_rows);
  scalar("padded_batch_size", p, p.padded_batch_size);
  scalar("partition_kv", p, p.partition_kv);
}

// Pointer roles (sentinel indices) shared by both examples.
enum Role {
  kQ, kK, kV, kQIndptr, kKVIndptr, kKVIndices, kLastPageLen, kO, kLse, kCustomMask, kMaskIndptr,
  kPrefixLen, kTokenPos, kMaxItemLen, kRequestIndices, kQoTileIndices, kKvTileIndices, kOIndptr,
  kKvChunkSize, kSink, kMergeIndptr, kBlockValidMask, kTotalNumRows
};

struct Example {
  int batch, num_qo_heads, num_kv_heads, page_size, head_dim, kv_layout, window_left;
  double logits_soft_cap, sm_scale;
  int padded_batch_size, total_num_rows;
  int64_t token_pos_in_items_len;
  bool custom, multi_item;
  bool split, cuda_graph;  // plan_info.split_kv / enable_cuda_graph
  bool no_lse;             // run(return_lse=False): maybe_lse is None
};

// The Run functions' plan-dependent fields, then the Dispatched launcher's substitution: a
// split plan launches with partition_kv = true and o/lse pointing at tmp_v/tmp_s (the probe
// keeps the kO/kLse sentinels for them).
template <class P>
void set_plan(P& params, Example const& e) {
  params.request_indices = static_cast<IdType*>(sentinel(kRequestIndices));
  params.qo_tile_indices = static_cast<IdType*>(sentinel(kQoTileIndices));
  params.kv_tile_indices = static_cast<IdType*>(sentinel(kKvTileIndices));
  params.o_indptr = static_cast<IdType*>(sentinel(kOIndptr));
  params.kv_chunk_size_ptr = static_cast<IdType*>(sentinel(kKvChunkSize));
  if (e.split) {
    params.merge_indptr = static_cast<IdType*>(sentinel(kMergeIndptr));
    if (e.cuda_graph) params.block_valid_mask = static_cast<bool*>(sentinel(kBlockValidMask));
  }
  params.padded_batch_size = e.padded_batch_size;
  params.max_total_num_rows = e.total_num_rows;
  if (e.cuda_graph) params.total_num_rows = static_cast<uint32_t*>(sentinel(kTotalNumRows));
  params.partition_kv = e.split;
  if (e.split) {  // params.o = tmp_v; params.lse = tmp_s;
    params.o = static_cast<DTypeO*>(sentinel(kO));
    params.lse = static_cast<float*>(sentinel(kLse));
  }
}

// AdditionalParams setter of the JIT module (generate_additional_params): optional tensors are
// null unless given; scalars are the Python wrapper's values.
#ifdef FA2_PROBE_SINK
template <class P>
void set_additional(P& params, Example const& e) {
  params.sink = static_cast<float*>(sentinel(kSink));
  params.sm_scale = e.sm_scale;
}
#else
template <class P>
void set_additional(P& params, Example const& e) {
  params.maybe_custom_mask = e.custom ? (uint8_t*)sentinel(kCustomMask) : nullptr;
  params.maybe_mask_indptr = e.custom ? (int32_t*)sentinel(kMaskIndptr) : nullptr;
  params.maybe_alibi_slopes = nullptr;
  params.maybe_prefix_len_ptr = e.multi_item ? (uint32_t*)sentinel(kPrefixLen) : nullptr;
  params.maybe_token_pos_in_items_ptr = e.multi_item ? (uint16_t*)sentinel(kTokenPos) : nullptr;
  params.maybe_max_item_len_ptr = e.multi_item ? (uint16_t*)sentinel(kMaxItemLen) : nullptr;
  params.maybe_k_cache_sf = nullptr;
  params.maybe_v_cache_sf = nullptr;
  params.logits_soft_cap = e.logits_soft_cap;
  params.sm_scale = e.sm_scale;
  params.rope_rcp_scale = 1.0;
  params.rope_rcp_theta = 1.0 / 1e4;
  params.token_pos_in_items_len = e.token_pos_in_items_len;
}
#endif

// csrc/batch_prefill_paged.cuh (BatchPrefillWithPagedKVCacheRun), non-split plan; k/v caches
// are contiguous [num_pages, page_size, H, D] (NHD) or [num_pages, H, page_size, D] (HND).
void paged_example(Example const& e) {
  alignas(PagedParams) unsigned char storage[sizeof(PagedParams)];
  std::memset(storage, 0, sizeof(storage));
  auto& params = *new (storage) PagedParams;
  const QKVLayout kv_layout = static_cast<QKVLayout>(e.kv_layout);
  const int64_t num_kv_heads = e.num_kv_heads, page_size = e.page_size;
  const int64_t D = e.head_dim;
  std::vector<int64_t> strides =
      kv_layout == QKVLayout::kHND
          ? std::vector<int64_t>{num_kv_heads * page_size * D, page_size * D, D, 1}
          : std::vector<int64_t>{page_size * num_kv_heads * D, num_kv_heads * D, D, 1};
  const auto q_stride_n = int64_t(e.num_qo_heads) * D;
  const auto q_stride_h = D;

  params.q = static_cast<DTypeQ*>(sentinel(kQ));
  paged_kv_t<DTypeKV, IdType> paged_kv(
      num_kv_heads, page_size, HEAD_DIM_VO, e.batch, kv_layout,
      static_cast<DTypeKV*>(sentinel(kK)), static_cast<DTypeKV*>(sentinel(kV)), strides.data(),
      strides.data(), static_cast<IdType*>(sentinel(kKVIndices)),
      static_cast<IdType*>(sentinel(kKVIndptr)), static_cast<IdType*>(sentinel(kLastPageLen)));
  params.paged_kv = paged_kv;
  params.q_indptr = static_cast<IdType*>(sentinel(kQIndptr));
  params.o = static_cast<DTypeO*>(sentinel(kO));
  params.lse = e.no_lse ? nullptr : static_cast<float*>(sentinel(kLse));
  params.num_qo_heads = e.num_qo_heads;
  params.group_size = uint_fastdiv(e.num_qo_heads / paged_kv.num_heads);
  params.q_stride_n = q_stride_n;
  params.q_stride_h = q_stride_h;
  params.window_left = e.window_left;
  params.request_indices = nullptr;
  params.qo_tile_indices = nullptr;
  params.kv_tile_indices = nullptr;
  params.merge_indptr = nullptr;
  params.o_indptr = nullptr;
  params.kv_chunk_size_ptr = nullptr;
  params.block_valid_mask = nullptr;
  params.total_num_rows = nullptr;
  params.max_total_num_rows = 0;
  params.padded_batch_size = 0;
  params.partition_kv = false;
  params.k_sf_stride_page = 0;
  params.k_sf_stride_n = 0;
  params.k_sf_stride_h = 0;
  params.v_sf_stride_page = 0;
  params.v_sf_stride_n = 0;
  params.v_sf_stride_h = 0;
  set_additional(params, e);
  set_plan(params, e);
  std::printf("{\"kind\": \"paged\", \"args\": {\"batch\": %d, \"num_qo_heads\": %d, "
              "\"num_kv_heads\": %d, \"page_size\": %d, \"head_dim\": %d, \"kv_layout\": %d, "
              "\"window_left\": %d, \"logits_soft_cap\": %.17g, \"sm_scale\": %.17g, "
              "\"padded_batch_size\": %d, \"total_num_rows\": %d, \"token_pos_in_items_len\": "
              "%lld, \"custom\": %s, \"multi_item\": %s, \"split\": %s, \"cuda_graph\": %s, \"no_lse\": %s}, "
              "\"bytes\": \"%s\"}",
              e.batch, e.num_qo_heads, e.num_kv_heads, e.page_size, e.head_dim, e.kv_layout,
              e.window_left, e.logits_soft_cap, e.sm_scale, e.padded_batch_size,
              e.total_num_rows, (long long)e.token_pos_in_items_len, e.custom ? "true" : "false",
              e.multi_item ? "true" : "false", e.split ? "true" : "false",
              e.cuda_graph ? "true" : "false", e.no_lse ? "true" : "false",
              fa2_probe::hex(storage, sizeof(storage)).c_str());
}

// csrc/batch_prefill.cu (BatchPrefillWithRaggedKVCacheRun), non-split plan, NHD k/v
// [total_kv, H, D] (or HND [H, total_kv, D]).
void ragged_example(Example const& e) {
  alignas(RaggedParams) unsigned char storage[sizeof(RaggedParams)];
  std::memset(storage, 0, sizeof(storage));
  auto& params = *new (storage) RaggedParams;
  const QKVLayout kv_layout = static_cast<QKVLayout>(e.kv_layout);
  const uint32_t D = e.head_dim;
  const uint32_t total_kv = e.page_size;  // ragged examples reuse page_size as total_kv
  uint32_t q_stride_n = e.num_qo_heads * D, q_stride_h = D, k_stride_n, k_stride_h, v_stride_n,
           v_stride_h;
  if (kv_layout == QKVLayout::kNHD) {
    k_stride_n = e.num_kv_heads * D;
    k_stride_h = D;
    v_stride_n = e.num_kv_heads * D;
    v_stride_h = D;
  } else {
    k_stride_h = total_kv * D;
    k_stride_n = D;
    v_stride_h = total_kv * D;
    v_stride_n = D;
  }
  params.q = static_cast<DTypeQ*>(sentinel(kQ));
  params.k = static_cast<DTypeKV*>(sentinel(kK));
  params.v = static_cast<DTypeKV*>(sentinel(kV));
  params.o = static_cast<DTypeO*>(sentinel(kO));
  params.lse = e.no_lse ? nullptr : static_cast<float*>(sentinel(kLse));
  params.q_indptr = static_cast<IdType*>(sentinel(kQIndptr));
  params.kv_indptr = static_cast<IdType*>(sentinel(kKVIndptr));
  params.num_qo_heads = e.num_qo_heads;
  params.num_kv_heads = e.num_kv_heads;
  params.group_size = uint_fastdiv(e.num_qo_heads / e.num_kv_heads);
  params.q_stride_n = q_stride_n;
  params.q_stride_h = q_stride_h;
  params.k_stride_n = k_stride_n;
  params.k_stride_h = k_stride_h;
  params.v_stride_n = v_stride_n;
  params.v_stride_h = v_stride_h;
  params.window_left = e.window_left;
  params.request_indices = nullptr;
  params.qo_tile_indices = nullptr;
  params.kv_tile_indices = nullptr;
  params.merge_indptr = nullptr;
  params.o_indptr = nullptr;
  params.kv_chunk_size_ptr = nullptr;
  params.block_valid_mask = nullptr;
  params.total_num_rows = nullptr;
  params.max_total_num_rows = 0;
  params.padded_batch_size = 0;
  params.partition_kv = false;
  params.k_sf_stride_page = 0;
  params.k_sf_stride_n = 0;
  params.k_sf_stride_h = 0;
  params.v_sf_stride_page = 0;
  params.v_sf_stride_n = 0;
  params.v_sf_stride_h = 0;
  set_additional(params, e);
  set_plan(params, e);
  std::printf("{\"kind\": \"ragged\", \"args\": {\"batch\": %d, \"num_qo_heads\": %d, "
              "\"num_kv_heads\": %d, \"total_kv\": %d, \"head_dim\": %d, \"kv_layout\": %d, "
              "\"window_left\": %d, \"logits_soft_cap\": %.17g, \"sm_scale\": %.17g, "
              "\"padded_batch_size\": %d, \"total_num_rows\": %d, \"token_pos_in_items_len\": "
              "%lld, \"custom\": %s, \"multi_item\": %s, \"split\": %s, \"cuda_graph\": %s, \"no_lse\": %s}, "
              "\"bytes\": \"%s\"}",
              e.batch, e.num_qo_heads, e.num_kv_heads, total_kv, e.head_dim, e.kv_layout,
              e.window_left, e.logits_soft_cap, e.sm_scale, e.padded_batch_size,
              e.total_num_rows, (long long)e.token_pos_in_items_len, e.custom ? "true" : "false",
              e.multi_item ? "true" : "false", e.split ? "true" : "false",
              e.cuda_graph ? "true" : "false", e.no_lse ? "true" : "false",
              fa2_probe::hex(storage, sizeof(storage)).c_str());
}

// ---- dispatch (prefill.cuh BatchPrefillWith{Paged,Ragged}KVCacheDispatched on sm_86) ----

template <bool PAGED, class DTypeQ_, class DTypeKV_, uint32_t CTA_TILE_Q, uint32_t HEAD_DIM_QK,
          uint32_t HEAD_DIM_VO, uint32_t NUM_MMA_Q, uint32_t NUM_WARPS_Q, uint32_t NUM_WARPS_KV,
          uint32_t NUM_MMA_KV>
void print_selected(const char* dtype_q, const char* dtype_kv, int max_mma_kv) {
  using Variant = DefaultAttention<false, false, false, false>;
  constexpr auto POS = PosEncodingMode::kNone;
  using KTraits =
      KernelTraits<MaskMode::kNone, CTA_TILE_Q, NUM_MMA_Q, NUM_MMA_KV, HEAD_DIM_QK / 16,
                   HEAD_DIM_VO / 16, NUM_WARPS_Q, NUM_WARPS_KV, POS, DTypeQ_, DTypeKV_, DTypeQ_,
                   float, int32_t, Variant, kPrefillLauncherRepacksFp4>;
  size_t smem = 0;
  bool invalid = KTraits::IsInvalid();
  if constexpr (!KTraits::IsInvalid()) {
    if constexpr (PAGED) {
      smem = sizeof(typename KTraits::SharedStoragePaged);
    } else {
      using SmemStorage =
          std::conditional_t<KTraits::USE_SOFTMAX_VO_SPLIT, typename KTraits::SharedStoragePaged,
                             typename KTraits::SharedStorage>;
      smem = sizeof(SmemStorage);
    }
  }
  std::printf("{\"kind\": \"%s\", \"dtype_q\": \"%s\", \"dtype_kv\": \"%s\", \"head_dim\": %u, "
              "\"cta_tile_q\": %u, \"num_mma_q\": %u, \"num_warps_q\": %u, \"num_warps_kv\": %u, "
              "\"max_num_mma_kv\": %d, \"num_mma_kv\": %u, \"invalid\": %s, \"shared_mem\": %zu, "
              "\"fits\": %s}",
              PAGED ? "paged" : "ragged", dtype_q, dtype_kv, HEAD_DIM_QK, CTA_TILE_Q, NUM_MMA_Q,
              NUM_WARPS_Q, NUM_WARPS_KV, max_mma_kv, (unsigned)NUM_MMA_KV,
              invalid ? "true" : "false", smem,
              (!invalid && smem <= size_t(fa2_probe::kMaxSmemPerBlockOptin)) ? "true" : "false");
}

template <bool PAGED, class DTypeQ_, class DTypeKV_, uint32_t CTA_TILE_Q, uint32_t HEAD_DIM>
void dispatch(const char* dtype_q, const char* dtype_kv) {
  using DTypeQ = DTypeQ_;
  using DTypeKV = DTypeKV_;
  constexpr uint32_t HEAD_DIM_QK = HEAD_DIM, HEAD_DIM_VO = HEAD_DIM;
  constexpr auto POS_ENCODING_MODE = PosEncodingMode::kNone;
  constexpr bool USE_FP16_QK_REDUCTION = false;
  using AttentionVariant = DefaultAttention<false, false, false, false>;
  // --- verbatim from the dispatchers (paged first, ragged differences guarded by PAGED) ---
  constexpr bool kLargeHead = ((sizeof(DTypeKV) == 2) || is_fp4_type_v<DTypeKV> ||
                               (!PAGED && sizeof(DTypeKV) <= 2)) &&
                              (HEAD_DIM_VO >= 512) && (CTA_TILE_Q == 32);
  constexpr uint32_t NUM_MMA_Q = kLargeHead ? 1 : get_num_mma_q(CTA_TILE_Q);
  constexpr uint32_t NUM_WARPS_Q = kLargeHead ? 2 : get_num_warps_q(CTA_TILE_Q);
  constexpr uint32_t NUM_WARPS_KV = kLargeHead ? 2 : get_num_warps_kv(CTA_TILE_Q);
  constexpr uint32_t NUM_MMA_D_QK = HEAD_DIM_QK / 16;
  int dev_id = 0;
  cudaGetDevice(&dev_id);
  int max_smem_per_sm = 0;
  cudaDeviceGetAttribute(&max_smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev_id);
  int max_smem_per_block_optin = 0;
  cudaDeviceGetAttribute(&max_smem_per_block_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                         dev_id);
  constexpr bool kUseRepack =
      use_kv_repack<DTypeKV, CTA_TILE_Q, HEAD_DIM_QK, HEAD_DIM_VO>(kPrefillLauncherRepacksFp4);
  constexpr bool kKVShared = !is_fp4_type_v<DTypeKV> && (HEAD_DIM_VO / 16 > 16) &&
                             ((HEAD_DIM_VO / 16) % NUM_WARPS_KV == 0) &&
                             (HEAD_DIM_QK == HEAD_DIM_VO) &&
                             (sizeof(DTypeKV) == 2 || CTA_TILE_Q > 16);
  constexpr bool kVOSplitDispatch =
      PAGED ? ((HEAD_DIM_VO / 16 > 16) && ((HEAD_DIM_VO / 16) % NUM_WARPS_KV == 0))
            : (((sizeof(DTypeKV) == 2) || is_fp4_type_v<DTypeKV>) &&
               AttentionVariant::use_softmax && (HEAD_DIM_VO / 16 > 16) &&
               ((HEAD_DIM_VO / 16) % NUM_WARPS_KV == 0));
  constexpr uint32_t kKVSmemPerMmaKV =
      (kKVShared ? (HEAD_DIM_QK * 16 * NUM_WARPS_KV * sizeof(DTypeKV))
                 : ((HEAD_DIM_QK + HEAD_DIM_VO) * 16 * NUM_WARPS_KV * sizeof(DTypeKV))) +
      (kUseRepack ? ((HEAD_DIM_QK > HEAD_DIM_VO ? HEAD_DIM_QK : HEAD_DIM_VO) * 16 * NUM_WARPS_KV *
                     sizeof(DTypeQ))
                  : 0u) +
      (is_fp4_type_v<DTypeKV>
           ? ((HEAD_DIM_QK + HEAD_DIM_VO) * 16 * NUM_WARPS_KV / NVFP4_SF_VEC_SIZE)
           : 0u) +
      (kVOSplitDispatch ? (CTA_TILE_Q * NUM_WARPS_KV * 16 * sizeof(DTypeQ)) : 0u);
  constexpr uint32_t kVOSplitFixedSmem =
      kVOSplitDispatch ? (NUM_WARPS_KV * CTA_TILE_Q * 8u + 2048u) : 0u;
  constexpr uint32_t kSharedRopeFreqSmem = (POS_ENCODING_MODE == PosEncodingMode::kRoPELlama &&
                                            HEAD_DIM_QK > 256 && HEAD_DIM_QK == HEAD_DIM_VO)
                                               ? (4u * (NUM_MMA_D_QK / 2) * 4u * sizeof(float))
                                               : 0u;
  constexpr uint32_t kFixedSmem =
      CTA_TILE_Q * HEAD_DIM_QK * sizeof(DTypeQ) + kVOSplitFixedSmem + kSharedRopeFreqSmem;
  constexpr uint32_t kMinValidMmaKV =
      (sizeof(DTypeKV) == 1 && NUM_WARPS_Q > 2) ? (NUM_WARPS_Q / 2) : 1;
  const int num_ctas_per_sm =
      max_smem_per_sm >= 2 * (kFixedSmem + kMinValidMmaKV * kKVSmemPerMmaKV) ? 2 : 1;
  const int max_smem_per_threadblock =
      std::min(max_smem_per_sm / num_ctas_per_sm, max_smem_per_block_optin);
  const uint32_t max_num_mma_kv_reg =
      (HEAD_DIM_VO >= 128 && NUM_MMA_Q == 2 && POS_ENCODING_MODE == PosEncodingMode::kRoPELlama &&
       !USE_FP16_QK_REDUCTION)
          ? 2
          : (8 / NUM_MMA_Q);
  const int max_num_mma_kv_smem =
      (max_smem_per_threadblock - static_cast<int>(kFixedSmem)) / static_cast<int>(kKVSmemPerMmaKV);
  const int max_mma_kv =
      std::min(static_cast<uint32_t>(std::max(max_num_mma_kv_smem, 0)), max_num_mma_kv_reg);
  // DISPATCH_NUM_MMA_KV
#define FA2_SELECT(N)                                                                           \
  print_selected<PAGED, DTypeQ, DTypeKV, CTA_TILE_Q, HEAD_DIM_QK, HEAD_DIM_VO, NUM_MMA_Q,      \
                 NUM_WARPS_Q, NUM_WARPS_KV, N>(dtype_q, dtype_kv, max_mma_kv)
  if (max_num_mma_kv_smem < 1) {
    std::printf("{\"kind\": \"%s\", \"dtype_q\": \"%s\", \"dtype_kv\": \"%s\", \"head_dim\": %u, "
                "\"cta_tile_q\": %u, \"max_num_mma_kv\": 0, \"num_mma_kv\": 0, \"fits\": false}",
                PAGED ? "paged" : "ragged", dtype_q, dtype_kv, HEAD_DIM, CTA_TILE_Q);
  } else if (max_mma_kv >= 8) {
    FA2_SELECT(8);
  } else if (max_mma_kv >= 4) {
    FA2_SELECT(4);
  } else if (max_mma_kv >= 2) {
    FA2_SELECT(2);
  } else {
    FA2_SELECT(1);
  }
#undef FA2_SELECT
}

bool first_entry = true;
void sep() {
  std::printf(first_entry ? "\n    " : ",\n    ");
  first_entry = false;
}

template <class DQ, class DKV>
void dispatch_all(const char* q, const char* kv) {
  sep(); dispatch<true, DQ, DKV, 16, 64>(q, kv);
  sep(); dispatch<true, DQ, DKV, 64, 64>(q, kv);
  sep(); dispatch<true, DQ, DKV, 128, 64>(q, kv);
  sep(); dispatch<true, DQ, DKV, 16, 128>(q, kv);
  sep(); dispatch<true, DQ, DKV, 64, 128>(q, kv);
  sep(); dispatch<true, DQ, DKV, 128, 128>(q, kv);
  sep(); dispatch<true, DQ, DKV, 16, 256>(q, kv);
  sep(); dispatch<true, DQ, DKV, 64, 256>(q, kv);
  sep(); dispatch<true, DQ, DKV, 128, 256>(q, kv);
  sep(); dispatch<false, DQ, DKV, 16, 64>(q, kv);
  sep(); dispatch<false, DQ, DKV, 64, 64>(q, kv);
  sep(); dispatch<false, DQ, DKV, 128, 64>(q, kv);
  sep(); dispatch<false, DQ, DKV, 16, 128>(q, kv);
  sep(); dispatch<false, DQ, DKV, 64, 128>(q, kv);
  sep(); dispatch<false, DQ, DKV, 128, 128>(q, kv);
  sep(); dispatch<false, DQ, DKV, 16, 256>(q, kv);
  sep(); dispatch<false, DQ, DKV, 64, 256>(q, kv);
  sep(); dispatch<false, DQ, DKV, 128, 256>(q, kv);
}

struct PlanExample {
  std::vector<int32_t> qo_indptr, kv_indptr;
  uint32_t num_qo_heads, num_kv_heads, head_dim, page_size, kv_dtype_bytes;
  bool disable_split_kv = true, enable_cuda_graph = false;
  int32_t window_left = -1, fixed_split_size = -1;
  int64_t uniform_q_len = 0;
  uint32_t total_num_rows = 0;  // 0: qo_indptr.back() (CUDA graph: max_total_num_rows)
};

void plan_example(PlanExample p) {
  const uint32_t batch = p.qo_indptr.size() - 1;
  const uint32_t total_num_rows = p.total_num_rows ? p.total_num_rows : p.qo_indptr.back();
  // PrefillPlanImpl step 0 on the sm_86 model (num_blocks_per_sm = 2, no colocated CTAs).
  int num_sm = 0, dev_id = 0;
  cudaGetDevice(&dev_id);
  cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id);
  int max_grid_size = 2 * num_sm;
  uint32_t max_batch_size_if_split = max_grid_size / p.num_kv_heads;
  auto [split_kv, new_batch_size, padded_batch_size, cta_tile_q, kv_chunk_size, request_indices,
        qo_tile_indices, kv_tile_indices, merge_indptr, o_indptr] =
      PrefillSplitQOKVIndptr(p.qo_indptr.data(), p.kv_indptr.data(), total_num_rows, batch,
                             p.num_qo_heads, p.num_kv_heads, p.head_dim, p.page_size,
                             max_batch_size_if_split, p.enable_cuda_graph, p.window_left,
                             p.fixed_split_size, p.disable_split_kv, p.uniform_q_len, p.head_dim,
                             p.kv_dtype_bytes);
  // PrefillPlanImpl stores kv_chunk_size into an IdType slot; the Run functions pass
  // block_valid_mask (i < new_batch_size) only for CUDA-graph split plans.
  const int32_t kv_chunk_size_id = static_cast<int32_t>(kv_chunk_size);
  std::vector<int32_t> block_valid_mask;
  if (split_kv && p.enable_cuda_graph) {
    for (uint32_t i = 0; i < padded_batch_size; ++i) block_valid_mask.push_back(i < new_batch_size);
  }
  std::printf("{\"qo_indptr\": ");
  fa2_probe::print_vec(p.qo_indptr);
  std::printf(", \"kv_indptr\": ");
  fa2_probe::print_vec(p.kv_indptr);
  std::printf(", \"num_qo_heads\": %u, \"num_kv_heads\": %u, \"head_dim\": %u, \"page_size\": %u, "
              "\"kv_dtype_bytes\": %u, \"disable_split_kv\": %s, \"cuda_graph\": %s, "
              "\"window_left\": %d, \"fixed_split_size\": %d, \"uniform_q_len\": %lld, "
              "\"total_num_rows\": %u, \"split_kv\": %s, \"new_batch_size\": %u, "
              "\"padded_batch_size\": %zu, \"cta_tile_q\": %u, \"kv_chunk_size\": %d, "
              "\"request_indices\": ",
              p.num_qo_heads, p.num_kv_heads, p.head_dim, p.page_size, p.kv_dtype_bytes,
              p.disable_split_kv ? "true" : "false", p.enable_cuda_graph ? "true" : "false",
              p.window_left, p.fixed_split_size, (long long)p.uniform_q_len, total_num_rows,
              split_kv ? "true" : "false", new_batch_size, size_t(padded_batch_size), cta_tile_q,
              kv_chunk_size_id);
  fa2_probe::print_vec(request_indices);
  std::printf(", \"qo_tile_indices\": ");
  fa2_probe::print_vec(qo_tile_indices);
  std::printf(", \"kv_tile_indices\": ");
  fa2_probe::print_vec(kv_tile_indices);
  std::printf(", \"merge_indptr\": ");
  fa2_probe::print_vec(merge_indptr);
  std::printf(", \"o_indptr\": ");
  fa2_probe::print_vec(o_indptr);
  std::printf(", \"block_valid_mask\": ");
  fa2_probe::print_vec(block_valid_mask);
  std::printf("}");
}

int main() {
  std::printf("{\n  \"paged\": {\"param_size\": %zu, \"fields\": ", sizeof(PagedParams));
  {
    PagedParams p;
    paged_fields(p);
  }
  fa2_probe::print_fields();
  std::printf("},\n  \"ragged\": {\"param_size\": %zu, \"fields\": ", sizeof(RaggedParams));
  {
    RaggedParams p;
    ragged_fields(p);
  }
  fa2_probe::print_fields();
  std::printf("},\n  \"examples\": [");
  const Example examples[] = {
      // batch, qo heads, kv heads, page_size (ragged: total_kv), head_dim, layout, window_left,
      // soft cap, sm_scale, padded batch, total rows, token_pos_in_items_len, custom, multi-item,
      // split, cuda graph, no lse (head dim: the rendered config's HEAD_DIM_VO, 128)
      {3, 32, 8, 16, 128, 0, -1, 0.0, 0.08838834764831845, 7, 40, 0, false, false, false, false, false},
      {5, 12, 4, 5, 128, 1, 37, 30.0, 0.125, 11, 333, 0, true, false, false, false, false},
      {1, 6, 6, 1, 128, 0, -1, 0.0, 0.5, 1, 9, 100, false, true, false, false, false},
      {9, 8, 1, 3, 128, 0, 15, 50.0, 1.0, 129, 4097, 0, false, false, false, false, false},
      {4, 16, 4, 16, 128, 1, -1, 0.0, 0.08838834764831845, 23, 77, 0, false, false, true, false, false},
      {2, 8, 2, 5, 128, 0, 63, 5.0, 0.125, 200, 9, 0, true, false, true, true, false},
      {6, 4, 4, 1, 128, 0, -1, 0.0, 0.0625, 6, 30, 0, false, false, false, false, true},
      {3, 32, 8, 16, 128, 0, -1, 0.0, 0.08838834764831845, 41, 12, 7, false, true, false, true, true},
  };
  bool first = true;
  for (auto const& e : examples) {
    std::printf(first ? "\n    " : ",\n    ");
    first = false;
    paged_example(e);
    std::printf(",\n    ");
    ragged_example(e);
  }
  std::printf("\n  ],\n  \"plans\": [");
  // qo_indptr, kv_indptr, qo heads, kv heads, head dim, page size, kv bytes, disable_split_kv,
  // cuda graph, window_left, fixed_split_size, uniform_q_len, total_num_rows
  const PlanExample plans[] = {
      {{0, 2, 3, 7}, {0, 3, 5, 9}, 32, 8, 128, 16, 2},
      {{0, 1, 2, 3, 4}, {0, 40, 80, 81, 200}, 4, 4, 64, 1, 1},
      {{0, 30, 60}, {0, 7, 15}, 12, 4, 256, 5, 2},
      {{0, 100, 101, 400}, {0, 100, 101, 400}, 8, 8, 128, 1, 2},
      {{0, 17, 34}, {0, 50, 70}, 2, 1, 64, 3, 1},
      {{0, 5, 10, 15}, {0, 20, 40, 60}, 8, 1, 256, 1, 2},
      {{0, 1, 2, 3}, {0, 400, 800, 1200}, 4, 4, 128, 1, 2, false},
      {{0, 1, 2, 3}, {0, 400, 800, 1200}, 4, 4, 128, 1, 2, false, false, 31},
      {{0, 37, 74, 111}, {0, 30, 31, 60}, 32, 4, 128, 5, 2, false, false, 100},
      {{0, 1, 2, 3}, {0, 400, 800, 1200}, 4, 4, 128, 1, 2, false, false, -1, 7},
      {{0, 3, 9, 10}, {0, 4, 25, 26}, 16, 8, 64, 16, 2, false, true},
      {{0, 2, 4, 6}, {0, 10, 20, 30}, 16, 2, 64, 16, 2, false, true, -1, -1, 2},
      {{0, 2, 4, 6}, {0, 10, 20, 30}, 16, 2, 64, 16, 2, false, true, -1, -1, 0, 64},
      {{0, 0, 5, 5, 6}, {0, 3, 3, 9, 9}, 8, 2, 128, 1, 2, false},
      {{0, 600, 1200}, {0, 900, 1500}, 32, 8, 128, 16, 2, false},
      {{0, 40, 41}, {0, 2000, 2003}, 8, 8, 256, 1, 1, false, false, 63},
  };
  first = true;
  for (auto const& p : plans) {
    std::printf(first ? "\n    " : ",\n    ");
    first = false;
    plan_example(p);
  }
  std::printf("\n  ],\n  \"dispatch\": [");
  dispatch_all<nv_bfloat16, nv_bfloat16>("bf16", "bf16");
  dispatch_all<nv_bfloat16, __nv_fp8_e4m3>("bf16", "e4m3");
  dispatch_all<half, __nv_fp8_e4m3>("f16", "e4m3");
  dispatch_all<half, half>("f16", "f16");
  std::printf("\n  ]\n}\n");
  return 0;
}
