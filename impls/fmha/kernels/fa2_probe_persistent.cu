// Host-only probe for the persistent (holistic) batch-attention kernels of the FlashInfer 0.7.0
// sm80 JIT cache (batch_attention_with_kv_cache_* modules, flashinfer.BatchAttention).
//
// Compiled at the compile stage against the batch_attention_config.inc that
// impls/fmha/fa2_build.py renders from csrc/batch_attention_customize_config.jinja (the exact
// PersistentParams of the JIT-cache modules). Needs no GPU: device queries are answered by the
// sm_86 model of fa2_probe_common.h and the plan's host-to-device copy is skipped (the plan
// arrays are read back from the page-locked buffer). Prints JSON with
//
//  * "params": sizeof and every field (offset, size, kind) of PersistentParams;
//  * "plans": TwoStageHolisticPlan (scheduler.cuh) results for sample batches: launch grid,
//    every per-task work array, merge arrays and the workspace offsets;
//  * "examples": the two PersistentParams BatchPagedAttentionRun (csrc/batch_attention.cu)
//    builds for each plan with fake tensor / workspace pointers;
//  * "dispatch": per dtype pair and head dim, the block size and dynamic shared memory
//    BatchPagedAttentionPersistent (persistent.cuh) launches with.
#include "fa2_probe_common.h"

#include <cuda_runtime.h>

namespace fa2_probe {
// The plan copies its page-locked buffer to the device: nothing to copy without a GPU.
inline cudaError_t memcpy_async(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t) {
  return cudaSuccess;
}
}  // namespace fa2_probe
#define cudaMemcpyAsync fa2_probe::memcpy_async

// clang-format off
#include "batch_attention_config.inc"
#include <flashinfer/attention/persistent.cuh>
#include <flashinfer/attention/scheduler.cuh>
// clang-format on

using fa2_probe::scalar;
using fa2_probe::sentinel;

void params_fields(PersistentParams const& p) {
  scalar("q", p, p.q);
  scalar("k", p, p.k);
  scalar("v", p, p.v);
  scalar("o", p, p.o);
  scalar("partial_o", p, p.partial_o);
  scalar("partial_lse", p, p.partial_lse);
  scalar("final_o", p, p.final_o);
  scalar("final_lse", p, p.final_lse);
  scalar("q_indptr", p, p.q_indptr);
  scalar("kv_indptr", p, p.kv_indptr);
  scalar("partial_indptr", p, p.partial_indptr);
  scalar("kv_indices", p, p.kv_indices);
  scalar("q_len", p, p.q_len);
  scalar("kv_len", p, p.kv_len);
  scalar("q_start", p, p.q_start);
  scalar("kv_start", p, p.kv_start);
  scalar("kv_end", p, p.kv_end);
  scalar("kv_head_idx_arr", p, p.kv_head_idx_arr);
  scalar("work_indptr", p, p.work_indptr);
  scalar("len_kv_chunk", p, p.len_kv_chunk);
  scalar("merge_indptr", p, p.merge_indptr);
  scalar("merge_o_indices", p, p.merge_o_indices);
  scalar("num_packed_qo_len", p, p.num_packed_qo_len);
  scalar("num_kv_heads", p, p.num_kv_heads);
  fa2_probe::field("gqa_group_size", "fastdiv", p, p.gqa_group_size);
  fa2_probe::field("page_size", "fastdiv", p, p.page_size);
  scalar("q_stride_n", p, p.q_stride_n);
  scalar("q_stride_h", p, p.q_stride_h);
  scalar("k_stride_page", p, p.k_stride_page);
  scalar("k_stride_h", p, p.k_stride_h);
  scalar("k_stride_n", p, p.k_stride_n);
  scalar("v_stride_page", p, p.v_stride_page);
  scalar("v_stride_h", p, p.v_stride_h);
  scalar("v_stride_n", p, p.v_stride_n);
  scalar("k_sf_stride_page", p, p.k_sf_stride_page);
  scalar("k_sf_stride_h", p, p.k_sf_stride_h);
  scalar("k_sf_stride_n", p, p.k_sf_stride_n);
  scalar("v_sf_stride_page", p, p.v_sf_stride_page);
  scalar("v_sf_stride_h", p, p.v_sf_stride_h);
  scalar("v_sf_stride_n", p, p.v_sf_stride_n);
  scalar("sm_scale", p, p.sm_scale);
  scalar("logits_soft_cap", p, p.logits_soft_cap);
  scalar("v_scale", p, p.v_scale);
  scalar("maybe_k_cache_sf", p, p.maybe_k_cache_sf);
  scalar("maybe_v_cache_sf", p, p.maybe_v_cache_sf);
}

// Fake tensor pointers (sentinel indices) and fake workspace bases.
enum Role { kQ, kK, kV, kKVIndices, kO, kLse, kIntBuffer, kFloatBuffer };

struct Example {
  std::vector<int32_t> qo_indptr, kv_indptr, kv_len;
  uint32_t num_qo_heads, num_kv_heads, head_dim, page_size;
  bool causal;
  int kv_layout;
  double sm_scale, logits_soft_cap;
  double v_scale = 1.0;  // BatchAttention.run(v_scale=...)
};

template <class T>
void print_array(const char* name, const void* base, int64_t offset, size_t count) {
  std::vector<T> values(count);
  std::memcpy(values.data(), static_cast<const char*>(base) + offset, count * sizeof(T));
  std::printf("\"%s\": ", name);
  fa2_probe::print_vec(values);
}

void example(Example const& e) {
  const uint32_t batch = e.kv_len.size();
  // Host stand-ins for the workspace buffers (the plan only writes the page-locked one).
  static std::vector<char> page_locked(64 << 20);
  std::memset(page_locked.data(), 0, page_locked.size());
  void* int_buffer = sentinel(kIntBuffer);
  void* float_buffer = sentinel(kFloatBuffer);
  HolisticPlanInfo<2> plan_info;
  auto qo = e.qo_indptr, kv = e.kv_indptr, kv_len = e.kv_len;
  cudaError_t status = TwoStageHolisticPlan<IdType>(
      float_buffer, size_t(1) << 40, int_buffer, page_locked.data(), page_locked.size(),
      plan_info, qo.data(), kv.data(), kv_len.data(), batch, e.num_qo_heads, e.num_kv_heads,
      e.head_dim, e.causal, nullptr);
  if (status != cudaSuccess) std::abort();
  std::printf("{\"args\": {\"qo_indptr\": ");
  fa2_probe::print_vec(e.qo_indptr);
  std::printf(", \"kv_indptr\": ");
  fa2_probe::print_vec(e.kv_indptr);
  std::printf(", \"kv_len\": ");
  fa2_probe::print_vec(e.kv_len);
  std::printf(", \"num_qo_heads\": %u, \"num_kv_heads\": %u, \"head_dim\": %u, \"page_size\": "
              "%u, \"causal\": %s, \"kv_layout\": %d, \"sm_scale\": %.17g, "
              "\"logits_soft_cap\": %.17g, \"v_scale\": %.17g}, ",
              e.num_qo_heads, e.num_kv_heads, e.head_dim, e.page_size,
              e.causal ? "true" : "false", e.kv_layout, e.sm_scale, e.logits_soft_cap, e.v_scale);
  std::vector<int64_t> info = plan_info.ToVector();
  std::printf("\"plan_info\": ");
  fa2_probe::print_vec(info);
  std::printf(", \"tasks\": [");
  for (int task = 0; task < 2; ++task) {
    auto const& t = plan_info.tasks[task];
    std::vector<int32_t> work_indptr(plan_info.num_blks_y + 1);
    std::memcpy(work_indptr.data(), page_locked.data() + t.work_indptr_offset,
                work_indptr.size() * sizeof(int32_t));
    const size_t works = work_indptr.back();
    std::printf("%s{", task ? ", " : "");
    print_array<int32_t>("q_indptr", page_locked.data(), t.q_indptr_offset, works);
    std::printf(", ");
    print_array<int32_t>("kv_indptr", page_locked.data(), t.kv_indptr_offset, works);
    std::printf(", ");
    print_array<int32_t>("partial_indptr", page_locked.data(), t.partial_indptr_offset, works);
    std::printf(", ");
    print_array<int32_t>("q_len", page_locked.data(), t.q_len_offset, works);
    std::printf(", ");
    print_array<int32_t>("kv_len", page_locked.data(), t.kv_len_offset, works);
    std::printf(", ");
    print_array<int32_t>("q_start", page_locked.data(), t.q_start_offset, works);
    std::printf(", ");
    print_array<int32_t>("kv_start", page_locked.data(), t.kv_start_offset, works);
    std::printf(", ");
    print_array<int32_t>("kv_end", page_locked.data(), t.kv_end_offset, works);
    std::printf(", ");
    print_array<int32_t>("kv_head_idx", page_locked.data(), t.kv_head_idx_offset, works);
    std::printf(", \"work_indptr\": ");
    fa2_probe::print_vec(work_indptr);
    std::printf("}");
  }
  int32_t num_merged = 0;
  std::memcpy(&num_merged, page_locked.data() + plan_info.num_qo_len_offset, sizeof(int32_t));
  std::printf("], ");
  print_array<int32_t>("len_kv_chunk", page_locked.data(), plan_info.len_kv_chunk_offset, 2);
  std::printf(", ");
  print_array<int32_t>("merge_indptr", page_locked.data(), plan_info.merge_indptr_offset,
                       num_merged + 1);
  std::printf(", ");
  print_array<int32_t>("merge_o_indices", page_locked.data(), plan_info.merge_o_indices_offset,
                       num_merged);
  std::printf(", \"num_packed_qo_len\": %d", num_merged);

  // BatchPagedAttentionRun (csrc/batch_attention.cu): the two task Params. The paged K/V caches
  // are contiguous [pages, page_size, H, D] (NHD) or [pages, H, page_size, D] (HND).
  const uint32_t H = e.num_kv_heads, P = e.page_size, D = e.head_dim;
  const uint32_t stride_page = H * P * D;
  const uint32_t stride_n = e.kv_layout == 0 ? H * D : D;
  const uint32_t stride_h = e.kv_layout == 0 ? D : P * D;
  std::printf(", \"params\": [");
  for (int i = 0; i < 2; ++i) {
    alignas(PersistentParams) unsigned char storage[sizeof(PersistentParams)];
    std::memset(storage, 0, sizeof(storage));
    auto& params = *new (storage) PersistentParams;
    IdType* len_kv_chunk = GetPtrFromBaseOffset<IdType>(int_buffer, plan_info.len_kv_chunk_offset);
    auto const& t = plan_info.tasks[i];
    params.q = static_cast<DTypeQ*>(sentinel(kQ));
    params.k = static_cast<DTypeKV*>(sentinel(kK));
    params.v = static_cast<DTypeKV*>(sentinel(kV));
    params.q_indptr = GetPtrFromBaseOffset<IdType>(int_buffer, t.q_indptr_offset);
    params.kv_indptr = GetPtrFromBaseOffset<IdType>(int_buffer, t.kv_indptr_offset);
    params.partial_indptr = GetPtrFromBaseOffset<IdType>(int_buffer, t.partial_indptr_offset);
    params.kv_indices = static_cast<int*>(sentinel(kKVIndices));
    params.q_len = GetPtrFromBaseOffset<IdType>(int_buffer, t.q_len_offset);
    params.kv_len = GetPtrFromBaseOffset<IdType>(int_buffer, t.kv_len_offset);
    params.q_start = GetPtrFromBaseOffset<IdType>(int_buffer, t.q_start_offset);
    params.kv_start = GetPtrFromBaseOffset<IdType>(int_buffer, t.kv_start_offset);
    params.kv_end = GetPtrFromBaseOffset<IdType>(int_buffer, t.kv_end_offset);
    params.kv_head_idx_arr = GetPtrFromBaseOffset<IdType>(int_buffer, t.kv_head_idx_offset);
    params.work_indptr = GetPtrFromBaseOffset<IdType>(int_buffer, t.work_indptr_offset);
    params.len_kv_chunk = len_kv_chunk + i;
    params.final_o = static_cast<DTypeO*>(sentinel(kO));
    params.final_lse = static_cast<float*>(sentinel(kLse));
    params.partial_o = GetPtrFromBaseOffset<DTypeO>(float_buffer, plan_info.partial_o_offset);
    params.partial_lse = GetPtrFromBaseOffset<float>(float_buffer, plan_info.partial_lse_offset);
    params.merge_indptr = GetPtrFromBaseOffset<IdType>(int_buffer, plan_info.merge_indptr_offset);
    params.merge_o_indices =
        GetPtrFromBaseOffset<IdType>(int_buffer, plan_info.merge_o_indices_offset);
    params.num_packed_qo_len = GetPtrFromBaseOffset<IdType>(int_buffer, plan_info.num_qo_len_offset);
    params.num_kv_heads = e.num_kv_heads;
    params.gqa_group_size = uint_fastdiv(e.num_qo_heads / e.num_kv_heads);
    params.page_size = uint_fastdiv(e.page_size);
    params.q_stride_n = e.num_qo_heads * D;
    params.q_stride_h = D;
    params.k_stride_page = stride_page;
    params.k_stride_h = stride_h;
    params.k_stride_n = stride_n;
    params.v_stride_page = stride_page;
    params.v_stride_h = stride_h;
    params.v_stride_n = stride_n;
    params.k_sf_stride_page = 0;
    params.k_sf_stride_h = 0;
    params.k_sf_stride_n = 0;
    params.v_sf_stride_page = 0;
    params.v_sf_stride_h = 0;
    params.v_sf_stride_n = 0;
    params.sm_scale = e.sm_scale;
    params.v_scale = e.v_scale;
    params.logits_soft_cap = e.logits_soft_cap;
    // ADDITIONAL_PARAMS_SETTER: no k/v cache scale factors.
    params.maybe_k_cache_sf = nullptr;
    params.maybe_v_cache_sf = nullptr;
    std::printf("%s\"%s\"", i ? ", " : "", fa2_probe::hex(storage, sizeof(storage)).c_str());
  }
  std::printf("]}");
}

bool first_entry = true;
void sep() {
  std::printf(first_entry ? "\n    " : ",\n    ");
  first_entry = false;
}

// BatchPagedAttentionPersistent<128, 16, HEAD_DIM, HEAD_DIM, MASK, Variant> launch arithmetic.
template <class DTypeQ_, class DTypeKV_, uint32_t HEAD_DIM>
void dispatch(const char* dtype_q, const char* dtype_kv) {
  using AttentionVariant = StandardAttention<false>;
  constexpr uint32_t CTA_TILE_Q_1 = 128, CTA_TILE_Q_2 = 16;
  constexpr uint32_t NUM_WARPS_Q_1 = get_num_warps_q(CTA_TILE_Q_1);
  constexpr uint32_t NUM_WARPS_KV_1 = get_num_warps_kv(CTA_TILE_Q_1);
  constexpr uint32_t NUM_MMA_Q_1 = get_num_mma_q(CTA_TILE_Q_1);
  constexpr uint32_t NUM_MMA_KV_1 = 4;
  constexpr uint32_t NUM_MMA_D = HEAD_DIM / 16;
  using KTraits1 = KernelTraits<MaskMode::kNone, CTA_TILE_Q_1, NUM_MMA_Q_1, NUM_MMA_KV_1, NUM_MMA_D,
                                NUM_MMA_D, NUM_WARPS_Q_1, NUM_WARPS_KV_1, PosEncodingMode::kNone,
                                DTypeQ_, DTypeKV_, DTypeQ_, float, int32_t, AttentionVariant>;
  constexpr uint32_t NUM_WARPS_Q_2 = get_num_warps_q(CTA_TILE_Q_2);
  constexpr uint32_t NUM_WARPS_KV_2 = get_num_warps_kv(CTA_TILE_Q_2);
  constexpr uint32_t NUM_MMA_Q_2 = get_num_mma_q(CTA_TILE_Q_2);
  constexpr uint32_t NUM_MMA_KV_2 = 2;
  using KTraits2 = KernelTraits<MaskMode::kNone, CTA_TILE_Q_2, NUM_MMA_Q_2, NUM_MMA_KV_2, NUM_MMA_D,
                                NUM_MMA_D, NUM_WARPS_Q_2, NUM_WARPS_KV_2, PosEncodingMode::kNone,
                                DTypeQ_, DTypeKV_, DTypeQ_, float, int32_t, AttentionVariant>;
  constexpr uint32_t NUM_THREADS =
      KTraits1::NUM_THREADS > KTraits2::NUM_THREADS ? KTraits1::NUM_THREADS : KTraits2::NUM_THREADS;
  using ReductionKTraits =
      StateReductionKernelTraits<HEAD_DIM, 4, NUM_THREADS, DTypeQ_, DTypeQ_, int32_t>;
  size_t smem_size =
      std::max(sizeof(typename KTraits1::SharedStorage), sizeof(typename KTraits2::SharedStorage));
  smem_size = std::max(smem_size, size_t(ReductionKTraits::SMEM_SIZE));
  sep();
  std::printf("{\"dtype_q\": \"%s\", \"dtype_kv\": \"%s\", \"head_dim\": %u, \"num_threads\": %u, "
              "\"shared_mem\": %zu, \"cta_tile_q\": [%u, %u], \"num_mma_kv\": [%u, %u]}",
              dtype_q, dtype_kv, HEAD_DIM, NUM_THREADS, smem_size, CTA_TILE_Q_1, CTA_TILE_Q_2,
              NUM_MMA_KV_1, NUM_MMA_KV_2);
}

template <class DQ, class DKV>
void dispatch_all(const char* q, const char* kv) {
  dispatch<DQ, DKV, 64>(q, kv);
  dispatch<DQ, DKV, 128>(q, kv);
  dispatch<DQ, DKV, 256>(q, kv);
}

int main() {
  std::printf("{\n  \"params\": {\"param_size\": %zu, \"fields\": ", sizeof(PersistentParams));
  {
    PersistentParams p;
    params_fields(p);
  }
  fa2_probe::print_fields();
  std::printf("},\n  \"examples\": [");
  const Example examples[] = {
      // qo_indptr, kv_indptr (pages), kv_len, Hq, Hkv, D, page, causal, layout, sm_scale, cap
      {{0, 1, 2, 3, 40}, {0, 3, 5, 9, 12}, {33, 20, 60, 40}, 32, 8, 128, 16, true, 0,
       0.08838834764831845, 0.0},
      {{0, 100, 101, 400}, {0, 100, 101, 400}, {100, 101, 400}, 8, 8, 64, 1, false, 1, 0.125,
       30.0},
      {{0, 7, 300}, {0, 600, 1000}, {3000, 2000}, 16, 2, 256, 5, true, 0, 0.0625, 0.0},
      {{0, 2, 4, 6, 8, 10, 12}, {0, 9, 18, 27, 36, 45, 54}, {9, 9, 9, 9, 9, 9}, 4, 1, 128, 1,
       true, 1, 0.5, 0.0},
      {{0, 1}, {0, 512}, {8190}, 32, 4, 128, 16, false, 0, 0.08838834764831845, 0.0},
      {{0, 235, 13588}, {0, 1, 2}, {2, 1}, 28, 4, 64, 8, true, 1, 0.125, 50.0, 2.0},
      {{0, 1, 2, 5, 22}, {0, 1, 7, 8, 520}, {3, 100, 16, 8192}, 7, 1, 64, 16, false, 0, 0.125,
       0.0, 2.0},
  };
  bool first = true;
  for (auto const& e : examples) {
    std::printf(first ? "\n    " : ",\n    ");
    first = false;
    example(e);
  }
  std::printf("\n  ],\n  \"dispatch\": [");
  dispatch_all<nv_bfloat16, nv_bfloat16>("bf16", "bf16");
  dispatch_all<nv_bfloat16, __nv_fp8_e4m3>("bf16", "e4m3");
  dispatch_all<nv_bfloat16, half>("bf16", "f16");
  dispatch_all<half, nv_bfloat16>("f16", "bf16");
  dispatch_all<half, __nv_fp8_e4m3>("f16", "e4m3");
  dispatch_all<half, half>("f16", "f16");
  std::printf("\n  ]\n}\n");
  return 0;
}
