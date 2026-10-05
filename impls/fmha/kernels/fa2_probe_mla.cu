// Host-only probe for the FA2 MLA kernels of the FlashInfer 0.7.0 sm80 JIT cache
// (batch_mla_attention_* modules, BatchMLAPagedAttentionWrapper with the fa2 backend).
//
// Compiled at the compile stage against the batch_mla_config.inc that impls/fmha/fa2_build.py
// renders from csrc/batch_mla_config.jinja. Needs no GPU: device queries are answered by the
// sm_86 model of fa2_probe_common.h and the plan's host-to-device copy is not needed (MLAPlan
// only fills the page-locked buffer). Prints JSON with
//
//  * "params": sizeof and every field (offset, size, kind) of MLAParams;
//  * "examples": MLAPlan (scheduler.cuh) results for sample batches (launch grid, every work
//    and merge array, workspace offsets) and the MLAParams BatchMLAPagedAttentionRun
//    (csrc/batch_mla_run.cu) builds for them with fake tensor / workspace pointers;
//  * "dispatch": the DISPATCH_SMEM_CONFIG choice of BatchMLAPagedAttention (mla.cuh) for the
//    sm_86 shared memory per SM, and the launch's dynamic shared memory per dtype.
#include "fa2_probe_common.h"

// clang-format off
#include "batch_mla_config.inc"
#include <flashinfer/attention/mla.cuh>
#include <flashinfer/attention/scheduler.cuh>
// clang-format on

using fa2_probe::scalar;
using fa2_probe::sentinel;
using Params = MLAParams<DTypeQ, DTypeKV, DTypeO, IdType>;

void params_fields(Params const& p) {
  scalar("q_nope", p, p.q_nope);
  scalar("q_pe", p, p.q_pe);
  scalar("ckv", p, p.ckv);
  scalar("kpe", p, p.kpe);
  scalar("partial_o", p, p.partial_o);
  scalar("partial_lse", p, p.partial_lse);
  scalar("final_o", p, p.final_o);
  scalar("final_lse", p, p.final_lse);
  scalar("q_indptr", p, p.q_indptr);
  scalar("kv_indptr", p, p.kv_indptr);
  scalar("partial_indptr", p, p.partial_indptr);
  scalar("merge_packed_offset_start", p, p.merge_packed_offset_start);
  scalar("merge_packed_offset_end", p, p.merge_packed_offset_end);
  scalar("merge_partial_packed_offset_start", p, p.merge_partial_packed_offset_start);
  scalar("merge_partial_packed_offset_end", p, p.merge_partial_packed_offset_end);
  scalar("merge_partial_stride", p, p.merge_partial_stride);
  scalar("kv_indices", p, p.kv_indices);
  scalar("q_len", p, p.q_len);
  scalar("kv_len", p, p.kv_len);
  scalar("q_start", p, p.q_start);
  scalar("kv_start", p, p.kv_start);
  scalar("kv_end", p, p.kv_end);
  scalar("work_indptr", p, p.work_indptr);
  fa2_probe::field("block_size", "fastdiv", p, p.block_size);
  fa2_probe::field("num_heads", "fastdiv", p, p.num_heads);
  scalar("q_nope_stride_n", p, p.q_nope_stride_n);
  scalar("q_nope_stride_h", p, p.q_nope_stride_h);
  scalar("q_pe_stride_n", p, p.q_pe_stride_n);
  scalar("q_pe_stride_h", p, p.q_pe_stride_h);
  scalar("ckv_stride_page", p, p.ckv_stride_page);
  scalar("ckv_stride_n", p, p.ckv_stride_n);
  scalar("kpe_stride_page", p, p.kpe_stride_page);
  scalar("kpe_stride_n", p, p.kpe_stride_n);
  scalar("o_stride_n", p, p.o_stride_n);
  scalar("o_stride_h", p, p.o_stride_h);
  scalar("sm_scale", p, p.sm_scale);
  scalar("ckv_scale", p, p.ckv_scale);
  scalar("kpe_scale", p, p.kpe_scale);
  scalar("ckv_scale_arr", p, p.ckv_scale_arr);
  scalar("return_lse_base_on_e", p, p.return_lse_base_on_e);
}

enum Role { kQNope, kQPe, kCkv, kKpe, kKVIndices, kO, kLse, kIntBuffer, kFloatBuffer };

struct Example {
  std::vector<int32_t> qo_indptr, kv_indptr, kv_len;
  uint32_t num_heads, page_size;
  bool causal;
  double sm_scale;
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
  static std::vector<char> page_locked(16 << 20);
  std::memset(page_locked.data(), 0, page_locked.size());
  void* int_buffer = sentinel(kIntBuffer);
  void* float_buffer = sentinel(kFloatBuffer);
  MLAPlanInfo plan_info;
  size_t staged = 0;
  auto qo = e.qo_indptr, kv = e.kv_indptr, kv_len = e.kv_len;
  cudaError_t status = MLAPlan<IdType>(float_buffer, size_t(1) << 40, int_buffer,
                                       page_locked.data(), page_locked.size(), plan_info, staged,
                                       qo.data(), kv.data(), kv_len.data(), batch, e.num_heads,
                                       HEAD_DIM_CKV, e.causal, nullptr);
  if (status != cudaSuccess) std::abort();
  std::printf("{\"args\": {\"qo_indptr\": ");
  fa2_probe::print_vec(e.qo_indptr);
  std::printf(", \"kv_indptr\": ");
  fa2_probe::print_vec(e.kv_indptr);
  std::printf(", \"kv_len\": ");
  fa2_probe::print_vec(e.kv_len);
  std::printf(", \"num_heads\": %u, \"page_size\": %u, \"causal\": %s, \"sm_scale\": %.17g}, ",
              e.num_heads, e.page_size, e.causal ? "true" : "false", e.sm_scale);
  std::vector<int64_t> info = plan_info.ToVector();
  std::printf("\"plan_info\": ");
  fa2_probe::print_vec(info);
  std::printf(", \"staged_int_workspace_bytes\": %zu, ", staged);
  const uint32_t clusters = plan_info.num_blks_y;
  std::vector<int32_t> work_indptr(clusters + 1);
  std::memcpy(work_indptr.data(), page_locked.data() + plan_info.work_indptr_offset,
              work_indptr.size() * sizeof(int32_t));
  const size_t works = work_indptr.back();
  const char* names[] = {"q_indptr", "kv_indptr", "partial_indptr", "q_len",
                         "kv_len",   "q_start",   "kv_start",       "kv_end"};
  const int64_t offsets[] = {plan_info.q_indptr_offset, plan_info.kv_indptr_offset,
                             plan_info.partial_indptr_offset, plan_info.q_len_offset,
                             plan_info.kv_len_offset, plan_info.q_start_offset,
                             plan_info.kv_start_offset, plan_info.kv_end_offset};
  for (int i = 0; i < 8; ++i) {
    print_array<int32_t>(names[i], page_locked.data(), offsets[i], works);
    std::printf(", ");
  }
  std::printf("\"work_indptr\": ");
  fa2_probe::print_vec(work_indptr);
  const int num_sm = fa2_probe::kNumSms;
  const char* merge_names[] = {"merge_packed_offset_start", "merge_packed_offset_end",
                               "merge_partial_packed_offset_start",
                               "merge_partial_packed_offset_end", "merge_partial_stride"};
  const int64_t merge_offsets[] = {plan_info.merge_packed_offset_start_offset,
                                   plan_info.merge_packed_offset_end_offset,
                                   plan_info.merge_partial_packed_offset_start_offset,
                                   plan_info.merge_partial_packed_offset_end_offset,
                                   plan_info.merge_partial_stride_offset};
  for (int i = 0; i < 5; ++i) {
    std::printf(", ");
    print_array<int32_t>(merge_names[i], page_locked.data(), merge_offsets[i], num_sm);
  }

  // BatchMLAPagedAttentionRun: q_nope [n, H, 512], q_pe [n, H, 64], ckv [pages, P, 512],
  // kpe [pages, P, 64], o [n, H, 512], all contiguous.
  alignas(Params) unsigned char storage[sizeof(Params)];
  std::memset(storage, 0, sizeof(storage));
  auto& params = *new (storage) Params;
  const uint32_t H = e.num_heads, P = e.page_size;
  params.q_nope = static_cast<DTypeQ*>(sentinel(kQNope));
  params.q_pe = static_cast<DTypeQ*>(sentinel(kQPe));
  params.ckv = static_cast<DTypeKV*>(sentinel(kCkv));
  params.kpe = static_cast<DTypeKV*>(sentinel(kKpe));
  params.q_indptr = GetPtrFromBaseOffset<IdType>(int_buffer, plan_info.q_indptr_offset);
  params.kv_indptr = GetPtrFromBaseOffset<IdType>(int_buffer, plan_info.kv_indptr_offset);
  params.partial_indptr = GetPtrFromBaseOffset<IdType>(int_buffer, plan_info.partial_indptr_offset);
  params.kv_indices = static_cast<IdType*>(sentinel(kKVIndices));
  params.q_len = GetPtrFromBaseOffset<IdType>(int_buffer, plan_info.q_len_offset);
  params.kv_len = GetPtrFromBaseOffset<IdType>(int_buffer, plan_info.kv_len_offset);
  params.q_start = GetPtrFromBaseOffset<IdType>(int_buffer, plan_info.q_start_offset);
  params.kv_start = GetPtrFromBaseOffset<IdType>(int_buffer, plan_info.kv_start_offset);
  params.kv_end = GetPtrFromBaseOffset<IdType>(int_buffer, plan_info.kv_end_offset);
  params.work_indptr = GetPtrFromBaseOffset<IdType>(int_buffer, plan_info.work_indptr_offset);
  params.merge_packed_offset_start =
      GetPtrFromBaseOffset<IdType>(int_buffer, plan_info.merge_packed_offset_start_offset);
  params.merge_packed_offset_end =
      GetPtrFromBaseOffset<IdType>(int_buffer, plan_info.merge_packed_offset_end_offset);
  params.merge_partial_packed_offset_start =
      GetPtrFromBaseOffset<IdType>(int_buffer, plan_info.merge_partial_packed_offset_start_offset);
  params.merge_partial_packed_offset_end =
      GetPtrFromBaseOffset<IdType>(int_buffer, plan_info.merge_partial_packed_offset_end_offset);
  params.merge_partial_stride =
      GetPtrFromBaseOffset<IdType>(int_buffer, plan_info.merge_partial_stride_offset);
  params.final_o = static_cast<DTypeO*>(sentinel(kO));
  params.final_lse = static_cast<float*>(sentinel(kLse));
  params.partial_o = GetPtrFromBaseOffset<DTypeO>(float_buffer, plan_info.partial_o_offset);
  params.partial_lse = GetPtrFromBaseOffset<float>(float_buffer, plan_info.partial_lse_offset);
  params.num_heads = uint_fastdiv(H);
  params.block_size = uint_fastdiv(P);
  params.q_nope_stride_n = H * HEAD_DIM_CKV;
  params.q_nope_stride_h = HEAD_DIM_CKV;
  params.q_pe_stride_n = H * HEAD_DIM_KPE;
  params.q_pe_stride_h = HEAD_DIM_KPE;
  params.ckv_stride_page = P * HEAD_DIM_CKV;
  params.ckv_stride_n = HEAD_DIM_CKV;
  params.kpe_stride_page = P * HEAD_DIM_KPE;
  params.kpe_stride_n = HEAD_DIM_KPE;
  params.o_stride_n = H * HEAD_DIM_CKV;
  params.o_stride_h = HEAD_DIM_CKV;
  params.sm_scale = e.sm_scale;
  params.ckv_scale = 1.0f;
  params.kpe_scale = 1.0f;
  params.ckv_scale_arr = nullptr;
  params.return_lse_base_on_e = false;
  std::printf(", \"params\": \"%s\"}", fa2_probe::hex(storage, sizeof(storage)).c_str());
}

bool first_entry = true;
void sep() {
  std::printf(first_entry ? "\n    " : ",\n    ");
  first_entry = false;
}

// DISPATCH_SMEM_CONFIG returns cudaErrorNotSupported below 92672 bytes per SM.
template <class DT>
cudaError_t dispatch(const char* dtype) {
  int smem_limit_per_sm = fa2_probe::kMaxSmemPerSm;
  DISPATCH_SMEM_CONFIG(smem_limit_per_sm, NUM_STAGES, CTA_TILE_KV, QK_SHARD, {
    using KTraits = mla::KernelTraits<false, NUM_STAGES, QK_SHARD, HEAD_DIM_CKV, HEAD_DIM_KPE,
                                      64, CTA_TILE_KV, DT, DT, DT, int32_t>;
    sep();
    std::printf("{\"dtype\": \"%s\", \"num_stages\": %u, \"cta_tile_kv\": %u, \"qk_shard\": %s, "
                "\"block\": [32, 4, 2], \"shared_mem\": %zu}",
                dtype, NUM_STAGES, CTA_TILE_KV, QK_SHARD ? "true" : "false",
                sizeof(typename KTraits::SharedStorage));
  });
  return cudaSuccess;
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
      // qo_indptr, kv_indptr (pages), kv_len, heads, page, causal, sm_scale
      {{0, 1, 2, 3, 4}, {0, 3, 5, 9, 12}, {33, 20, 60, 40}, 16, 16, false, 0.07216878364870322},
      {{0, 4, 8}, {0, 600, 1000}, {600, 400}, 128, 1, true, 0.1},
      {{0, 1, 70}, {0, 512, 520}, {8190, 120}, 16, 16, true, 0.0625},
      {{0, 1}, {0, 256}, {4096}, 128, 16, false, 0.07216878364870322},
      {{0, 17, 34, 51}, {0, 0, 17, 531}, {0, 17, 514}, 64, 1, false, 0.07216878364870322},
      {{0, 7, 14, 21, 28, 35, 42, 49}, {0, 7, 14, 21, 28, 35, 42, 49},
       {197, 197, 197, 197, 197, 197, 197}, 16, 32, true, 0.07216878364870322},
  };
  bool first = true;
  for (auto const& e : examples) {
    std::printf(first ? "\n    " : ",\n    ");
    first = false;
    example(e);
  }
  std::printf("\n  ],\n  \"dispatch\": [");
  if (dispatch<nv_bfloat16>("bf16") != cudaSuccess || dispatch<half>("f16") != cudaSuccess) {
    std::abort();
  }
  std::printf("\n  ]\n}\n");
  return 0;
}
