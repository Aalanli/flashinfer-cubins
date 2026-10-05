// Host-only probe for the sm_100a CUTLASS MLA kernel parameter struct
// (Sm100FmhaMlaKernelTmaWarpspecialized<...>::Params, passed by value).
//
// Builds the Arguments exactly as the previous native host code did
// (impls/templates/dsa_attention_sm100a.cu: flashinfer args_from_options followed by
// the DSA overrides, split_kv = 1), then calls the upstream
// Kernel::to_underlying_arguments, Kernel::get_grid_shape and MLA::can_implement.
// cuTensorMapEncodeTiled is replaced by a recorder (CUTLASS direct driver call), so the
// probe needs neither a GPU nor libcuda: each call's arguments are printed and the
// produced 128-byte descriptor is filled with a marker that locates it in Params.
// Usage: probe <batch> <page_count> <sm_count> <device_id> <softmax_scale>
#define CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL 1
#include <cuda.h>

#include <string>
#include <vector>

#include "probe_common.h"

namespace {
std::vector<std::string> tma_calls;
}

extern "C" CUresult CUDAAPI cuTensorMapEncodeTiled(
    CUtensorMap* tensorMap, CUtensorMapDataType tensorDataType, cuuint32_t tensorRank,
    void* globalAddress, const cuuint64_t* globalDim, const cuuint64_t* globalStrides,
    const cuuint32_t* boxDim, const cuuint32_t* elementStrides, CUtensorMapInterleave interleave,
    CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion,
    CUtensorMapFloatOOBfill oobFill) {
  auto list = [](auto const* values, unsigned n) {
    std::string s = "[";
    for (unsigned i = 0; i < n; ++i)
      s += (i ? ", " : "") + std::to_string((unsigned long long)values[i]);
    return s + "]";
  };
  char head[256];
  std::snprintf(head, sizeof(head),
                "{\"data_type\": %d, \"rank\": %u, \"address\": %llu, ", int(tensorDataType),
                tensorRank, (unsigned long long)(uintptr_t)globalAddress);
  std::string call = head;
  call += "\"global_dims\": " + list(globalDim, tensorRank);
  call += ", \"global_strides\": " + list(globalStrides, tensorRank - 1);
  call += ", \"box_dims\": " + list(boxDim, tensorRank);
  call += ", \"element_strides\": " + list(elementStrides, tensorRank);
  char tail[160];
  std::snprintf(tail, sizeof(tail),
                ", \"interleave\": %d, \"swizzle\": %d, \"l2_promotion\": %d, \"oob_fill\": %d}",
                int(interleave), int(swizzle), int(l2Promotion), int(oobFill));
  call += tail;
  // Marker: word 0 = 0x7e5a000000000000 | call index; words 1..15 zero.
  std::memset(tensorMap, 0, sizeof(*tensorMap));
  reinterpret_cast<uint64_t*>(tensorMap)[0] = 0x7e5a000000000000ull | tma_calls.size();
  tma_calls.push_back(call);
  return CUDA_SUCCESS;
}

#include <flashinfer/attention/cutlass_mla.cuh>

int main(int argc, char** argv) {
  using namespace cute;
  using dsa_probe::field;
  using dsa_probe::sentinel;
  using DsaElement = cutlass::bfloat16_t;
  using DsaMla = flashinfer::attention::MlaSm100<DsaElement>;
  using Kernel = DsaMla::FmhaKernel;
  using Params = Kernel::Params;
  if (argc != 6) return 2;
  const int batch = std::atoi(argv[1]);
  const int page_count = std::atoi(argv[2]);
  const int sm_count = std::atoi(argv[3]);
  const int device_id = std::atoi(argv[4]);
  const float softmax_scale = std::strtof(argv[5], nullptr);

  // Pointer roles (sentinel index): 0 padded q latent, 1 padded q rope, 2 ckv cache,
  // 3 kpe cache, 4 lengths, 5 page table, 6 padded output, 7 padded lse.
  auto* q = (DsaElement*)sentinel(0);
  auto* qp = (DsaElement*)sentinel(1);
  auto* ckv = (DsaElement*)sentinel(2);
  auto* kpe = (DsaElement*)sentinel(3);
  auto options = flashinfer::attention::args_from_options<DsaMla>(
      sentinel(6), sentinel(7), q, ckv, sentinel(4), sentinel(5), batch, 2048, page_count, 1,
      device_id);
  // args_from_options queried the (absent) device; use the given SM count instead.
  options.hw_info.device_id = device_id;
  options.hw_info.sm_count = sm_count;
  options.mainloop.softmax_scale = softmax_scale;
  options.mainloop.ptr_q_latent = q;
  options.mainloop.stride_q_latent = make_stride(int64_t(512), _1{}, int64_t(128 * 512));
  options.mainloop.ptr_q_rope = qp;
  options.mainloop.stride_q_rope = make_stride(int64_t(64), _1{}, int64_t(128 * 64));
  options.mainloop.ptr_c_latent = ckv;
  options.mainloop.stride_c_latent = make_stride(int64_t(512), _1{}, int64_t(512));
  options.mainloop.ptr_k_rope = kpe;
  options.mainloop.stride_k_rope = make_stride(int64_t(64), _1{}, int64_t(64));
  options.epilogue.ptr_lse = (float*)sentinel(7);
  options.split_kv = 1;
  if (DsaMla::Fmha::can_implement(options) != cutlass::Status::kSuccess) {
    std::fprintf(stderr, "can_implement failed\n");
    return 1;
  }

  alignas(Params) unsigned char storage[sizeof(Params)];
  std::memset(storage, 0, sizeof(storage));
  // As in cutlass::fmha::device::MLA::initialize: split_kv == 1 needs no workspace.
  auto* p = new (storage) Params(Kernel::to_underlying_arguments(options, nullptr));
  dim3 grid = Kernel::get_grid_shape(*p);
  dim3 block = Kernel::get_block_shape();

  auto& r = *p;
  field("problem_shape.K", "i32", r, get<1>(r.problem_shape));
  field("problem_shape.B", "i32", r, get<3>(r.problem_shape));
  field("mainloop.softmax_scale", "f32", r, r.mainloop.softmax_scale);
  field("mainloop.ptr_q_latent", "ptr", r, r.mainloop.ptr_q_latent);
  field("mainloop.stride_q_latent.0", "i64", r, get<0>(r.mainloop.stride_q_latent));
  field("mainloop.stride_q_latent.2", "i64", r, get<2>(r.mainloop.stride_q_latent));
  field("mainloop.ptr_q_rope", "ptr", r, r.mainloop.ptr_q_rope);
  field("mainloop.stride_q_rope.0", "i64", r, get<0>(r.mainloop.stride_q_rope));
  field("mainloop.stride_q_rope.2", "i64", r, get<2>(r.mainloop.stride_q_rope));
  field("mainloop.ptr_c_latent", "ptr", r, r.mainloop.ptr_c_latent);
  field("mainloop.stride_c_latent.0", "i64", r, get<0>(r.mainloop.stride_c_latent));
  field("mainloop.stride_c_latent.2", "i64", r, get<2>(r.mainloop.stride_c_latent));
  field("mainloop.ptr_k_rope", "ptr", r, r.mainloop.ptr_k_rope);
  field("mainloop.stride_k_rope.0", "i64", r, get<0>(r.mainloop.stride_k_rope));
  field("mainloop.stride_k_rope.2", "i64", r, get<2>(r.mainloop.stride_k_rope));
  field("mainloop.ptr_seq", "ptr", r, r.mainloop.ptr_seq);
  field("mainloop.ptr_page_table", "ptr", r, r.mainloop.ptr_page_table);
  field("mainloop.stride_page_table.1", "i32", r, get<1>(r.mainloop.stride_page_table));
  field("mainloop.page_count", "i32", r, r.mainloop.page_count);
  field("mainloop.page_size", "i32", r, r.mainloop.page_size);
  field("epilogue.ptr_o", "ptr", r, r.epilogue.ptr_o);
  field("epilogue.ptr_o_acc", "ptr", r, r.epilogue.ptr_o_acc);
  field("epilogue.stride_o.0", "i64", r, get<0>(r.epilogue.stride_o));
  field("epilogue.stride_o.2", "i64", r, get<2>(r.epilogue.stride_o));
  field("epilogue.stride_o_acc.0", "i64", r, get<0>(r.epilogue.stride_o_acc));
  field("epilogue.stride_o_acc.2", "i64", r, get<2>(r.epilogue.stride_o_acc));
  field("epilogue.ptr_lse", "ptr", r, r.epilogue.ptr_lse);
  field("epilogue.ptr_lse_acc", "ptr", r, r.epilogue.ptr_lse_acc);
  field("epilogue.stride_lse.1", "i32", r, get<1>(r.epilogue.stride_lse));
  field("epilogue.stride_lse_acc.1", "i32", r, get<1>(r.epilogue.stride_lse_acc));
  field("epilogue.output_scale", "f32", r, r.epilogue.output_scale);
  field("mainloop_params.tma_load_q_latent", "tma_atom", r, r.mainloop_params.tma_load_q_latent);
  field("mainloop_params.tma_load_q_rope", "tma_atom", r, r.mainloop_params.tma_load_q_rope);
  field("mainloop_params.tma_load_c_latent", "tma_atom", r, r.mainloop_params.tma_load_c_latent);
  field("mainloop_params.tma_load_k_rope", "tma_atom", r, r.mainloop_params.tma_load_k_rope);
  field("mainloop_params.tma_load_c_latent_transpose", "tma_atom", r,
        r.mainloop_params.tma_load_c_latent_transpose);
  field("tile_scheduler.num_blocks", "i32", r, r.tile_scheduler.num_blocks);
  field("tile_scheduler.divmod_m_block", "fast_divmod", r, r.tile_scheduler.divmod_m_block);
  field("tile_scheduler.divmod_b", "fast_divmod", r, r.tile_scheduler.divmod_b);
  field("tile_scheduler.divmod_split_kv", "fast_divmod", r, r.tile_scheduler.divmod_split_kv);
  field("tile_scheduler.hw_info.device_id", "i32", r, r.tile_scheduler.hw_info.device_id);
  field("tile_scheduler.hw_info.sm_count", "i32", r, r.tile_scheduler.hw_info.sm_count);
  field("tile_scheduler.hw_info.max_active_clusters", "i32", r,
        r.tile_scheduler.hw_info.max_active_clusters);
  field("tile_scheduler.hw_info.cluster_shape", "dim3", r, r.tile_scheduler.hw_info.cluster_shape);
  field("tile_scheduler.hw_info.cluster_shape_fallback", "dim3", r,
        r.tile_scheduler.hw_info.cluster_shape_fallback);
  field("split_kv", "i32", r, r.split_kv);
  field("ptr_split_kv", "ptr", r, r.ptr_split_kv);

  std::printf("{\n  \"param_size\": %zu,\n", sizeof(Params));
  std::printf("  \"constants\": {\"shared_mem\": %d, \"block\": [%u, %u, %u], "
              "\"cluster\": [%d, %d, %d], \"heads\": %d, \"max_seq_len\": %d, "
              "\"workspace_bytes_split1_per_batch\": %zu},\n",
              Kernel::SharedStorageSize, block.x, block.y, block.z,
              int(size<0>(Kernel::ClusterShape{})), int(size<1>(Kernel::ClusterShape{})),
              int(size<2>(Kernel::ClusterShape{})), int(get<0>(r.problem_shape)),
              get<1>(r.problem_shape), Kernel::get_workspace_size(options) / batch);
  dsa_probe::print_fields();
  std::printf(",\n  \"example\": {\"batch\": %d, \"page_count\": %d, \"sm_count\": %d, "
              "\"device_id\": %d, \"softmax_scale\": %.9g, \"grid\": [%u, %u, %u],\n"
              "    \"bytes\": \"%s\",\n    \"tma_calls\": [\n",
              batch, page_count, sm_count, device_id, softmax_scale, grid.x, grid.y, grid.z,
              dsa_probe::hex(storage, sizeof(storage)).c_str());
  for (size_t i = 0; i < tma_calls.size(); ++i)
    std::printf("      %s%s\n", tma_calls[i].c_str(), i + 1 < tma_calls.size() ? "," : "");
  std::printf("    ]}\n}\n");
  return 0;
}
