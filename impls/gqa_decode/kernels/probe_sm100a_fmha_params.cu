// Host-only probe for the sm_100a CUTLASS FMHA kernel parameter struct
// (Sm100FmhaFwdKernelTmaWarpspecialized<...>::Params, passed by value; see sm100a_fmha.cuh).
//
// Builds the Arguments exactly as flashinfer::FwdRunner::run does for the call the previous
// native host code made (impls/templates/gqa_decode_sm100a.cu: run_fmha_fwd with 32 query
// heads, 4 KV heads, head dim 128, packed K/V strides 512/128, max_qo_len = 1, batch query
// tokens, total_kv packed rows), then calls the upstream Operation::can_implement,
// Kernel::to_underlying_arguments, get_grid_shape and get_block_shape (FMHA::initialize/run
// without the CUDA runtime calls). The hardware SM count is an argument instead of a device
// query. Needs neither a GPU nor libcuda:
//
// * cuTensorMapEncodeTiled is interposed (CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL): every
//   call's arguments are printed and the 128-byte descriptor is filled with a deterministic
//   packing of them (fake_tensor_map; harness/workloads/gqa_decode.py fake_encode mirrors it),
//   so a byte comparison of Params covers every encode argument;
// * cudaDriverGetVersion is interposed (shared cudart) so CUTLASS's driver-dependent
//   descriptor fix-up (bit 21 of word 1 for tensors under 128 KiB on drivers <= 13.1) is
//   exercised deterministically for the given version.
//
// Usage: probe <batch> <total_kv> <sm_count> <sm_scale> <driver_version>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <string>
#include <utility>
#include <vector>

#include "probe_common.h"
#include "sm100a_fmha.cuh"

#ifndef CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL
#error "the probe interposes cuTensorMapEncodeTiled; build with -DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL"
#endif

namespace {

std::vector<std::string> g_calls;
std::vector<std::vector<unsigned char>> g_descriptors;  // fake bytes written by each call
int g_driver_version = 0;

template <class T>
std::string list(const T* values, unsigned n) {
  std::string s = "[";
  for (unsigned i = 0; i < n; ++i) s += (i ? ", " : "") + std::to_string((unsigned long long)values[i]);
  return s + "]";
}

}  // namespace

// Deterministic stand-in for the opaque CUtensorMap: little-endian
//   u8 data_type, u8 rank, u8 interleave, u8 swizzle, u8 l2, u8 oob, u16 0,
//   u64 address, u64 dims[5], u64 strides[4], u32 box[5], u32 element_strides[5]
// (unused trailing entries are zero), exactly 128 bytes.
extern "C" CUresult CUDAAPI cuTensorMapEncodeTiled(
    CUtensorMap* tensorMap, CUtensorMapDataType tensorDataType, cuuint32_t tensorRank,
    void* globalAddress, const cuuint64_t* globalDim, const cuuint64_t* globalStrides,
    const cuuint32_t* boxDim, const cuuint32_t* elementStrides, CUtensorMapInterleave interleave,
    CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion,
    CUtensorMapFloatOOBfill oobFill) {
  if (tensorRank < 1 || tensorRank > 5) return CUDA_ERROR_INVALID_VALUE;
  unsigned char bytes[128] = {};
  bytes[0] = (unsigned char)tensorDataType;
  bytes[1] = (unsigned char)tensorRank;
  bytes[2] = (unsigned char)interleave;
  bytes[3] = (unsigned char)swizzle;
  bytes[4] = (unsigned char)l2Promotion;
  bytes[5] = (unsigned char)oobFill;
  uint64_t address = (uint64_t)(uintptr_t)globalAddress;
  std::memcpy(bytes + 8, &address, 8);
  for (unsigned i = 0; i < tensorRank; ++i) {
    uint64_t dim = globalDim[i];
    uint32_t box = boxDim[i], element = elementStrides[i];
    std::memcpy(bytes + 16 + 8 * i, &dim, 8);
    std::memcpy(bytes + 88 + 4 * i, &box, 4);
    std::memcpy(bytes + 108 + 4 * i, &element, 4);
    if (i + 1 < tensorRank) {
      uint64_t stride = globalStrides[i];
      std::memcpy(bytes + 56 + 8 * i, &stride, 8);
    }
  }
  static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap must be 128 bytes");
  std::memcpy(tensorMap, bytes, sizeof(bytes));
  g_descriptors.emplace_back(bytes, bytes + sizeof(bytes));
  char head[160];
  std::snprintf(head, sizeof(head), "{\"data_type\": %d, \"rank\": %u, \"address\": %llu, ",
                int(tensorDataType), tensorRank, (unsigned long long)address);
  std::string call = head;
  call += "\"global_dims\": " + list(globalDim, tensorRank);
  call += ", \"global_strides\": " + list(globalStrides, tensorRank - 1);
  call += ", \"box_dims\": " + list(boxDim, tensorRank);
  call += ", \"element_strides\": " + list(elementStrides, tensorRank);
  char tail[128];
  std::snprintf(tail, sizeof(tail),
                ", \"interleave\": %d, \"swizzle\": %d, \"l2_promotion\": %d, \"oob_fill\": %d}",
                int(interleave), int(swizzle), int(l2Promotion), int(oobFill));
  g_calls.push_back(call + tail);
  return CUDA_SUCCESS;
}

extern "C" cudaError_t CUDARTAPI cudaDriverGetVersion(int* driverVersion) {
  *driverVersion = g_driver_version;
  return cudaSuccess;
}

namespace {

using gqa_probe::scalar;

template <class T>
struct is_varlen : std::false_type {};
template <>
struct is_varlen<cutlass::fmha::collective::VariableLength> : std::true_type {};

// Every non-empty leaf of a (nested) cute tuple / layout, named prefix.i.j...
template <class Base, class T>
void leaves(std::string const& name, Base const& base, T const& value);

template <class Base, class T, size_t... I>
void tuple_leaves(std::string const& name, Base const& base, T const& value,
                  std::index_sequence<I...>) {
  (leaves(name + "." + std::to_string(I), base, cute::get<I>(value)), ...);
}

template <class Base, class T>
void leaves(std::string const& name, Base const& base, T const& value) {
  if constexpr (is_varlen<T>::value) {
    scalar(name, base, value.segment_offsets);
  } else if constexpr (cute::is_tuple<T>::value) {
    tuple_leaves(name, base, value, std::make_index_sequence<cute::tuple_size<T>::value>{});
  } else if constexpr (std::is_empty_v<T>) {
    // static (compile-time) value: no storage
  } else {
    scalar(name, base, value);
  }
}

template <class Base, class L>
void layout(std::string const& name, Base const& base, L const& value) {
  leaves(name + ".shape", base, value.shape());
  leaves(name + ".stride", base, value.stride());
}

std::vector<std::pair<std::string, int>> g_tma_fields;  // field -> encode call index

// A TMA copy atom: its 128-byte descriptor plus (static, hence empty) auxiliary strides.
// The descriptor is matched to the encode call that produced it (up to CUTLASS's bit-21
// driver fix-up of word 1, i.e. bit 5 of byte 10).
template <class Base, class Atom>
void tma(std::string const& name, Base const& base, Atom const& atom) {
  gqa_probe::field(name, "tma", base, *atom.get_tma_descriptor());
  auto* desc = reinterpret_cast<const unsigned char*>(atom.get_tma_descriptor());
  int match = -1;
  for (size_t i = 0; i < g_descriptors.size(); ++i) {
    bool same = true;
    for (size_t b = 0; b < 128 && same; ++b) {
      unsigned char mask = b == 10 ? 0xdf : 0xff;
      same = (desc[b] & mask) == (g_descriptors[i][b] & mask);
    }
    if (same) {
      if (match >= 0) {
        std::fprintf(stderr, "%s matches several encode calls\n", name.c_str());
        std::exit(1);
      }
      match = int(i);
    }
  }
  if (match < 0) {
    std::fprintf(stderr, "%s matches no encode call\n", name.c_str());
    std::exit(1);
  }
  g_tma_fields.emplace_back(name, match);
  static_assert(std::is_empty_v<std::remove_cv_t<std::remove_reference_t<decltype(atom.aux_params_.g_stride_)>>>,
                "TMA auxiliary strides are expected to be static");
}

}  // namespace

int main(int argc, char** argv) {
  using namespace cute;
  using gqa_probe::sentinel;
  using namespace gqa_decode;
  using Params = Kernel::Params;
  if (argc != 6) return 2;
  const int batch = std::atoi(argv[1]);
  const int total_kv_len = std::atoi(argv[2]);
  const int sm_count = std::atoi(argv[3]);
  const float sm_scale = std::strtof(argv[4], nullptr);
  g_driver_version = std::atoi(argv[5]);

  // run_fmha_fwd arguments of the previous native launcher.
  const int num_qo_heads = 32, num_kv_heads = 4, head_dim_qk = 128, head_dim_vo = 128;
  const int q_stride_n = num_qo_heads * head_dim_qk, q_stride_h = head_dim_qk;
  const int k_stride_n = num_kv_heads * head_dim_qk, k_stride_h = head_dim_qk;
  const int v_stride_n = num_kv_heads * head_dim_vo, v_stride_h = head_dim_vo;
  const int batch_size = batch, total_qo_len = batch, max_qo_len = 1;
  const double q_scale = 1., k_scale = 1., v_scale = 1., o_scale = 1.;

  // Pointer roles (sentinel index): 0 q, 1 packed_k, 2 packed_v, 3 qo_offsets, 4 kv_offsets,
  // 5 work_indptr, 6 qo_tile_indices, 7 qo_head_indices, 8 batch_indices, 9 o (row 0 of the
  // output; the upstream epilogue addresses it from o - max_qo_len rows), 10 lse.
  auto* q = (Element*)sentinel(0);
  auto* k = (Element*)sentinel(1);
  auto* v = (Element*)sentinel(2);
  auto* qo_segment_offsets = (int*)sentinel(3);
  auto* kv_segment_offsets = (int*)sentinel(4);
  auto* work_indptr = (int*)sentinel(5);
  auto* qo_tile_indices = (int*)sentinel(6);
  auto* qo_head_indices = (int*)sentinel(7);
  auto* batch_indices = (int*)sentinel(8);
  auto* o = (Element*)sentinel(9);
  auto* maybe_lse = (float*)sentinel(10);

  // ---- verbatim body of flashinfer::FwdRunner::run up to Operation::initialize ----
  using StrideQ = Runner::StrideQ;
  using StrideK = Runner::StrideK;
  using StrideV = Runner::StrideV;
  using StrideO = Runner::StrideO;
  using StrideLSE = Runner::StrideLSE;
  using ProblemShapeVarlen = Runner::ProblemShapeVarlen;
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = sm_count;  // upstream: query_device_multiprocessor_count(0)

  StrideQ stride_Q;
  StrideK stride_K;
  StrideV stride_V;
  StrideO stride_O;
  StrideLSE stride_LSE;

  int h_r = num_qo_heads / num_kv_heads;
  ProblemShapeVarlen problem_shape = cute::make_tuple(
      cutlass::fmha::collective::VariableLength{qo_segment_offsets},
      cutlass::fmha::collective::VariableLength{kv_segment_offsets}, head_dim_qk,
      cute::make_tuple(cute::make_tuple(h_r, num_kv_heads), batch_size));

  stride_Q = make_stride(q_stride_n, _1{}, make_stride(q_stride_h, h_r * q_stride_h));
  stride_O = make_stride(
      num_qo_heads * head_dim_vo, _1{},
      make_stride(make_stride(head_dim_vo, h_r * head_dim_vo), num_qo_heads * head_dim_vo));
  stride_K = make_stride(k_stride_n, _1{}, make_stride(_0{}, k_stride_h));
  stride_V = make_stride(_1{}, v_stride_n, make_stride(_0{}, v_stride_h));
  stride_LSE = make_stride(num_qo_heads, make_stride(_1{}, h_r));

  auto shape_Q = make_shape(total_qo_len, head_dim_qk, make_shape(h_r, num_kv_heads));
  auto shape_O = make_shape(max_qo_len, head_dim_vo,
                            make_shape(make_shape(h_r, num_kv_heads), max_qo_len + total_qo_len));
  auto shape_K = make_shape(total_kv_len, head_dim_qk, make_shape(h_r, num_kv_heads));
  auto shape_V = make_shape(head_dim_vo, total_kv_len, make_shape(h_r, num_kv_heads));
  auto shape_LSE = make_shape(total_qo_len, make_shape(h_r, num_kv_heads));

  Runner::LayoutQ layout_Q = make_layout(shape_Q, stride_Q);
  Runner::LayoutK layout_K = make_layout(shape_K, stride_K);
  Runner::LayoutV layout_V = make_layout(shape_V, stride_V);
  Runner::LayoutO layout_O = make_layout(shape_O, stride_O);
  Runner::LayoutLSE layout_LSE = make_layout(shape_LSE, stride_LSE);

  // The upstream braced initializer narrows the double scales to float implicitly.
  typename Operation::Arguments arguments{
      problem_shape,
      {q, layout_Q, k, layout_K, v, layout_V, float(sm_scale), float(q_scale), float(k_scale),
       float(v_scale), float(o_scale)},
      {o - max_qo_len * get<0>(stride_O), layout_O, maybe_lse, layout_LSE, max_qo_len},
      {work_indptr, qo_tile_indices, qo_head_indices, batch_indices},
      hw_info};
  // ---- end of the verbatim part ----

  if (Operation::get_workspace_size(arguments) != 0) {
    std::fprintf(stderr, "unexpected workspace\n");
    return 1;
  }
  if (Operation::can_implement(arguments) != cutlass::Status::kSuccess) {
    std::fprintf(stderr, "can_implement failed\n");
    return 1;
  }
  alignas(Params) unsigned char storage[sizeof(Params)];
  std::memset(storage, 0, sizeof(storage));
  auto* p = new (storage) Params(Kernel::to_underlying_arguments(arguments, nullptr));
  dim3 grid = Operation::get_grid_shape(*p);
  dim3 block = Kernel::get_block_shape();

  auto& r = *p;
  leaves("problem_shape", r, r.problem_shape);
  tma("mainloop.load.tma_load_Q", r, r.mainloop.load.tma_load_Q);
  layout("mainloop.load.layout_Q", r, r.mainloop.load.layout_Q);
  tma("mainloop.load.tma_load_K", r, r.mainloop.load.tma_load_K);
  layout("mainloop.load.layout_K", r, r.mainloop.load.layout_K);
  tma("mainloop.load.tma_load_V", r, r.mainloop.load.tma_load_V);
  layout("mainloop.load.layout_V", r, r.mainloop.load.layout_V);
  scalar("mainloop.scale_softmax", r, r.mainloop.scale_softmax);
  scalar("mainloop.scale_softmax_log2", r, r.mainloop.scale_softmax_log2);
  scalar("mainloop.scale_output", r, r.mainloop.scale_output);
  tma("epilogue.tma_store_o", r, r.epilogue.tma_store_o);
  layout("epilogue.layout_O", r, r.epilogue.layout_O);
  scalar("epilogue.ptr_LSE", r, r.epilogue.ptr_LSE);
  layout("epilogue.layout_LSE", r, r.epilogue.layout_LSE);
  scalar("epilogue.max_qo_len", r, r.epilogue.max_qo_len);
  scalar("tile_scheduler.work_indptr", r, r.tile_scheduler.work_indptr);
  scalar("tile_scheduler.qo_tile_indices", r, r.tile_scheduler.qo_tile_indices);
  scalar("tile_scheduler.qo_head_indices", r, r.tile_scheduler.qo_head_indices);
  scalar("tile_scheduler.batch_indices", r, r.tile_scheduler.batch_indices);
  scalar("tile_scheduler.num_sm", r, r.tile_scheduler.num_sm);

  std::printf("{\n  \"param_size\": %zu,\n", sizeof(Params));
  std::printf("  \"constants\": {\"shared_mem\": %d, \"block\": [%u, %u, %u], "
              "\"cluster\": [%d, %d, %d], \"tile_q\": %d, \"tile_kv\": %d, "
              "\"log2_e\": %.9g},\n",
              int(Kernel::SharedStorageSize), block.x, block.y, block.z,
              int(size<0>(Kernel::ClusterShape{})), int(size<1>(Kernel::ClusterShape{})),
              int(size<2>(Kernel::ClusterShape{})), int(get<0>(Kernel::TileShape{})),
              int(get<1>(Kernel::TileShape{})), static_cast<float>(std::log2(std::exp(1.0))));
  gqa_probe::print_fields();
  std::printf(",\n  \"example\": {\"batch\": %d, \"total_kv\": %d, \"sm_count\": %d, "
              "\"sm_scale\": %.9g, \"driver_version\": %d, \"grid\": [%u, %u, %u],\n"
              "    \"bytes\": \"%s\",\n",
              batch, total_kv_len, sm_count, sm_scale, g_driver_version, grid.x, grid.y, grid.z,
              gqa_probe::hex(storage, sizeof(storage)).c_str());
  std::string tma_fields;
  for (auto const& [name, index] : g_tma_fields)
    tma_fields += (tma_fields.empty() ? "" : ", ") + ("\"" + name + "\": " + std::to_string(index));
  std::printf("    \"tma_fields\": {%s},\n    \"tma_calls\": [\n", tma_fields.c_str());
  for (size_t i = 0; i < g_calls.size(); ++i)
    std::printf("      %s%s\n", g_calls[i].c_str(), i + 1 < g_calls.size() ? "," : "");
  std::printf("    ]}\n}\n");
  return 0;
}
