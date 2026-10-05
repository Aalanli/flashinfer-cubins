// SPDX-License-Identifier: Apache-2.0
// Compile-time, host-only probe for the nvfp4_dual_gemm GEMM kernel's by-value
// Params (adapted from impls/fp8_gemm/kernels/fp8_gemm_probe.cu).
//
// Built and run by impls/nvfp4_dual_gemm/compiler.py; it is never part of a
// benchmark run, launches nothing and needs neither a GPU nor libcuda. Two
// modes, both printing one JSON document to stdout:
//
//   nvfp4_dual_gemm_probe layout
//       sizeof/offsetof of every Params field the Python builder writes, the
//       kernel's compile-time launch constants and enum values.
//
//   nvfp4_dual_gemm_probe params M N K L A B SFA SFB D ALPHA DRIVER_VERSION
//       Builds the Arguments exactly as FlashInfer's
//       prepareGemmArgs_float_128_128_256_1_1_1_1SM (fp4_gemm_template_sm100.h)
//       does, runs CUTLASS's own GemmKernel::to_underlying_arguments with fake
//       (aligned, never dereferenced) device addresses, and prints the
//       resulting Params bytes, the launch configuration
//       (get_grid_shape/get_block_shape/SharedStorageSize), can_implement, the
//       workspace size and every cuTensorMapEncodeTiled call CUTLASS made.
//
// cuTensorMapEncodeTiled is interposed: the real driver refuses to encode on
// pre-Hopper devices (CUDA_ERROR_NOT_SUPPORTED on sm_86) and the descriptor
// format is opaque, so the probe records the exact encode arguments and fills
// the 128-byte descriptor with a deterministic packing of those arguments
// (fake_tensor_map below; harness/workloads/nvfp4_dual_gemm.py has the
// identical Python packing for tests). A Params byte comparison against these
// fixtures therefore checks every non-descriptor byte and every encode
// argument. cudaDriverGetVersion is interposed too (the probe links the shared
// cudart) so that CUTLASS's driver-version-dependent descriptor fix-up is
// exercised deterministically for the DRIVER_VERSION given on the command line.
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <type_traits>
#include <vector>

#include "nvfp4_dual_gemm_sm100.cuh"

#ifndef CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL
#error "the probe interposes cuTensorMapEncodeTiled; build with -DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL"
#endif

namespace {

struct EncodeCall {
  int data_type, rank, interleave, swizzle, l2_promotion, oob_fill;
  uint64_t address;
  std::vector<uint64_t> dims, strides;
  std::vector<uint32_t> box, element_strides;
};
std::vector<EncodeCall> g_calls;
size_t g_calls_first = 0;
int g_driver_version = 0;

// Fill a large stack region with `fill` (see params_mode).
__attribute__((noinline)) void poison_stack(unsigned char fill) {
  volatile unsigned char region[1 << 16];
  for (size_t i = 0; i < sizeof(region); ++i) region[i] = fill;
}

std::string hex(const void* data, size_t size) {
  static const char* digits = "0123456789abcdef";
  std::string out;
  auto* bytes = static_cast<const unsigned char*>(data);
  for (size_t i = 0; i < size; ++i) {
    out += digits[bytes[i] >> 4];
    out += digits[bytes[i] & 15];
  }
  return out;
}

template <class T>
std::string list(const std::vector<T>& values) {
  std::string out = "[";
  for (size_t i = 0; i < values.size(); ++i) {
    out += (i ? ", " : "") + std::to_string(values[i]);
  }
  return out + "]";
}

// Deterministic stand-in for the opaque CUtensorMap: little-endian
//   u8 data_type, u8 rank, u8 interleave, u8 swizzle, u8 l2, u8 oob, u16 0,
//   u64 address, u64 dims[5], u64 strides[4], u32 box[5], u32 element_strides[5]
// (unused trailing entries are zero), exactly 128 bytes.
void fake_tensor_map(const EncodeCall& c, CUtensorMap* out) {
  unsigned char bytes[128] = {};
  bytes[0] = static_cast<unsigned char>(c.data_type);
  bytes[1] = static_cast<unsigned char>(c.rank);
  bytes[2] = static_cast<unsigned char>(c.interleave);
  bytes[3] = static_cast<unsigned char>(c.swizzle);
  bytes[4] = static_cast<unsigned char>(c.l2_promotion);
  bytes[5] = static_cast<unsigned char>(c.oob_fill);
  std::memcpy(bytes + 8, &c.address, 8);
  for (size_t i = 0; i < c.dims.size(); ++i) std::memcpy(bytes + 16 + 8 * i, &c.dims[i], 8);
  for (size_t i = 0; i < c.strides.size(); ++i) std::memcpy(bytes + 56 + 8 * i, &c.strides[i], 8);
  for (size_t i = 0; i < c.box.size(); ++i) std::memcpy(bytes + 88 + 4 * i, &c.box[i], 4);
  for (size_t i = 0; i < c.element_strides.size(); ++i) {
    std::memcpy(bytes + 108 + 4 * i, &c.element_strides[i], 4);
  }
  static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap must be 128 bytes");
  std::memcpy(out, bytes, sizeof(bytes));
}

}  // namespace

// Interposed driver entry point (CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL makes
// CUTLASS call this symbol directly instead of cudaGetDriverEntryPoint).
extern "C" CUresult CUDAAPI cuTensorMapEncodeTiled(
    CUtensorMap* tensorMap, CUtensorMapDataType tensorDataType, cuuint32_t tensorRank,
    void* globalAddress, const cuuint64_t* globalDim, const cuuint64_t* globalStrides,
    const cuuint32_t* boxDim, const cuuint32_t* elementStrides, CUtensorMapInterleave interleave,
    CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion,
    CUtensorMapFloatOOBfill oobFill) {
  if (tensorRank < 1 || tensorRank > 5) return CUDA_ERROR_INVALID_VALUE;
  EncodeCall call{};
  call.data_type = tensorDataType;
  call.rank = static_cast<int>(tensorRank);
  call.interleave = interleave;
  call.swizzle = swizzle;
  call.l2_promotion = l2Promotion;
  call.oob_fill = oobFill;
  call.address = reinterpret_cast<uint64_t>(globalAddress);
  for (cuuint32_t i = 0; i < tensorRank; ++i) {
    call.dims.push_back(globalDim[i]);
    call.box.push_back(boxDim[i]);
    call.element_strides.push_back(elementStrides[i]);
    if (i + 1 < tensorRank) call.strides.push_back(globalStrides[i]);
  }
  fake_tensor_map(call, tensorMap);
  g_calls.push_back(call);
  return CUDA_SUCCESS;
}

// Interposes the shared cudart's cudaDriverGetVersion (see the header comment).
extern "C" cudaError_t CUDARTAPI cudaDriverGetVersion(int* driverVersion) {
  *driverVersion = g_driver_version;
  return cudaSuccess;
}

namespace {

using namespace cute;
using namespace nvfp4_dual_gemm;
using Kernel = GemmKernel;
using Params = Kernel::Params;
using Arguments = Kernel::Arguments;

constexpr int kClusterM = 1, kClusterN = 1, kClusterK = 1;  // CGA_M_, CGA_N_, CGA_K_
constexpr int kScale = flashinfer::gemm::SMTypeAdapter<flashinfer::gemm::_1SM>::Scale;

// Mirrors flashinfer::gemm::prepareGemmArgs_float_128_128_256_1_1_1_1SM line by line.
Arguments make_arguments(void* D, void const* A, void const* B, void const* input_sf,
                         void const* weight_sf, float const* global_sf, int m, int n, int k,
                         int batch_count) {
  using Sm1xxBlkScaledConfig = typename Kernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  using ElementA = typename Kernel::ElementA;
  using ElementB = typename Kernel::ElementB;
  using ElementSFA = cutlass::float_ue4m3_t;
  using ElementSFB = cutlass::float_ue4m3_t;
  using ElementC = void;
  using ElementD = typename Kernel::ElementD;
  using ElementCompute = float;

  Arguments operator_args;
  operator_args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  auto& fusion_args = operator_args.epilogue.thread;
  fusion_args.alpha_ptr = static_cast<ElementCompute const*>(global_sf);

  operator_args.problem_shape = cute::make_shape(m, n, k, batch_count);

  operator_args.mainloop.ptr_A = static_cast<ElementA const*>(A);
  operator_args.mainloop.ptr_B = static_cast<ElementB const*>(B);
  operator_args.mainloop.ptr_SFA = static_cast<ElementSFA const*>(input_sf);
  operator_args.mainloop.ptr_SFB = static_cast<ElementSFB const*>(weight_sf);
  operator_args.epilogue.ptr_C = static_cast<ElementC const*>(D);
  operator_args.epilogue.ptr_D = static_cast<ElementD*>(D);

  int const stride_A = batch_count == 1 ? 0 : m * k;
  int const stride_B = batch_count == 1 ? 0 : n * k;
  int const stride_C = batch_count == 1 ? 0 : m * n;

  operator_args.mainloop.dA = cute::make_int_tuple_from<typename Kernel::StrideA>(k, stride_A);
  operator_args.mainloop.dB = cute::make_int_tuple_from<typename Kernel::StrideB>(k, stride_B);
  operator_args.epilogue.dC = cute::make_int_tuple_from<typename Kernel::StrideC>(n, stride_C);
  operator_args.epilogue.dD = operator_args.epilogue.dC;

  operator_args.mainloop.layout_SFA =
      Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(operator_args.problem_shape);
  operator_args.mainloop.layout_SFB =
      Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(operator_args.problem_shape);

  if constexpr (!std::is_const_v<decltype(operator_args.scheduler.max_swizzle_size)>) {
    operator_args.scheduler.max_swizzle_size = 1;
  }
  if constexpr (!std::is_const_v<decltype(operator_args.scheduler.raster_order)>) {
    using Enum_t = decltype(operator_args.scheduler.raster_order);
    operator_args.scheduler.raster_order = Enum_t::Heuristic;
  }
  operator_args.hw_info.cluster_shape = dim3(kClusterM, kClusterN, kClusterK);
  operator_args.hw_info.cluster_shape_fallback = dim3(kScale, 1, 1);

  return operator_args;
}

struct Field {
  std::string name;
  size_t offset, size;
};

// A member of empty type (e.g. a fully static cute stride) holds no data; its
// one byte of storage is padding, so it is reported with size 0.
template <class Base, class T>
Field data_field(const char* name, const Base& base, const T& member) {
  return {name,
          static_cast<size_t>(reinterpret_cast<const char*>(&member) -
                              reinterpret_cast<const char*>(&base)),
          std::is_empty_v<T> ? 0 : sizeof(T)};
}

template <class Base, class T>
Field field(const char* name, const Base& base, const T& member) {
  return {name,
          static_cast<size_t>(reinterpret_cast<const char*>(&member) -
                              reinterpret_cast<const char*>(&base)),
          sizeof(T)};
}

// Offsets of the sentinel `value` within the fusion (epilogue.thread) params.
template <class ThreadParams, class T>
std::vector<size_t> find_sentinel(const ThreadParams& params, T value) {
  std::vector<size_t> offsets;
  auto* bytes = reinterpret_cast<const unsigned char*>(&params);
  for (size_t i = 0; i + sizeof(T) <= sizeof(ThreadParams); i += alignof(T)) {
    if (std::memcmp(bytes + i, &value, sizeof(T)) == 0) offsets.push_back(i);
  }
  return offsets;
}


#define TMA_FIELDS(prefix, atom)                                                          \
  field(prefix, p, atom), field(prefix ".desc", p, *atom.get_tma_descriptor()),           \
      data_field(prefix ".aux_g_stride", p, atom.aux_params_.g_stride_)

int layout_mode() {
  Params p{};
  auto& ml = p.mainloop;
  auto& ep = p.epilogue;
  auto& sc = p.scheduler;
  auto& hw = p.hw_info;
  std::vector<Field> fields = {
      field("mode", p, p.mode),
      field("problem_shape", p, p.problem_shape),
      field("problem_m", p, get<0>(p.problem_shape)),
      field("problem_n", p, get<1>(p.problem_shape)),
      field("problem_k", p, get<2>(p.problem_shape)),
      field("problem_l", p, get<3>(p.problem_shape)),
      field("mainloop", p, p.mainloop),
      TMA_FIELDS("mainloop.tma_load_a", ml.tma_load_a),
      TMA_FIELDS("mainloop.tma_load_b", ml.tma_load_b),
      TMA_FIELDS("mainloop.tma_load_sfa", ml.tma_load_sfa),
      TMA_FIELDS("mainloop.tma_load_sfb", ml.tma_load_sfb),
      TMA_FIELDS("mainloop.tma_load_a_fallback", ml.tma_load_a_fallback),
      TMA_FIELDS("mainloop.tma_load_b_fallback", ml.tma_load_b_fallback),
      TMA_FIELDS("mainloop.tma_load_sfa_fallback", ml.tma_load_sfa_fallback),
      TMA_FIELDS("mainloop.tma_load_sfb_fallback", ml.tma_load_sfb_fallback),
      // Layout<((( _32,_4),M_tiles),((_16,_4),K_tiles),(_1,L)) : (((_16,_4),M_tile_stride),((_0,_1),_512),(_0,L_stride))>
      field("mainloop.layout_SFA", p, ml.layout_SFA),
      field("mainloop.layout_SFA.shape_m_tiles", p, get<0, 1>(ml.layout_SFA.shape())),
      field("mainloop.layout_SFA.shape_k_tiles", p, get<1, 1>(ml.layout_SFA.shape())),
      field("mainloop.layout_SFA.shape_l", p, get<2, 1>(ml.layout_SFA.shape())),
      field("mainloop.layout_SFA.stride_m_tile", p, get<0, 1>(ml.layout_SFA.stride())),
      field("mainloop.layout_SFA.stride_l", p, get<2, 1>(ml.layout_SFA.stride())),
      field("mainloop.layout_SFB", p, ml.layout_SFB),
      field("mainloop.layout_SFB.shape_n_tiles", p, get<0, 1>(ml.layout_SFB.shape())),
      field("mainloop.layout_SFB.shape_k_tiles", p, get<1, 1>(ml.layout_SFB.shape())),
      field("mainloop.layout_SFB.shape_l", p, get<2, 1>(ml.layout_SFB.shape())),
      field("mainloop.layout_SFB.stride_n_tile", p, get<0, 1>(ml.layout_SFB.stride())),
      field("mainloop.layout_SFB.stride_l", p, get<2, 1>(ml.layout_SFB.stride())),
      field("mainloop.cluster_shape_fallback", p, ml.cluster_shape_fallback),
      field("mainloop.runtime_data_type_a", p, ml.runtime_data_type_a),
      field("mainloop.runtime_data_type_b", p, ml.runtime_data_type_b),
      field("epilogue", p, p.epilogue),
      field("epilogue.thread", p, ep.thread),
      TMA_FIELDS("epilogue.tma_load_c", ep.tma_load_c),
      TMA_FIELDS("epilogue.tma_store_d", ep.tma_store_d),
      field("scheduler", p, p.scheduler),
      field("scheduler.problem_tiles_m", p, sc.problem_tiles_m_),
      field("scheduler.problem_tiles_n", p, sc.problem_tiles_n_),
      field("scheduler.problem_tiles_l", p, sc.problem_tiles_l_),
      field("scheduler.divmod_cluster_shape_m", p, sc.divmod_cluster_shape_m_),
      field("scheduler.divmod_cluster_shape_n", p, sc.divmod_cluster_shape_n_),
      field("scheduler.divmod_swizzle_size", p, sc.divmod_swizzle_size_),
      field("scheduler.raster_order", p, sc.raster_order_),
      field("scheduler.log_swizzle_size", p, sc.log_swizzle_size_),
      field("hw_info", p, p.hw_info),
      field("hw_info.device_id", p, hw.device_id),
      field("hw_info.sm_count", p, hw.sm_count),
      field("hw_info.max_active_clusters", p, hw.max_active_clusters),
      field("hw_info.cluster_shape", p, hw.cluster_shape),
      field("hw_info.cluster_shape_fallback", p, hw.cluster_shape_fallback),
  };
  static_assert(sizeof(ml.layout_SFA) == 5 * sizeof(int), "unexpected dynamic SFA layout");
  static_assert(sizeof(ml.layout_SFB) == 5 * sizeof(int), "unexpected dynamic SFB layout");

  // The fusion-callback tree's data members (alpha, beta, their pointers and
  // batch strides) are located by unique sentinels passed through the
  // epilogue's own FusionCallbacks::to_underlying_arguments. Every other byte
  // of epilogue.thread is padding or empty (static) members.
  using Fusion = typename Kernel::CollectiveEpilogue::FusionCallbacks;
  typename Fusion::Arguments thread{};
  thread.alpha = 1.25f;
  thread.beta = -3.5f;
  thread.alpha_ptr = reinterpret_cast<const float*>(0x1111111111111110ull);
  thread.beta_ptr = reinterpret_cast<const float*>(0x2222222222222220ull);
  get<2>(thread.dAlpha) = int64_t(0x3333333333333333ll);
  get<2>(thread.dBeta) = int64_t(0x4444444444444444ll);
  auto fusion = Fusion::to_underlying_arguments(make_shape(1, 1, 1, 1), thread, nullptr);
  size_t thread_offset = field("epilogue.thread", p, ep.thread).offset;
  auto locate = [&](const char* name, auto value) {
    auto offsets = find_sentinel(fusion, value);
    if (offsets.size() != 1) {
      std::fprintf(stderr, "sentinel for %s not unique in fusion params\n", name);
      std::exit(1);
    }
    fields.push_back({name, thread_offset + offsets[0], sizeof(value)});
  };
  locate("epilogue.thread.alpha", thread.alpha);
  locate("epilogue.thread.beta", thread.beta);
  locate("epilogue.thread.alpha_ptr", thread.alpha_ptr);
  locate("epilogue.thread.beta_ptr", thread.beta_ptr);
  locate("epilogue.thread.alpha_stride_l", get<2>(thread.dAlpha));
  locate("epilogue.thread.beta_stride_l", get<2>(thread.dBeta));

  // Defaults of the FusionCallbacks Arguments FlashInfer leaves untouched.
  typename Fusion::Arguments defaults{};

  std::printf("{\n");
  std::printf("  \"kernel_symbol_check\": \"cutlass::device_kernel<nvfp4_dual_gemm::GemmKernel>\",\n");
  std::printf("  \"params_size\": %zu,\n  \"params_align\": %zu,\n", sizeof(Params),
              alignof(Params));
  std::printf("  \"fields\": {\n");
  for (size_t i = 0; i < fields.size(); ++i) {
    std::printf("    \"%s\": {\"offset\": %zu, \"size\": %zu}%s\n", fields[i].name.c_str(),
                fields[i].offset, fields[i].size, i + 1 < fields.size() ? "," : "");
  }
  std::printf("  },\n");
  std::printf("  \"constants\": {\n");
  std::printf("    \"shared_storage_size\": %d,\n", Kernel::SharedStorageSize);
  std::printf("    \"max_threads_per_block\": %u,\n", Kernel::MaxThreadsPerBlock);
  std::printf("    \"min_blocks_per_multiprocessor\": %u,\n", Kernel::MinBlocksPerMultiprocessor);
  std::printf("    \"tile_shape_mnk\": [%d, %d, %d],\n", int(size<0>(Kernel::TileShape{})),
              int(size<1>(Kernel::TileShape{})), int(size<2>(Kernel::TileShape{})));
  std::printf("    \"cta_shape_mnk\": [%d, %d, %d],\n", int(size<0>(Kernel::CtaShape_MNK{})),
              int(size<1>(Kernel::CtaShape_MNK{})), int(size<2>(Kernel::CtaShape_MNK{})));
  std::printf("    \"atom_thr_shape_mnk\": [%d, %d, %d],\n",
              int(size<0>(Kernel::AtomThrShapeMNK{})), int(size<1>(Kernel::AtomThrShapeMNK{})),
              int(size<2>(Kernel::AtomThrShapeMNK{})));
  std::printf("    \"is_dynamic_cluster\": %s,\n", Kernel::IsDynamicCluster ? "true" : "false");
  std::printf("    \"cluster_shape\": [%d, %d, %d],\n", kClusterM, kClusterN, kClusterK);
  std::printf("    \"cluster_shape_fallback\": [%d, 1, 1],\n", kScale);
  std::printf("    \"mainloop_stages\": %d,\n", Kernel::DispatchPolicy::Stages);
  std::printf("    \"scheduler_pipeline_stages\": %u,\n", Kernel::SchedulerPipelineStageCount);
  std::printf("    \"accumulator_pipeline_stages\": %u,\n", Kernel::AccumulatorPipelineStageCount);
  std::printf("    \"sf_vec_size\": %d,\n", int(Kernel::CollectiveMainloop::SFVecSize));
  std::printf("    \"default_alpha\": %.9g,\n", double(defaults.alpha));
  std::printf("    \"default_beta\": %.9g,\n", double(defaults.beta));
  std::printf("    \"gemm_mode_kGemm\": %d,\n", int(cutlass::gemm::GemmUniversalMode::kGemm));
  using RasterOrder = typename Kernel::TileScheduler::RasterOrder;
  std::printf("    \"raster_order_AlongM\": %d,\n", int(RasterOrder::AlongM));
  std::printf("    \"raster_order_AlongN\": %d,\n", int(RasterOrder::AlongN));
  std::printf("    \"sizeof_TileScheduler_Params\": %zu,\n", sizeof(Kernel::TileSchedulerParams));
  std::printf("    \"sizeof_FastDivmod\": %zu\n", sizeof(cutlass::FastDivmod));
  std::printf("  }\n}\n");
  return 0;
}

int params_mode(int argc, char** argv) {
  if (argc != 13) {
    std::fprintf(stderr, "usage: params M N K L A B SFA SFB D ALPHA DRIVER_VERSION\n");
    return 1;
  }
  int m = std::atoi(argv[2]), n = std::atoi(argv[3]), k = std::atoi(argv[4]);
  int l = std::atoi(argv[5]);
  auto ptr = [&](int i) { return std::strtoull(argv[i], nullptr, 0); };
  uint64_t a = ptr(6), b = ptr(7), sfa = ptr(8), sfb = ptr(9), d = ptr(10), alpha = ptr(11);
  g_driver_version = std::atoi(argv[12]);
  void* workspace = nullptr;  // FlashInfer passes its 32 MiB buffer; CUTLASS needs none.
  auto as_ptr = [](uint64_t value) { return reinterpret_cast<void*>(value); };
  Arguments args = make_arguments(as_ptr(d), as_ptr(a), as_ptr(b), as_ptr(sfa), as_ptr(sfb),
                                  reinterpret_cast<const float*>(alpha), m, n, k, l);
  bool can_implement = Kernel::can_implement(args);
  size_t workspace_size = Kernel::get_workspace_size(args);
  // Padding bytes of the returned Params are indeterminate (they come from
  // stack temporaries). Build Params twice, with the destination and the stack
  // poisoned by 0x00 and then 0xff; bytes that differ are padding the kernel
  // never reads and are reported as such. CUTLASS encodes the descriptors on
  // both runs; only the first run's encode calls are reported.
  alignas(Params) static unsigned char storage[2][sizeof(Params)];
  for (int run = 0; run < 2; ++run) {
    unsigned char fill = run ? 0xff : 0x00;
    std::memset(storage[run], fill, sizeof(Params));
    poison_stack(fill);
    new (storage[run]) Params(Kernel::to_underlying_arguments(args, workspace));
    if (run == 0) g_calls_first = g_calls.size();
  }
  g_calls.resize(g_calls_first);
  std::vector<size_t> indeterminate;
  for (size_t i = 0; i < sizeof(Params); ++i) {
    if (storage[0][i] != storage[1][i]) indeterminate.push_back(i);
  }
  auto* params = reinterpret_cast<Params*>(storage[0]);
  dim3 grid = Kernel::get_grid_shape(*params);
  dim3 block = Kernel::get_block_shape();

  std::printf("{\n");
  std::printf("  \"problem\": {\"m\": %d, \"n\": %d, \"k\": %d, \"l\": %d},\n", m, n, k, l);
  std::printf(
      "  \"pointers\": {\"a\": %llu, \"b\": %llu, \"sfa\": %llu, \"sfb\": %llu, \"d\": %llu, "
      "\"alpha\": %llu},\n",
      (unsigned long long)a, (unsigned long long)b, (unsigned long long)sfa,
      (unsigned long long)sfb, (unsigned long long)d, (unsigned long long)alpha);
  std::printf("  \"driver_version\": %d,\n", g_driver_version);
  std::printf("  \"can_implement\": %s,\n", can_implement ? "true" : "false");
  std::printf("  \"workspace_size\": %zu,\n", workspace_size);
  std::printf("  \"grid\": [%u, %u, %u],\n  \"block\": [%u, %u, %u],\n", grid.x, grid.y, grid.z,
              block.x, block.y, block.z);
  std::printf("  \"shared_mem\": %d,\n", Kernel::SharedStorageSize);
  std::printf("  \"tensor_maps\": [\n");
  for (size_t i = 0; i < g_calls.size(); ++i) {
    auto& c = g_calls[i];
    std::printf(
        "    {\"data_type\": %d, \"rank\": %d, \"address\": %llu, \"dims\": %s, "
        "\"strides\": %s, \"box\": %s, \"element_strides\": %s, \"interleave\": %d, "
        "\"swizzle\": %d, \"l2_promotion\": %d, \"oob_fill\": %d}%s\n",
        c.data_type, c.rank, (unsigned long long)c.address, list(c.dims).c_str(),
        list(c.strides).c_str(), list(c.box).c_str(), list(c.element_strides).c_str(),
        c.interleave, c.swizzle, c.l2_promotion, c.oob_fill,
        i + 1 < g_calls.size() ? "," : "");
  }
  std::printf("  ],\n");
  std::printf("  \"indeterminate_bytes\": %s,\n", list(indeterminate).c_str());
  std::printf("  \"params_hex\": \"%s\"\n}\n", hex(storage[0], sizeof(Params)).c_str());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::string mode = argc > 1 ? argv[1] : "";
  if (mode == "layout") return layout_mode();
  if (mode == "params") return params_mode(argc, argv);
  std::fprintf(stderr, "usage: %s layout | params M N K L A B SFA SFB D ALPHA DRIVER_VERSION\n", argv[0]);
  return 1;
}
