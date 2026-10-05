// SPDX-License-Identifier: Apache-2.0
// Compile-time, host-only probe for the nvfp4_group_gemm kernel's by-value
// Params and its per-group device argument arrays.
//
// Built and run by impls/nvfp4_group_gemm/compiler.py; it is never part of a
// benchmark run, launches nothing and needs neither a GPU nor libcuda. Two
// modes, both printing one JSON document to stdout:
//
//   nvfp4_group_gemm_probe layout
//       sizeof/offsetof of every Params field the Python builder writes,
//       sizes/offsets of the per-group array elements, the kernel's
//       compile-time launch constants, workspace sizing and enum values.
//
//   nvfp4_group_gemm_probe params SM_COUNT DRIVER_VERSION BASE M,N,K [M,N,K ...]
//       Builds the grouped Arguments exactly as CUTLASS's example 75
//       (75_blackwell_grouped_gemm_block_scaled.cu) does for this kernel --
//       host-side problem shapes, per-group packed strides and scale-factor
//       layouts, per-group pointer arrays, alpha = 1, beta = 0, no C,
//       hw_info.sm_count = SM_COUNT -- with fake (aligned, never
//       dereferenced) device addresses derived from BASE, runs CUTLASS's own
//       GemmKernel::can_implement / get_workspace_size /
//       to_underlying_arguments / get_grid_shape and prints the Params
//       bytes, the launch configuration, the workspace size, the bytes of
//       every per-group device array CUTLASS's host code fills and every
//       cuTensorMapEncodeTiled call CUTLASS made.
//
// cuTensorMapEncodeTiled is interposed: the real driver refuses to encode on
// pre-Hopper devices (CUDA_ERROR_NOT_SUPPORTED on sm_86) and the descriptor
// format is opaque, so the probe records the exact encode arguments and fills
// the 128-byte descriptor with a deterministic packing of those arguments
// (fake_tensor_map below; harness/workloads/nvfp4_group_gemm.py has the
// identical Python packing for tests). The packing sets bit 21 of descriptor
// word 1 so that CUTLASS's driver-version-dependent fix-up (which clears it
// for small tensors) is visible even for the null addresses grouped kernels
// encode. cudaDriverGetVersion is interposed too (the probe links the shared
// cudart) so that the fix-up is exercised deterministically for the
// DRIVER_VERSION given on the command line.
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

#include "nvfp4_group_gemm_sm100.cuh"

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
//   u64 address | (1 << 21), u64 dims[5], u64 strides[4], u32 box[5],
//   u32 element_strides[5]
// (unused trailing entries are zero), exactly 128 bytes.
void fake_tensor_map(const EncodeCall& c, CUtensorMap* out) {
  unsigned char bytes[128] = {};
  bytes[0] = static_cast<unsigned char>(c.data_type);
  bytes[1] = static_cast<unsigned char>(c.rank);
  bytes[2] = static_cast<unsigned char>(c.interleave);
  bytes[3] = static_cast<unsigned char>(c.swizzle);
  bytes[4] = static_cast<unsigned char>(c.l2_promotion);
  bytes[5] = static_cast<unsigned char>(c.oob_fill);
  uint64_t word = c.address | (uint64_t(1) << 21);
  std::memcpy(bytes + 8, &word, 8);
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

using namespace nvfp4_group_gemm;
using Kernel = GemmKernel;
using Params = Kernel::Params;
using Arguments = Kernel::Arguments;
using Shape3 = ProblemShape::UnderlyingProblemShape;
using StrideA = Kernel::InternalStrideA;
using StrideB = Kernel::InternalStrideB;
using StrideC = Kernel::InternalStrideC;
using StrideD = Kernel::InternalStrideD;
using LayoutSFA = Kernel::CollectiveMainloop::InternalLayoutSFA;
using LayoutSFB = Kernel::CollectiveMainloop::InternalLayoutSFB;
using BlkScaledConfig = Kernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
using ElementSF = Kernel::ElementSF;
using MainloopElementA = Kernel::CollectiveMainloop::ArrayElementA;
using MainloopElementB = Kernel::CollectiveMainloop::ArrayElementB;
using ElementD = Kernel::ElementD;
using Fusion = Kernel::CollectiveEpilogue::FusionCallbacks;

static_assert(Kernel::IsGroupedGemmKernel, "expected a grouped GEMM kernel");
static_assert(std::is_same_v<ElementC, void>, "expected a sourceless epilogue");

struct Field {
  std::string name;
  size_t offset, size;
};

template <class Base, class T>
Field field(const char* name, const Base& base, const T& member) {
  return {name,
          static_cast<size_t>(reinterpret_cast<const char*>(&member) -
                              reinterpret_cast<const char*>(&base)),
          sizeof(T)};
}

// A member of empty type (e.g. a fully static cute stride) holds no data; its
// storage is padding (or outside the object), so it is reported as {0, 0}.
template <class Base, class T>
Field data_field(const char* name, const Base& base, const T& member) {
  if (std::is_empty_v<T>) return {name, 0, 0};
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

void print_fields(const char* key, const std::vector<Field>& fields, bool last) {
  std::printf("  \"%s\": {\n", key);
  for (size_t i = 0; i < fields.size(); ++i) {
    std::printf("    \"%s\": {\"offset\": %zu, \"size\": %zu}%s\n", fields[i].name.c_str(),
                fields[i].offset, fields[i].size, i + 1 < fields.size() ? "," : "");
  }
  std::printf("  }%s\n", last ? "" : ",");
}

int layout_mode() {
  Params p{};
  auto& ps = p.problem_shape;
  auto& ml = p.mainloop;
  auto& ep = p.epilogue;
  auto& sc = p.scheduler.params_sm90_;
  auto& hw = p.hw_info;
#define TMA_FIELDS(prefix, atom)                                              \
  field(prefix, p, atom), field(prefix ".desc", p, *atom.get_tma_descriptor()), \
      data_field(prefix ".aux_g_stride", p, atom.aux_params_.g_stride_)
  std::vector<Field> fields = {
      field("mode", p, p.mode),
      field("problem_shape", p, ps),
      field("problem_shape.num_groups", p, ps.num_groups),
      field("problem_shape.problem_shapes", p, ps.problem_shapes),
      field("problem_shape.host_problem_shapes", p, ps.host_problem_shapes),
      field("mainloop", p, ml),
      TMA_FIELDS("mainloop.tma_load_a", ml.tma_load_a),
      TMA_FIELDS("mainloop.tma_load_b", ml.tma_load_b),
      TMA_FIELDS("mainloop.tma_load_sfa", ml.tma_load_sfa),
      TMA_FIELDS("mainloop.tma_load_sfb", ml.tma_load_sfb),
      TMA_FIELDS("mainloop.tma_load_a_fallback", ml.tma_load_a_fallback),
      TMA_FIELDS("mainloop.tma_load_b_fallback", ml.tma_load_b_fallback),
      TMA_FIELDS("mainloop.tma_load_sfa_fallback", ml.tma_load_sfa_fallback),
      TMA_FIELDS("mainloop.tma_load_sfb_fallback", ml.tma_load_sfb_fallback),
      field("mainloop.cluster_shape_fallback", p, ml.cluster_shape_fallback),
      data_field("mainloop.runtime_data_type_a", p, ml.runtime_data_type_a),
      data_field("mainloop.runtime_data_type_b", p, ml.runtime_data_type_b),
      field("mainloop.tensormaps", p, ml.tensormaps),
      field("mainloop.ptr_A", p, ml.ptr_A),
      field("mainloop.dA", p, ml.dA),
      field("mainloop.ptr_B", p, ml.ptr_B),
      field("mainloop.dB", p, ml.dB),
      field("mainloop.ptr_SFA", p, ml.ptr_SFA),
      field("mainloop.layout_SFA", p, ml.layout_SFA),
      field("mainloop.ptr_SFB", p, ml.ptr_SFB),
      field("mainloop.layout_SFB", p, ml.layout_SFB),
      field("epilogue", p, ep),
      field("epilogue.thread", p, ep.thread),
      TMA_FIELDS("epilogue.tma_load_c", ep.tma_load_c),
      TMA_FIELDS("epilogue.tma_store_d", ep.tma_store_d),
      field("epilogue.tensormaps", p, ep.tensormaps),
      field("epilogue.ptr_C", p, ep.ptr_C),
      field("epilogue.dC", p, ep.dC),
      field("epilogue.ptr_D", p, ep.ptr_D),
      field("epilogue.dD", p, ep.dD),
      field("scheduler", p, p.scheduler),
      field("scheduler.divmod_cluster_shape_major.divisor", p, sc.divmod_cluster_shape_major_.divisor),
      field("scheduler.divmod_cluster_shape_major.shift_right", p, sc.divmod_cluster_shape_major_.shift_right),
      field("scheduler.divmod_cluster_shape_minor.divisor", p, sc.divmod_cluster_shape_minor_.divisor),
      field("scheduler.divmod_cluster_shape_minor.shift_right", p, sc.divmod_cluster_shape_minor_.shift_right),
      field("scheduler.divmod_cta_shape_m.divisor", p, sc.divmod_cta_shape_m_.divisor),
      field("scheduler.divmod_cta_shape_m.multiplier", p, sc.divmod_cta_shape_m_.multiplier),
      field("scheduler.divmod_cta_shape_m.shift_right", p, sc.divmod_cta_shape_m_.shift_right),
      field("scheduler.divmod_cta_shape_m.round_up", p, sc.divmod_cta_shape_m_.round_up),
      field("scheduler.divmod_cta_shape_n.divisor", p, sc.divmod_cta_shape_n_.divisor),
      field("scheduler.divmod_cta_shape_n.multiplier", p, sc.divmod_cta_shape_n_.multiplier),
      field("scheduler.divmod_cta_shape_n.shift_right", p, sc.divmod_cta_shape_n_.shift_right),
      field("scheduler.divmod_cta_shape_n.round_up", p, sc.divmod_cta_shape_n_.round_up),
      field("scheduler.blocks_across_problem", p, sc.blocks_across_problem_),
      field("scheduler.pre_processed_problem_shapes", p, sc.pre_processed_problem_shapes),
      field("scheduler.max_swizzle_size", p, sc.max_swizzle_size_),
      field("scheduler.raster_order", p, sc.raster_order_),
      field("scheduler.problem_shapes.num_groups", p, sc.problem_shapes_.num_groups),
      field("scheduler.problem_shapes.problem_shapes", p, sc.problem_shapes_.problem_shapes),
      field("scheduler.problem_shapes.host_problem_shapes", p, sc.problem_shapes_.host_problem_shapes),
      field("scheduler.cta_shape", p, sc.cta_shape_),
      field("scheduler.cluster_shape", p, sc.cluster_shape_),
      field("hw_info", p, hw),
      field("hw_info.device_id", p, hw.device_id),
      field("hw_info.sm_count", p, hw.sm_count),
      field("hw_info.max_active_clusters", p, hw.max_active_clusters),
      field("hw_info.cluster_shape", p, hw.cluster_shape),
      field("hw_info.cluster_shape_fallback", p, hw.cluster_shape_fallback),
  };
#undef TMA_FIELDS

  // The fusion-callback tree's data members (alpha, beta, their pointers,
  // pointer arrays and batch strides) are located by unique sentinels passed
  // through the epilogue's own FusionCallbacks::to_underlying_arguments.
  // Every other byte of epilogue.thread is padding or empty (static) members.
  typename Fusion::Arguments thread{};
  thread.alpha = 1.25f;
  thread.beta = -3.5f;
  thread.alpha_ptr = reinterpret_cast<const float*>(0x1111111111111110ull);
  thread.beta_ptr = reinterpret_cast<const float*>(0x2222222222222220ull);
  thread.alpha_ptr_array = reinterpret_cast<const float* const*>(0x5555555555555550ull);
  thread.beta_ptr_array = reinterpret_cast<const float* const*>(0x6666666666666660ull);
  get<2>(thread.dAlpha) = int64_t(0x3333333333333333ll);
  get<2>(thread.dBeta) = int64_t(0x4444444444444444ll);
  ProblemShape one_group{};
  auto fusion = Fusion::to_underlying_arguments(one_group, thread, nullptr);
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
  locate("epilogue.thread.alpha_ptr_array", thread.alpha_ptr_array);
  locate("epilogue.thread.beta_ptr_array", thread.beta_ptr_array);
  locate("epilogue.thread.alpha_stride_l", get<2>(thread.dAlpha));
  locate("epilogue.thread.beta_stride_l", get<2>(thread.dBeta));

  // Per-group device array elements.
  Shape3 shape{};
  StrideA stride_a{};
  StrideB stride_b{};
  StrideD stride_d{};
  LayoutSFA layout_sfa{};
  std::vector<Field> elements = {
      field("problem_shape", shape, shape),
      field("problem_shape.m", shape, get<0>(shape)),
      field("problem_shape.n", shape, get<1>(shape)),
      field("problem_shape.k", shape, get<2>(shape)),
      field("stride_a", stride_a, stride_a),
      data_field("stride_a.0", stride_a, get<0>(stride_a)),
      data_field("stride_a.1", stride_a, get<1>(stride_a)),
      data_field("stride_a.2", stride_a, get<2>(stride_a)),
      field("stride_b", stride_b, stride_b),
      data_field("stride_b.0", stride_b, get<0>(stride_b)),
      data_field("stride_b.1", stride_b, get<1>(stride_b)),
      data_field("stride_b.2", stride_b, get<2>(stride_b)),
      field("stride_d", stride_d, stride_d),
      data_field("stride_d.0", stride_d, get<0>(stride_d)),
      data_field("stride_d.1", stride_d, get<1>(stride_d)),
      data_field("stride_d.2", stride_d, get<2>(stride_d)),
      field("layout_sf", layout_sfa, layout_sfa),
      // (((32, 4), MN blocks), ((16, 4), K blocks), (1, L)) :
      // (((16, 4), s_mn), ((0, 1), s_k), (s_l0, s_l)); static members are empty.
      data_field("layout_sf.shape_mn_blocks", layout_sfa, get<0, 1>(layout_sfa.shape())),
      data_field("layout_sf.shape_k_blocks", layout_sfa, get<1, 1>(layout_sfa.shape())),
      data_field("layout_sf.shape_l0", layout_sfa, get<2, 0>(layout_sfa.shape())),
      data_field("layout_sf.shape_l", layout_sfa, get<2, 1>(layout_sfa.shape())),
      data_field("layout_sf.stride_mn_blocks", layout_sfa, get<0, 1>(layout_sfa.stride())),
      data_field("layout_sf.stride_k_blocks", layout_sfa, get<1, 1>(layout_sfa.stride())),
      data_field("layout_sf.stride_l0", layout_sfa, get<2, 0>(layout_sfa.stride())),
      data_field("layout_sf.stride_l", layout_sfa, get<2, 1>(layout_sfa.stride())),
  };
  static_assert(std::is_same_v<LayoutSFA, LayoutSFB>, "SFA and SFB share one layout type");

  // Workspace: CUTLASS sizes the tensormap buffers per SM.
  Arguments args{};
  args.problem_shape = one_group;
  size_t fusion_ws = Fusion::get_workspace_size(one_group, args.epilogue.thread);
  size_t epi_ws0 = Kernel::CollectiveEpilogue::get_workspace_size(one_group, args.epilogue, 0);
  size_t epi_ws1 = Kernel::CollectiveEpilogue::get_workspace_size(one_group, args.epilogue, 1);
  size_t ml_ws0 = Kernel::CollectiveMainloop::get_workspace_size(one_group, args.mainloop, 0);
  size_t ml_ws1 = Kernel::CollectiveMainloop::get_workspace_size(one_group, args.mainloop, 1);

  std::printf("{\n");
  std::printf("  \"kernel_symbol_check\": \"cutlass::device_kernel<nvfp4_group_gemm::GemmKernel>\",\n");
  std::printf("  \"params_size\": %zu,\n  \"params_align\": %zu,\n", sizeof(Params),
              alignof(Params));
  print_fields("fields", fields, false);
  print_fields("elements", elements, false);
  std::printf("  \"element_align\": {\"problem_shape\": %zu, \"stride\": %zu, \"layout_sf\": %zu},\n",
              alignof(Shape3), alignof(StrideA), alignof(LayoutSFA));
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
  std::printf("    \"cluster_shape_mnk\": [%d, %d, %d],\n", int(size<0>(Kernel::ClusterShape{})),
              int(size<1>(Kernel::ClusterShape{})), int(size<2>(Kernel::ClusterShape{})));
  std::printf("    \"is_dynamic_cluster\": %s,\n", Kernel::IsDynamicCluster ? "true" : "false");
  std::printf("    \"is_sched_dynamic_persistent\": %s,\n",
              Kernel::IsSchedDynamicPersistent ? "true" : "false");
  std::printf("    \"mainloop_stages\": %d,\n", Kernel::DispatchPolicy::Stages);
  std::printf("    \"scheduler_pipeline_stages\": %u,\n", Kernel::SchedulerPipelineStageCount);
  std::printf("    \"accumulator_pipeline_stages\": %u,\n", Kernel::AccumulatorPipelineStageCount);
  std::printf("    \"sf_vec_size\": %d,\n", Kernel::CollectiveMainloop::SFVecSize);
  std::printf("    \"gemm_mode_kGrouped\": %d,\n", int(cutlass::gemm::GemmUniversalMode::kGrouped));
  using RasterOrder = typename Kernel::TileScheduler::RasterOrder;
  std::printf("    \"raster_order_AlongM\": %d,\n", int(RasterOrder::AlongM));
  std::printf("    \"raster_order_AlongN\": %d,\n", int(RasterOrder::AlongN));
  std::printf("    \"min_tensormap_workspace_alignment\": %u,\n",
              Kernel::MinTensorMapWorkspaceAlignment);
  std::printf("    \"fusion_workspace_size\": %zu,\n", fusion_ws);
  std::printf("    \"epilogue_workspace_base\": %zu,\n", epi_ws0);
  std::printf("    \"epilogue_workspace_per_sm\": %zu,\n", epi_ws1 - epi_ws0);
  std::printf("    \"mainloop_workspace_base\": %zu,\n", ml_ws0);
  std::printf("    \"mainloop_workspace_per_sm\": %zu,\n", ml_ws1 - ml_ws0);
  std::printf("    \"sizeof_TileScheduler_Params\": %zu\n", sizeof(Kernel::TileSchedulerParams));
  std::printf("  }\n}\n");
  return 0;
}

// Fake device addresses derived from BASE (the Python test reproduces them
// from the printed "pointers"): per-group operands far apart, argument arrays
// and the workspace each 64 KiB-aligned.
struct FakeAddresses {
  uint64_t base;
  uint64_t operand(int group, int slot) const {
    return base + (uint64_t(group + 1) << 32) + (uint64_t(slot) << 28);
  }
  uint64_t array(int slot) const { return base + (uint64_t(slot) << 20); }
};

enum ArraySlot {
  kWorkspace = 0,
  kProblemShapes,
  kPtrA,
  kPtrB,
  kPtrSFA,
  kPtrSFB,
  kPtrD,
  kStrideA,
  kStrideB,
  kStrideD,
  kLayoutSFA,
  kLayoutSFB,
  kNumArrays
};
const char* kArrayNames[kNumArrays] = {
    "workspace", "problem_shapes", "ptr_a",    "ptr_b",    "ptr_sfa",    "ptr_sfb",
    "ptr_d",     "stride_a",       "stride_b", "stride_d", "layout_sfa", "layout_sfb"};
enum OperandSlot { kA = 0, kB, kSFA, kSFB, kD, kNumOperands };
const char* kOperandNames[kNumOperands] = {"a", "b", "sfa", "sfb", "d"};

template <class T>
std::string vector_hex(const std::vector<T>& values) {
  return hex(values.data(), values.size() * sizeof(T));
}

int params_mode(int argc, char** argv) {
  if (argc < 6) {
    std::fprintf(stderr, "usage: params SM_COUNT DRIVER_VERSION BASE M,N,K [M,N,K ...]\n");
    return 1;
  }
  int sm_count = std::atoi(argv[2]);
  g_driver_version = std::atoi(argv[3]);
  FakeAddresses fake{std::strtoull(argv[4], nullptr, 0)};
  std::vector<Shape3> problems;
  for (int i = 5; i < argc; ++i) {
    int m, n, k;
    if (std::sscanf(argv[i], "%d,%d,%d", &m, &n, &k) != 3) return 1;
    problems.push_back({m, n, k});
  }
  int groups = static_cast<int>(problems.size());

  // Per-group host data, exactly as CUTLASS example 75 fills it.
  std::vector<StrideA> stride_a;
  std::vector<StrideB> stride_b;
  std::vector<StrideD> stride_d;
  std::vector<LayoutSFA> layout_sfa;
  std::vector<LayoutSFB> layout_sfb;
  std::vector<uint64_t> ptr[kNumOperands];
  for (int g = 0; g < groups; ++g) {
    auto [m, n, k] = problems[g];
    stride_a.push_back(cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1}));
    stride_b.push_back(cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1}));
    stride_d.push_back(cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1}));
    layout_sfa.push_back(BlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1)));
    layout_sfb.push_back(BlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1)));
    for (int s = 0; s < kNumOperands; ++s) ptr[s].push_back(fake.operand(g, s));
  }

  auto array = [&](int slot) { return fake.array(slot); };
  Arguments args;
  args.mode = cutlass::gemm::GemmUniversalMode::kGrouped;
  args.problem_shape = {groups, reinterpret_cast<Shape3*>(array(kProblemShapes)),
                        problems.data()};
  args.mainloop = {reinterpret_cast<const MainloopElementA**>(array(kPtrA)),
                   reinterpret_cast<StrideA*>(array(kStrideA)),
                   reinterpret_cast<const MainloopElementB**>(array(kPtrB)),
                   reinterpret_cast<StrideB*>(array(kStrideB)),
                   reinterpret_cast<const ElementSF**>(array(kPtrSFA)),
                   reinterpret_cast<LayoutSFA*>(array(kLayoutSFA)),
                   reinterpret_cast<const ElementSF**>(array(kPtrSFB)),
                   reinterpret_cast<LayoutSFB*>(array(kLayoutSFB))};
  args.epilogue = {{},
                   nullptr,
                   nullptr,
                   reinterpret_cast<ElementD**>(array(kPtrD)),
                   reinterpret_cast<StrideD*>(array(kStrideD))};
  auto& fusion_args = args.epilogue.thread;
  fusion_args.alpha = 1.0f;
  fusion_args.beta = 0.0f;
  fusion_args.alpha_ptr = nullptr;
  fusion_args.beta_ptr = nullptr;
  fusion_args.alpha_ptr_array = nullptr;
  fusion_args.beta_ptr_array = nullptr;
  fusion_args.dAlpha = {_0{}, _0{}, 0};
  fusion_args.dBeta = {_0{}, _0{}, 0};
  args.hw_info.device_id = 0;
  args.hw_info.sm_count = sm_count;
  // Default scheduler arguments (Heuristic raster order, no swizzle).

  void* workspace = reinterpret_cast<void*>(array(kWorkspace));
  bool can_implement = Kernel::can_implement(args);
  size_t workspace_size = Kernel::get_workspace_size(args);
  // Padding bytes of the returned Params are indeterminate (they come from
  // stack temporaries). Build Params twice, with the destination and the stack
  // poisoned by 0x00 and then 0xff; bytes that differ are padding the kernel
  // never reads and are reported as such. CUTLASS encodes the descriptors on
  // both runs; only the first run's encode calls are reported.
  alignas(Params) static unsigned char storage[2][sizeof(Params)];
  size_t first_calls = 0;
  for (int run = 0; run < 2; ++run) {
    unsigned char fill = run ? 0xff : 0x00;
    std::memset(storage[run], fill, sizeof(Params));
    poison_stack(fill);
    new (storage[run]) Params(Kernel::to_underlying_arguments(args, workspace));
    if (run == 0) first_calls = g_calls.size();
  }
  g_calls.resize(first_calls);
  std::vector<size_t> indeterminate;
  for (size_t i = 0; i < sizeof(Params); ++i) {
    if (storage[0][i] != storage[1][i]) indeterminate.push_back(i);
  }
  auto* params = reinterpret_cast<Params*>(storage[0]);
  dim3 grid = Kernel::get_grid_shape(*params);
  dim3 block = Kernel::get_block_shape();

  std::printf("{\n");
  std::printf("  \"problems\": [");
  for (int g = 0; g < groups; ++g) {
    std::printf("%s[%d, %d, %d]", g ? ", " : "", get<0>(problems[g]), get<1>(problems[g]),
                get<2>(problems[g]));
  }
  std::printf("],\n");
  std::printf("  \"sm_count\": %d,\n", sm_count);
  std::printf("  \"driver_version\": %d,\n", g_driver_version);
  std::printf("  \"base\": %llu,\n", (unsigned long long)fake.base);
  std::printf("  \"host_problem_shapes\": %llu,\n",
              (unsigned long long)reinterpret_cast<uint64_t>(problems.data()));
  std::printf("  \"pointers\": {\n");
  for (int s = 0; s < kNumArrays; ++s) {
    std::printf("    \"%s\": %llu,\n", kArrayNames[s], (unsigned long long)array(s));
  }
  for (int s = 0; s < kNumOperands; ++s) {
    std::printf("    \"%s\": %s%s\n", kOperandNames[s], list(ptr[s]).c_str(),
                s + 1 < kNumOperands ? "," : "");
  }
  std::printf("  },\n");
  std::printf("  \"arrays_hex\": {\n");
  std::printf("    \"problem_shapes\": \"%s\",\n", vector_hex(problems).c_str());
  std::printf("    \"ptr_a\": \"%s\",\n", vector_hex(ptr[kA]).c_str());
  std::printf("    \"ptr_b\": \"%s\",\n", vector_hex(ptr[kB]).c_str());
  std::printf("    \"ptr_sfa\": \"%s\",\n", vector_hex(ptr[kSFA]).c_str());
  std::printf("    \"ptr_sfb\": \"%s\",\n", vector_hex(ptr[kSFB]).c_str());
  std::printf("    \"ptr_d\": \"%s\",\n", vector_hex(ptr[kD]).c_str());
  std::printf("    \"stride_a\": \"%s\",\n", vector_hex(stride_a).c_str());
  std::printf("    \"stride_b\": \"%s\",\n", vector_hex(stride_b).c_str());
  std::printf("    \"stride_d\": \"%s\",\n", vector_hex(stride_d).c_str());
  std::printf("    \"layout_sfa\": \"%s\",\n", vector_hex(layout_sfa).c_str());
  std::printf("    \"layout_sfb\": \"%s\"\n", vector_hex(layout_sfb).c_str());
  std::printf("  },\n");
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
  std::fprintf(stderr,
               "usage: %s layout | params SM_COUNT DRIVER_VERSION BASE M,N,K [M,N,K ...]\n",
               argv[0]);
  return 1;
}
