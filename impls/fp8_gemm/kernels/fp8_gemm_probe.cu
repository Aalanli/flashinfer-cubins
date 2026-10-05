// SPDX-License-Identifier: Apache-2.0
// Compile-time, host-only probe for the fp8_gemm package kernels' by-value Params.
//
// Built and run by impls/fp8_gemm/compiler.py; it is never part of a benchmark
// run, launches nothing and needs neither a GPU nor libcuda. WORKLOAD selects
// the kernel: fp8_gemm (CutlassGroupwiseScaledGEMMSM100) or fp8_gemm_small_m
// (CutlassGroupwiseScaledGEMMSM100LowLatency). Two modes, both printing one
// JSON document to stdout:
//
//   fp8_gemm_probe layout WORKLOAD
//       sizeof/offsetof of every Params field the Python builder writes, the
//       kernel's compile-time launch constants and enum values.
//
//   fp8_gemm_probe params WORKLOAD M N K A B SFA SFB D DRIVER_VERSION SM_COUNT
//                  MAX_ACTIVE_CLUSTERS [WORKSPACE]
//       Runs CUTLASS's own GemmKernel::to_underlying_arguments for the
//       (original, unswapped) problem D[M, N] = A[M, K] B[N, K]^T exactly as
//       the FlashInfer host function builds its Arguments (including the
//       low-latency kernel's swap-AB and KernelHardwareInfo query), with fake
//       (aligned, never dereferenced) device addresses, and prints the
//       resulting Params bytes, the launch configuration
//       (get_grid_shape/get_block_shape/SharedStorageSize and the cluster
//       dimensions GemmUniversalAdapter::run launches with), the workspace
//       size, every cuTensorMapEncodeTiled call and every hardware query
//       CUTLASS made.
//
// cuTensorMapEncodeTiled is interposed: the real driver refuses to encode on
// pre-Hopper devices (CUDA_ERROR_NOT_SUPPORTED on sm_86) and the descriptor
// format is opaque, so the probe records the exact encode arguments and fills
// the 128-byte descriptor with a deterministic packing of those arguments
// (fake_tensor_map below; harness/workloads/fp8_gemm.py has the identical
// Python packing for tests). A Params byte comparison against these fixtures
// therefore checks every non-descriptor byte and every encode argument.
// The shared cudart's cudaDriverGetVersion (CUTLASS's driver-version-dependent
// descriptor fix-up), cudaGetDevice, cudaDeviceGetAttribute and
// cudaOccupancyMaxActiveClusters (KernelHardwareInfo::make_kernel_hardware_info)
// are interposed too, so that the hardware-dependent Params bytes are
// deterministic for the DRIVER_VERSION, SM_COUNT and MAX_ACTIVE_CLUSTERS given
// on the command line (MAX_ACTIVE_CLUSTERS < 0 makes the occupancy query fail,
// as it does whenever the driver rejects it; CUTLASS then uses 0). The
// recorded occupancy-query arguments tell the Python builder which query to
// repeat at run time.
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "fp8_gemm_sm100.cuh"
#include "fp8_gemm_small_m_sm100.cuh"

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
int g_sm_count = 0;
int g_max_active_clusters = 0;
std::vector<std::string> g_queries;  // JSON objects, in call order

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

std::string dims(dim3 d) {
  return "[" + std::to_string(d.x) + ", " + std::to_string(d.y) + ", " + std::to_string(d.z) + "]";
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

// The CUDA headers also define device-side versions of these two in the device
// compilation pass; the interposers are host code only.
#ifndef __CUDA_ARCH__
// KernelHardwareInfo::query_device_multiprocessor_count: cudaGetDevice, then
// cudaDeviceGetAttribute(cudaDevAttrMultiProcessorCount).
extern "C" cudaError_t CUDARTAPI cudaGetDevice(int* device) {
  *device = 0;
  g_queries.push_back("{\"call\": \"cudaGetDevice\"}");
  return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaDeviceGetAttribute(int* value, enum cudaDeviceAttr attr,
                                                        int device) {
  g_queries.push_back("{\"call\": \"cudaDeviceGetAttribute\", \"attribute\": " +
                      std::to_string(int(attr)) + ", \"device\": " + std::to_string(device) +
                      "}");
  if (attr != cudaDevAttrMultiProcessorCount) return cudaErrorInvalidValue;
  *value = g_sm_count;
  return cudaSuccess;
}
#endif  // __CUDA_ARCH__

// KernelHardwareInfo::query_device_max_active_clusters.
extern "C" cudaError_t CUDARTAPI cudaOccupancyMaxActiveClusters(
    int* numClusters, const void* func, const cudaLaunchConfig_t* config) {
  (void)func;
  std::string attrs = "[";
  for (unsigned i = 0; i < config->numAttrs; ++i) {
    auto& a = config->attrs[i];
    attrs += (i ? ", " : "");
    attrs += "{\"id\": " + std::to_string(int(a.id));
    if (a.id == cudaLaunchAttributeClusterDimension) {
      attrs += ", \"cluster_dim\": [" + std::to_string(a.val.clusterDim.x) + ", " +
               std::to_string(a.val.clusterDim.y) + ", " + std::to_string(a.val.clusterDim.z) +
               "]";
    }
    attrs += "}";
  }
  attrs += "]";
  g_queries.push_back("{\"call\": \"cudaOccupancyMaxActiveClusters\", \"grid\": " +
                      dims(config->gridDim) + ", \"block\": " + dims(config->blockDim) +
                      ", \"shared_mem\": " + std::to_string(config->dynamicSmemBytes) +
                      ", \"attrs\": " + attrs + "}");
  if (g_max_active_clusters < 0) return cudaErrorInvalidValue;
  *numClusters = g_max_active_clusters;
  return cudaSuccess;
}

namespace {

// Using-declarations, not a using-directive: cute has its own unnamed
// namespace, which a directive would make ambiguous in nvcc's generated stub.
using cute::get;
using cute::make_shape;
using cute::size;

// -- the two kernels, with their Arguments exactly as FlashInfer builds them ----

struct Fp8Gemm {
  using Kernel = fp8_gemm::GemmKernel;
  static constexpr const char* name = "fp8_gemm";
  static constexpr const char* symbol = "cutlass::device_kernel<fp8_gemm::GemmKernel>";
  static constexpr bool swap_ab = false;
  // Granularities of the kernel's own ScaleConfig (its M/N are the problem's).
  static constexpr int kernel_scale_granularity[3] = {
      fp8_gemm::ScaleGranularityM, fp8_gemm::ScaleGranularityN, fp8_gemm::ScaleGranularityK};
  static constexpr int problem_scale_granularity[3] = {
      fp8_gemm::ScaleGranularityM, fp8_gemm::ScaleGranularityN, fp8_gemm::ScaleGranularityK};

  // Mirrors flashinfer::gemm::CutlassGroupwiseScaledGEMMSM100 exactly.
  static Kernel::Arguments make_arguments(int m, int n, int k, uint64_t a, uint64_t b,
                                          uint64_t sfa, uint64_t sfb, uint64_t d) {
    using namespace fp8_gemm;
    const int l = 1;
    using StrideA = Kernel::StrideA;
    using StrideB = Kernel::StrideB;
    using StrideC = Kernel::StrideC;
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, l));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, l));
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, l));
    auto layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, l));
    auto layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, l));
    auto* D = reinterpret_cast<ElementD*>(d);
    Kernel::Arguments arguments{cutlass::gemm::GemmUniversalMode::kGemm,
                                {m, n, k, l},
                                {
                                    reinterpret_cast<ElementA*>(a),
                                    stride_A,
                                    reinterpret_cast<ElementB*>(b),
                                    stride_B,
                                    reinterpret_cast<float*>(sfa),
                                    layout_SFA,
                                    reinterpret_cast<float*>(sfb),
                                    layout_SFB,
                                },
                                {
                                    {},  // epilogue.thread
                                    D,
                                    stride_C,
                                    D,
                                    stride_C,
                                }};
    auto& fusion_args = arguments.epilogue.thread;
    fusion_args.alpha = 1.0f;
    fusion_args.beta = 0.0f;
    return arguments;
  }
};

struct Fp8GemmSmallM {
  using Kernel = fp8_gemm_small_m::GemmKernel;
  static constexpr const char* name = "fp8_gemm_small_m";
  static constexpr const char* symbol = "cutlass::device_kernel<fp8_gemm_small_m::GemmKernel>";
  static constexpr bool swap_ab = true;
  // The kernel's ScaleConfig has the M/N granularities swapped.
  static constexpr int kernel_scale_granularity[3] = {fp8_gemm_small_m::ScaleGranularityN,
                                                      fp8_gemm_small_m::ScaleGranularityM,
                                                      fp8_gemm_small_m::ScaleGranularityK};
  static constexpr int problem_scale_granularity[3] = {fp8_gemm_small_m::ScaleGranularityM,
                                                       fp8_gemm_small_m::ScaleGranularityN,
                                                       fp8_gemm_small_m::ScaleGranularityK};

  // Mirrors flashinfer::gemm::CutlassGroupwiseScaledGEMMSM100LowLatency exactly
  // (the swap, the Arguments and the KernelHardwareInfo lambda).
  static Kernel::Arguments make_arguments(int m, int n, int k, uint64_t a, uint64_t b,
                                          uint64_t sfa, uint64_t sfb, uint64_t d) {
    using namespace fp8_gemm_small_m;
    using GemmKernel = Kernel;
    const int l = 1;
    auto* A_ptr = reinterpret_cast<DTypeIn*>(a);
    auto* B_ptr = reinterpret_cast<DTypeIn*>(b);
    auto* SFA_ptr = reinterpret_cast<float*>(sfa);
    auto* SFB_ptr = reinterpret_cast<float*>(sfb);
    auto* D_ptr = reinterpret_cast<DTypeOut*>(d);
    std::swap(m, n);
    std::swap(A_ptr, B_ptr);
    std::swap(SFA_ptr, SFB_ptr);
    using StrideA = Kernel::StrideA;
    using StrideB = Kernel::StrideB;
    using StrideC = Kernel::StrideC;
    using StrideD = Kernel::StrideD;
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, l));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, l));
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, l));
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, l));
    auto layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, l));
    auto layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, l));
    Kernel::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k, l},
        {
            A_ptr,
            stride_A,
            B_ptr,
            stride_B,
            SFA_ptr,
            layout_SFA,
            SFB_ptr,
            layout_SFB,
        },
        {
            {},  // epilogue.thread
            nullptr,
            stride_C,
            D_ptr,
            stride_D,
        },
        // KernelHardwareInfo
        []() {
          // For some reason can_implement fails if this is not defined
          auto hw_info = cutlass::KernelHardwareInfo::make_kernel_hardware_info<GemmKernel>();
          hw_info.cluster_shape = {1, 1, 1};
          hw_info.cluster_shape_fallback = {1, 1, 1};
          return hw_info;
        }(),
    };
    auto& fusion_args = arguments.epilogue.thread;
    fusion_args.alpha = 1.0f;
    fusion_args.beta = 0.0f;
    return arguments;
  }
};

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

template <class V>
int layout_mode() {
  using Kernel = typename V::Kernel;
  using Params = typename Kernel::Params;
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
      field("mainloop.tma_load_a", p, ml.tma_load_a),
      field("mainloop.tma_load_a.desc", p, *ml.tma_load_a.get_tma_descriptor()),
      data_field("mainloop.tma_load_a.aux_g_stride", p, ml.tma_load_a.aux_params_.g_stride_),
      field("mainloop.tma_load_b", p, ml.tma_load_b),
      field("mainloop.tma_load_b.desc", p, *ml.tma_load_b.get_tma_descriptor()),
      data_field("mainloop.tma_load_b.aux_g_stride", p, ml.tma_load_b.aux_params_.g_stride_),
      field("mainloop.tma_load_a_fallback", p, ml.tma_load_a_fallback),
      field("mainloop.tma_load_a_fallback.desc", p, *ml.tma_load_a_fallback.get_tma_descriptor()),
      data_field("mainloop.tma_load_a_fallback.aux_g_stride", p, ml.tma_load_a_fallback.aux_params_.g_stride_),
      field("mainloop.tma_load_b_fallback", p, ml.tma_load_b_fallback),
      field("mainloop.tma_load_b_fallback.desc", p, *ml.tma_load_b_fallback.get_tma_descriptor()),
      data_field("mainloop.tma_load_b_fallback.aux_g_stride", p, ml.tma_load_b_fallback.aux_params_.g_stride_),
      field("mainloop.cluster_shape_fallback", p, ml.cluster_shape_fallback),
      field("mainloop.runtime_data_type_a", p, ml.runtime_data_type_a),
      field("mainloop.runtime_data_type_b", p, ml.runtime_data_type_b),
      field("mainloop.ptr_SFA", p, ml.ptr_SFA),
      field("mainloop.layout_SFA", p, ml.layout_SFA),
      field("mainloop.layout_SFA.shape_m_blocks", p, get<0, 1>(ml.layout_SFA.shape())),
      field("mainloop.layout_SFA.shape_k_blocks", p, get<1, 1>(ml.layout_SFA.shape())),
      field("mainloop.layout_SFA.shape_l", p, get<2>(ml.layout_SFA.shape())),
      field("mainloop.layout_SFA.stride_m", p, get<0, 1>(ml.layout_SFA.stride())),
      field("mainloop.layout_SFA.stride_l", p, get<2>(ml.layout_SFA.stride())),
      field("mainloop.ptr_SFB", p, ml.ptr_SFB),
      field("mainloop.layout_SFB", p, ml.layout_SFB),
      field("mainloop.layout_SFB.shape_n_blocks", p, get<0, 1>(ml.layout_SFB.shape())),
      field("mainloop.layout_SFB.shape_k_blocks", p, get<1, 1>(ml.layout_SFB.shape())),
      field("mainloop.layout_SFB.shape_l", p, get<2>(ml.layout_SFB.shape())),
      field("mainloop.layout_SFB.stride_n", p, get<0, 1>(ml.layout_SFB.stride())),
      field("mainloop.layout_SFB.stride_l", p, get<2>(ml.layout_SFB.stride())),
      field("epilogue", p, p.epilogue),
      field("epilogue.thread", p, ep.thread),
      field("epilogue.tma_load_c", p, ep.tma_load_c),
      field("epilogue.tma_load_c.desc", p, *ep.tma_load_c.get_tma_descriptor()),
      data_field("epilogue.tma_load_c.aux_g_stride", p, ep.tma_load_c.aux_params_.g_stride_),
      field("epilogue.tma_store_d", p, ep.tma_store_d),
      field("epilogue.tma_store_d.desc", p, *ep.tma_store_d.get_tma_descriptor()),
      data_field("epilogue.tma_store_d.aux_g_stride", p, ep.tma_store_d.aux_params_.g_stride_),
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

  std::printf("{\n");
  std::printf("  \"workload\": \"%s\",\n", V::name);
  std::printf("  \"kernel_symbol_check\": \"%s\",\n", V::symbol);
  std::printf("  \"params_size\": %zu,\n  \"params_align\": %zu,\n", sizeof(Params),
              alignof(Params));
  std::printf("  \"fields\": {\n");
  for (size_t i = 0; i < fields.size(); ++i) {
    std::printf("    \"%s\": {\"offset\": %zu, \"size\": %zu}%s\n", fields[i].name.c_str(),
                fields[i].offset, fields[i].size, i + 1 < fields.size() ? "," : "");
  }
  std::printf("  },\n");
  using TileShape = typename Kernel::TileShape;
  using CtaShape = typename Kernel::CtaShape_MNK;
  using AtomThr = typename Kernel::AtomThrShapeMNK;
  using ClusterShape = typename Kernel::ClusterShape;
  std::printf("  \"constants\": {\n");
  std::printf("    \"shared_storage_size\": %d,\n", Kernel::SharedStorageSize);
  std::printf("    \"max_threads_per_block\": %u,\n", Kernel::MaxThreadsPerBlock);
  std::printf("    \"min_blocks_per_multiprocessor\": %u,\n", Kernel::MinBlocksPerMultiprocessor);
  std::printf("    \"tile_shape_mnk\": [%d, %d, %d],\n", int(size<0>(TileShape{})),
              int(size<1>(TileShape{})), int(size<2>(TileShape{})));
  std::printf("    \"cta_shape_mnk\": [%d, %d, %d],\n", int(size<0>(CtaShape{})),
              int(size<1>(CtaShape{})), int(size<2>(CtaShape{})));
  std::printf("    \"atom_thr_shape_mnk\": [%d, %d, %d],\n", int(size<0>(AtomThr{})),
              int(size<1>(AtomThr{})), int(size<2>(AtomThr{})));
  if constexpr (Kernel::IsDynamicCluster) {
    // Only the static extents are meaningful; the dynamic ones are 0.
    std::printf("    \"cluster_shape_mnk\": null,\n");
  } else {
    std::printf("    \"cluster_shape_mnk\": [%d, %d, %d],\n", int(size<0>(ClusterShape{})),
                int(size<1>(ClusterShape{})), int(size<2>(ClusterShape{})));
  }
  std::printf("    \"is_dynamic_cluster\": %s,\n", Kernel::IsDynamicCluster ? "true" : "false");
  std::printf("    \"swap_ab\": %s,\n", V::swap_ab ? "true" : "false");
  std::printf("    \"epilogue_source_supported\": %s,\n",
              std::is_void_v<typename Kernel::ElementC> ? "false" : "true");
  std::printf("    \"mainloop_stages\": %d,\n", Kernel::DispatchPolicy::Stages);
  std::printf("    \"scheduler_pipeline_stages\": %u,\n", Kernel::SchedulerPipelineStageCount);
  std::printf("    \"accumulator_pipeline_stages\": %u,\n", Kernel::AccumulatorPipelineStageCount);
  std::printf("    \"scale_granularity_mnk\": [%d, %d, %d],\n", V::kernel_scale_granularity[0],
              V::kernel_scale_granularity[1], V::kernel_scale_granularity[2]);
  std::printf("    \"problem_scale_granularity_mnk\": [%d, %d, %d],\n",
              V::problem_scale_granularity[0], V::problem_scale_granularity[1],
              V::problem_scale_granularity[2]);
  std::printf("    \"gemm_mode_kGemm\": %d,\n", int(cutlass::gemm::GemmUniversalMode::kGemm));
  using RasterOrder = typename Kernel::TileScheduler::RasterOrder;
  std::printf("    \"raster_order_AlongM\": %d,\n", int(RasterOrder::AlongM));
  std::printf("    \"raster_order_AlongN\": %d,\n", int(RasterOrder::AlongN));
  std::printf("    \"sizeof_TileScheduler_Params\": %zu,\n",
              sizeof(typename Kernel::TileSchedulerParams));
  std::printf("    \"sizeof_FastDivmod\": %zu\n", sizeof(cutlass::FastDivmod));
  std::printf("  }\n}\n");
  return 0;
}

template <class V>
int params_mode(int argc, char** argv) {
  using Kernel = typename V::Kernel;
  using Params = typename Kernel::Params;
  using Arguments = typename Kernel::Arguments;
  if (argc < 14) {
    std::fprintf(stderr,
                 "usage: params WORKLOAD M N K A B SFA SFB D DRIVER_VERSION SM_COUNT "
                 "MAX_ACTIVE_CLUSTERS [WORKSPACE]\n");
    return 1;
  }
  int m = std::atoi(argv[3]), n = std::atoi(argv[4]), k = std::atoi(argv[5]);
  auto ptr = [&](int i) { return std::strtoull(argv[i], nullptr, 0); };
  uint64_t a = ptr(6), b = ptr(7), sfa = ptr(8), sfb = ptr(9), d = ptr(10);
  g_driver_version = std::atoi(argv[11]);
  g_sm_count = std::atoi(argv[12]);
  g_max_active_clusters = std::atoi(argv[13]);
  void* workspace = argc > 14 ? reinterpret_cast<void*>(ptr(14)) : nullptr;
  Arguments args = V::make_arguments(m, n, k, a, b, sfa, sfb, d);
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
  std::printf("  \"workload\": \"%s\",\n", V::name);
  std::printf("  \"problem\": {\"m\": %d, \"n\": %d, \"k\": %d},\n", m, n, k);
  std::printf(
      "  \"pointers\": {\"a\": %llu, \"b\": %llu, \"sfa\": %llu, \"sfb\": %llu, \"d\": %llu, "
      "\"workspace\": %llu},\n",
      (unsigned long long)a, (unsigned long long)b, (unsigned long long)sfa,
      (unsigned long long)sfb, (unsigned long long)d,
      (unsigned long long)reinterpret_cast<uint64_t>(workspace));
  std::printf("  \"driver_version\": %d,\n", g_driver_version);
  std::printf("  \"sm_count\": %d,\n", g_sm_count);
  std::printf("  \"max_active_clusters\": %d,\n", g_max_active_clusters);
  std::printf("  \"hardware_queries\": [");
  for (size_t i = 0; i < g_queries.size(); ++i) {
    std::printf("%s\n    %s", i ? "," : "", g_queries[i].c_str());
  }
  std::printf("%s],\n", g_queries.empty() ? "" : "\n  ");
  std::printf("  \"can_implement\": %s,\n", can_implement ? "true" : "false");
  std::printf("  \"workspace_size\": %zu,\n", workspace_size);
  std::printf("  \"grid\": [%u, %u, %u],\n  \"block\": [%u, %u, %u],\n", grid.x, grid.y, grid.z,
              block.x, block.y, block.z);
  std::printf("  \"shared_mem\": %d,\n", Kernel::SharedStorageSize);
  // GemmUniversalAdapter::run: a static 1x1x1 cluster kernel is launched
  // without cluster attributes; a dynamic-cluster kernel through
  // ClusterLauncher::launch_with_fallback_cluster with the cluster dimension
  // attribute set to hw_info.cluster_shape_fallback and the preferred cluster
  // dimension to hw_info.cluster_shape.
  if constexpr (Kernel::IsDynamicCluster) {
    std::printf("  \"cluster\": %s,\n  \"preferred_cluster\": %s,\n",
                dims(params->hw_info.cluster_shape_fallback).c_str(),
                dims(params->hw_info.cluster_shape).c_str());
  } else {
    static_assert(cute::size(typename Kernel::ClusterShape{}) == 1, "static 1x1x1 cluster");
    std::printf("  \"cluster\": null,\n  \"preferred_cluster\": null,\n");
  }
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

template <class V>
int dispatch(const std::string& mode, int argc, char** argv) {
  if (mode == "layout") return layout_mode<V>();
  return params_mode<V>(argc, argv);
}

}  // namespace

int main(int argc, char** argv) {
  std::string mode = argc > 1 ? argv[1] : "";
  std::string workload = argc > 2 ? argv[2] : "";
  if (mode == "layout" || mode == "params") {
    if (workload == Fp8Gemm::name) return dispatch<Fp8Gemm>(mode, argc, argv);
    if (workload == Fp8GemmSmallM::name) return dispatch<Fp8GemmSmallM>(mode, argc, argv);
  }
  std::fprintf(stderr,
               "usage: %s layout WORKLOAD | params WORKLOAD M N K A B SFA SFB D DRIVER_VERSION "
               "SM_COUNT MAX_ACTIVE_CLUSTERS [WORKSPACE]\n"
               "WORKLOAD: fp8_gemm | fp8_gemm_small_m\n",
               argv[0]);
  return 1;
}
