// SPDX-License-Identifier: Apache-2.0
// Compile-time, host-only probe of FlashInfer v0.2.10's routed FP8 block-scale
// MoE host code (DeepSeek-V3 routing, trtllm-gen batched GEMMs).
//
// Built and run by impls/moe/compiler.py; never part of a benchmark run. It
// needs neither a GPU nor libcuda: every CUDA runtime/driver entry point the
// upstream host code reaches is interposed below and only *recorded*.
//
// The pinned upstream sources are included verbatim (routing, dev kernels,
// MoE runner, batched-GEMM runner + the full 408-config KernelMetaInfo.h), and
// run_mode() repeats the call sequence of csrc/trtllm_fused_moe_kernel_launcher.cu
// (trtllm_fp8_block_scale_moe -> trtllm_fp8_block_scale_moe_launcher) with
// fake, aligned, never-dereferenced device addresses instead of tensors.
//
//   moe_probe layout
//       sizeof/offsetof of every field of the by-value kernel parameter
//       structs the Python builders write.
//   moe_probe run TOKENS LOCAL_EXPERT_OFFSET ROUTED_SCALING_FACTOR
//       One JSON document: buffer sizes, the MoE config upstream selects, and
//       every launch (kernel, grid, block, dynamic smem, launch attributes,
//       parameter bytes, bytes that are indeterminate in upstream's struct,
//       and the cuTensorMapEncodeTiled calls made for it).
//
// Indeterminate bytes: upstream default-initializes its parameter structs on
// the stack, leaving unset members (e.g. the dynamic-batch GEMM's unused CTA
// tables and descriptors) indeterminate. The whole flow is run twice with the
// stack poisoned by 0x00 and then 0xff, at the start of each run and again in
// the interposed cudaDeviceGetAttribute (called right before each GEMM's
// KernelParams is built); parameter bytes that differ are reported. Struct
// padding not covered by any field (see layout mode) is never read either.
//
// Tensor maps: cuTensorMapEncodeTiled is interposed; the real driver refuses to
// encode on sm_86 and the descriptor is opaque. The probe fills each 128-byte
// descriptor with a deterministic packing of its encode arguments (see
// fake_tensor_map; harness/workloads/moe.py has the identical Python packing).
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <map>
#include <new>
#include <string>
#include <vector>

// The four pinned upstream translation units, in one TU.
#include "csrc/trtllm_fused_moe_routing_deepseek.cu"
#include "csrc/trtllm_fused_moe_dev_kernel.cu"
#include "csrc/trtllm_fused_moe_runner.cu"
#include "csrc/trtllm_batched_gemm_runner.cu"

namespace {

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

struct EncodeCall {
  int data_type, rank, interleave, swizzle, l2_promotion, oob_fill;
  uint64_t address;
  std::vector<uint64_t> dims, strides;
  std::vector<uint32_t> box, element_strides;
};

struct Launch {
  std::string api;  // "runtime" or "driver"
  std::string kernel;
  unsigned grid[3], block[3];
  unsigned shared_mem;
  std::string attrs;  // JSON array
  std::vector<unsigned char> params;
  std::vector<EncodeCall> tensor_maps;  // encodes since the previous launch
};

std::vector<Launch> g_launches;
std::vector<EncodeCall> g_pending_encodes;
std::vector<std::string> g_events;  // other recorded calls (JSON objects)
std::vector<std::string> g_function_names;
int g_sm_count = 148;
unsigned char g_fill = 0;  // the current run's stack poison byte
__attribute__((noinline)) void poison_stack(unsigned char fill);

// Fill a large stack region with `fill` (see the header comment).
__attribute__((noinline)) void poison_stack(unsigned char fill) {
  volatile unsigned char region[1 << 20];
  for (size_t i = 0; i < sizeof(region); ++i) region[i] = fill;
}

void fake_tensor_map(const EncodeCall& c, CUtensorMap* out) {
  // u8 data_type, u8 rank, u8 interleave, u8 swizzle, u8 l2, u8 oob, u16 0,
  // u64 address, u64 dims[5], u64 strides[4], u32 box[5], u32 element_strides[5]
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

// Kernel host stubs -> (name, parameter size) for runtime launches. Both
// UsePdl instantiations are listed: upstream launches UsePdl = true.
struct KernelEntry {
  std::string name;
  size_t param_size;
};
std::map<const void*, KernelEntry> g_kernels;

namespace rd = moe::dev::routing::routingDeepSeek;
namespace act = moe::dev::activation;
namespace fin = moe::dev::finalize;

template <bool Pdl>
void register_kernels() {
  using R = rd::KernelParams<float, __nv_bfloat16, true, Pdl>;
  using A = act::KernelParams<cutlass::float_e4m3_t, Pdl>;
  using F = fin::KernelParams<cutlass::bfloat16_t, cutlass::bfloat16_t, Pdl>;
  std::string pdl = Pdl ? "true" : "false";
  g_kernels[(const void*)rd::routingMainKernel<R>] = {"routingMainKernel<UsePdl=" + pdl + ">", sizeof(R)};
  g_kernels[(const void*)rd::routingIndicesClusterKernel<R>] = {
      "routingIndicesClusterKernel<UsePdl=" + pdl + ">", sizeof(R)};
  g_kernels[(const void*)rd::routingIndicesCoopKernel<R>] = {
      "routingIndicesCoopKernel<UsePdl=" + pdl + ">", sizeof(R)};
  g_kernels[(const void*)moe::dev::routing::routingIndicesHistogramKernel<R>] = {
      "routingIndicesHistogramKernel<UsePdl=" + pdl + ">", sizeof(R)};
  g_kernels[(const void*)moe::dev::routing::routingIndicesOffsetsKernel<R>] = {
      "routingIndicesOffsetsKernel<UsePdl=" + pdl + ">", sizeof(R)};
  g_kernels[(const void*)act::activationDeepSeekKernel<A>] = {
      "activationDeepSeekKernel<UsePdl=" + pdl + ">", sizeof(A)};
  g_kernels[(const void*)fin::finalizeKernel<F>] = {"finalizeKernel<UsePdl=" + pdl + ">", sizeof(F)};
  g_kernels[(const void*)fin::finalizeKernelVecLoad<F>] = {
      "finalizeKernelVecLoad<UsePdl=" + pdl + ">", sizeof(F)};
}

std::string runtime_attrs(const cudaLaunchConfig_t* config) {
  std::string out = "[";
  for (unsigned i = 0; i < config->numAttrs; ++i) {
    auto& a = config->attrs[i];
    out += i ? ", " : "";
    out += "{\"id\": " + std::to_string(int(a.id));
    if (a.id == cudaLaunchAttributeProgrammaticStreamSerialization) {
      out += ", \"value\": " + std::to_string(int(a.val.programmaticStreamSerializationAllowed));
    } else if (a.id == cudaLaunchAttributeCooperative) {
      out += ", \"value\": " + std::to_string(int(a.val.cooperative));
    } else {
      out += ", \"value\": null";
    }
    out += "}";
  }
  return out + "]";
}

}  // namespace

// ---- interposed CUDA runtime (host side of the shared cudart) ---------------
#ifndef __CUDA_ARCH__
extern "C" cudaError_t CUDARTAPI cudaLaunchKernelExC(const cudaLaunchConfig_t* config,
                                                     const void* func, void** args) {
  auto it = g_kernels.find(func);
  if (it == g_kernels.end()) {
    std::fprintf(stderr, "moe_probe: launch of an unregistered kernel\n");
    std::abort();
  }
  Launch l;
  l.api = "runtime";
  l.kernel = it->second.name;
  l.grid[0] = config->gridDim.x, l.grid[1] = config->gridDim.y, l.grid[2] = config->gridDim.z;
  l.block[0] = config->blockDim.x, l.block[1] = config->blockDim.y, l.block[2] = config->blockDim.z;
  l.shared_mem = static_cast<unsigned>(config->dynamicSmemBytes);
  l.attrs = runtime_attrs(config);
  auto* bytes = static_cast<unsigned char*>(args[0]);
  l.params.assign(bytes, bytes + it->second.param_size);
  l.tensor_maps.swap(g_pending_encodes);
  g_launches.push_back(l);
  return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaDeviceGetAttribute(int* value, enum cudaDeviceAttr attr,
                                                        int device) {
  // TrtllmGenBatchedGemmRunner::run queries the SM count right before
  // BatchedGemmInterface::run builds the GEMM's KernelParams at this stack
  // depth: poison the stack below here so its unset members (the dynamic
  // batch's CTA tables, unused descriptors) differ between the two runs.
  poison_stack(g_fill);
  g_events.push_back("{\"call\": \"cudaDeviceGetAttribute\", \"attribute\": " +
                     std::to_string(int(attr)) + ", \"device\": " + std::to_string(device) + "}");
  if (attr != cudaDevAttrMultiProcessorCount) return cudaErrorInvalidValue;
  *value = g_sm_count;
  return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaMemsetAsync(void* ptr, int value, size_t count,
                                                 cudaStream_t) {
  g_events.push_back("{\"call\": \"cudaMemsetAsync\", \"ptr\": " +
                     std::to_string(reinterpret_cast<uint64_t>(ptr)) + ", \"value\": " +
                     std::to_string(value) + ", \"count\": " + std::to_string(count) + "}");
  return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaFuncSetAttribute(const void*, enum cudaFuncAttribute attr,
                                                      int value) {
  g_events.push_back("{\"call\": \"cudaFuncSetAttribute\", \"attribute\": " +
                     std::to_string(int(attr)) + ", \"value\": " + std::to_string(value) + "}");
  return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaStreamIsCapturing(cudaStream_t,
                                                       cudaStreamCaptureStatus* status) {
  *status = cudaStreamCaptureStatusNone;
  return cudaSuccess;
}
#endif  // __CUDA_ARCH__

// ---- interposed CUDA driver (the probe does not link libcuda) ---------------
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
  g_pending_encodes.push_back(call);
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuModuleLoadData(CUmodule* module, const void*) {
  static int fake_module;
  *module = reinterpret_cast<CUmodule>(&fake_module);
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuModuleUnload(CUmodule) { return CUDA_SUCCESS; }

extern "C" CUresult CUDAAPI cuModuleGetFunction(CUfunction* function, CUmodule,
                                                const char* name) {
  g_function_names.push_back(name);
  // Encodes the index into g_function_names (offset by one so it is non-null).
  *function = reinterpret_cast<CUfunction>(static_cast<uintptr_t>(g_function_names.size()));
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuFuncSetAttribute(CUfunction function, CUfunction_attribute attrib,
                                               int value) {
  auto index = reinterpret_cast<uintptr_t>(function) - 1;
  g_events.push_back("{\"call\": \"cuFuncSetAttribute\", \"function\": \"" +
                     g_function_names.at(index) + "\", \"attribute\": " +
                     std::to_string(int(attrib)) + ", \"value\": " + std::to_string(value) + "}");
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuLaunchKernelEx(const CUlaunchConfig* config, CUfunction f,
                                             void** kernelParams, void** extra) {
  if (extra != nullptr) return CUDA_ERROR_INVALID_VALUE;
  Launch l;
  l.api = "driver";
  l.kernel = g_function_names.at(reinterpret_cast<uintptr_t>(f) - 1);
  l.grid[0] = config->gridDimX, l.grid[1] = config->gridDimY, l.grid[2] = config->gridDimZ;
  l.block[0] = config->blockDimX, l.block[1] = config->blockDimY, l.block[2] = config->blockDimZ;
  l.shared_mem = config->sharedMemBytes;
  std::string attrs = "[";
  for (unsigned i = 0; i < config->numAttrs; ++i) {
    auto& a = config->attrs[i];
    attrs += i ? ", " : "";
    attrs += "{\"id\": " + std::to_string(int(a.id));
    if (a.id == CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION) {
      attrs += ", \"value\": [" + std::to_string(a.value.clusterDim.x) + ", " +
               std::to_string(a.value.clusterDim.y) + ", " +
               std::to_string(a.value.clusterDim.z) + "]";
    } else if (a.id == CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE) {
      attrs += ", \"value\": " + std::to_string(int(a.value.clusterSchedulingPolicyPreference));
    } else if (a.id == CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION) {
      attrs += ", \"value\": " + std::to_string(int(a.value.programmaticStreamSerializationAllowed));
    } else {
      attrs += ", \"value\": null";
    }
    attrs += "}";
  }
  l.attrs = attrs + "]";
  auto* bytes = static_cast<unsigned char*>(kernelParams[0]);
  l.params.assign(bytes, bytes + sizeof(batchedGemm::KernelParams));
  l.tensor_maps.swap(g_pending_encodes);
  g_launches.push_back(l);
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuCtxGetCurrent(CUcontext* ctx) {
  static int fake_context;
  *ctx = reinterpret_cast<CUcontext>(&fake_context);
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuCtxGetId(CUcontext, unsigned long long* id) {
  *id = 1;
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuGetErrorString(CUresult, const char** str) {
  *str = "moe_probe";
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuGetErrorName(CUresult, const char** str) {
  *str = "moe_probe";
  return CUDA_SUCCESS;
}

// ---- other upstream link dependencies -----------------------------------------
namespace flashinfer::trtllm_cubin_loader {
// BatchedGemmInterface::run loads the cubin by name; the module load itself is
// interposed, so no bytes are needed.
std::string getCubin(const std::string&, const std::string&) { return std::string(); }
}  // namespace flashinfer::trtllm_cubin_loader

// Routing::Runner::run references the other routing methods' launchers; this
// package only runs DeepSeekV3, so reaching either aborts the probe.
namespace moe::dev::routing {
namespace routingLlama4 {
void run(Data const&, void*) {
  std::fprintf(stderr, "moe_probe: Llama4 routing is not part of this package\n");
  std::abort();
}
}  // namespace routingLlama4
namespace routingRenormalize {
void run(Data const&, void*) {
  std::fprintf(stderr, "moe_probe: Renormalize routing is not part of this package\n");
  std::abort();
}
}  // namespace routingRenormalize
}  // namespace moe::dev::routing

namespace tensorrt_llm::common {
// csrc/nv_internal/cpp/common/envUtils.cpp: PDL for the batched GEMMs only when
// TRTLLM_ENABLE_PDL=1 (on sm_90+); unset by default.
bool getEnvEnablePDL() { return false; }
}  // namespace tensorrt_llm::common

namespace {

namespace moe_ns = tensorrt_llm::kernels::trtllmgen_moe;
namespace btg = batchedGemm::trtllm::gen;
namespace tg = batchedGemm::trtllm::gen;


// Fake device address of each launcher buffer: 1 GiB aligned, distinct.
std::vector<std::pair<std::string, uint64_t>> g_pointers;
void* fake(const std::string& name) {
  uint64_t address = 0x7F0000000000ull + (uint64_t(g_pointers.size() + 1) << 30);
  g_pointers.emplace_back(name, address);
  return reinterpret_cast<void*>(address);
}

constexpr int kNumExperts = 256, kTopK = 8, kNGroup = 8, kTopkGroup = 4;
constexpr int kHidden = 7168, kIntermediate = 2048, kLocalExperts = 32, kTile = 8;

struct RunResult {
  int64_t config_index;
  int32_t workspace_fc1, workspace_fc2;
  int32_t max_num_padded_tokens, max_num_ctas;
};

// trtllm_fp8_block_scale_moe + trtllm_fp8_block_scale_moe_launcher with
// fake pointers (csrc/trtllm_fused_moe_kernel_launcher.cu).
__attribute__((noinline)) RunResult run_once(int tokens, int offset, float scale) {
  g_pointers.clear();
  btg::Dtype mDtypeElt{btg::Dtype::E4m3};
  moe_ns::MoE::Runner runner(mDtypeElt, /*useDeepSeekFp8=*/true, kTile,
                             /*use_shuffled_weight=*/false,
                             static_cast<batchedGemm::gemm::MatrixLayout>(0));
  RunResult result{};
  result.config_index =
      runner.getDefaultValidConfigIndex(kTopK, kHidden, kIntermediate, kLocalExperts, tokens);

  moe_ns::MoE::MoERunnerArgs args;
  moe_ns::MoE::MoEWorkspace workspace;
  args.mDtypeElt = btg::Dtype::E4m3;
  args.mDtypeExpW = btg::Dtype::Bfloat16;  // routing_bias is bfloat16
  args.routing_logits = fake("routing_logits");
  args.routing_bias = fake("routing_bias");
  args.hidden_states = fake("hidden_states");
  args.hidden_states_scale = fake("hidden_states_scale");
  args.gemm1_weights = fake("gemm1_weights");
  args.gemm1_weights_scale = fake("gemm1_weights_scale");
  args.gemm2_weights = fake("gemm2_weights");
  args.gemm2_weights_scale = fake("gemm2_weights_scale");
  args.num_tokens = tokens;
  args.num_experts = kNumExperts;
  args.hidden_size = kHidden;
  args.hidden_size_output = args.hidden_size;
  args.top_k = kTopK;
  args.n_group = kNGroup;
  args.topk_group = kTopkGroup;
  args.local_expert_offset = offset;
  args.local_num_experts = kLocalExperts;
  args.routed_scaling_factor = scale;
  args.intermediate_size = kIntermediate;
  args.mUseDeepSeekFp8 = true;

  void* num_tokens_per_expert = fake("num_tokens_per_expert");
  result.max_num_padded_tokens =
      moe_ns::Routing::getMaxPermutedPaddedCount(tokens, kTopK, kNumExperts, kTile);
  auto* total_num_padded_tokens = static_cast<int*>(fake("total_num_padded_tokens"));
  auto* expanded_idx_to_permuted_idx = static_cast<int*>(fake("expanded_idx_to_permuted_idx"));
  auto* permuted_idx_to_token_idx = static_cast<int*>(fake("permuted_idx_to_token_idx"));
  void* expert_weights = fake("expert_weights");
  auto* expert_indexes = static_cast<int*>(fake("expert_indexes"));
  auto* expert_count_histogram = static_cast<int*>(fake("expert_count_histogram"));
  void* gemm1_output = fake("gemm1_output");
  auto* gemm1_output_scale = static_cast<float*>(fake("gemm1_output_scale"));
  void* activation_output = fake("activation_output");
  auto* activation_output_scale = static_cast<float*>(fake("activation_output_scale"));
  void* gemm2_output = fake("gemm2_output");
  result.max_num_ctas =
      moe_ns::Routing::getMaxNumCtasInBatchDim(tokens, kTopK, kNumExperts, kTile);
  auto* cta_idx_xy_to_batch_idx = static_cast<int*>(fake("cta_idx_xy_to_batch_idx"));
  auto* cta_idx_xy_to_mn_limit = static_cast<int*>(fake("cta_idx_xy_to_mn_limit"));
  auto* num_non_exiting_ctas = static_cast<int*>(fake("num_non_exiting_ctas"));

  moe_ns::Routing::Runner routing_runner(kTile);
  routing_runner.run(args.routing_logits, args.routing_bias, args.num_tokens, args.num_experts,
                     args.top_k, args.n_group, args.topk_group, args.local_expert_offset,
                     args.local_num_experts, args.routed_scaling_factor, expert_indexes,
                     expert_count_histogram, total_num_padded_tokens,
                     expanded_idx_to_permuted_idx, nullptr, permuted_idx_to_token_idx,
                     expert_weights, static_cast<int*>(num_tokens_per_expert),
                     cta_idx_xy_to_batch_idx, cta_idx_xy_to_mn_limit, num_non_exiting_ctas,
                     args.mDtypeElt, false, true,
                     moe_ns::Routing::RoutingMethodType::DeepSeekV3, /*stream=*/nullptr);

  void* output = fake("output");
  workspace.total_num_padded_tokens = total_num_padded_tokens;
  workspace.total_max_padded_tokens = result.max_num_padded_tokens;
  workspace.ProjUpTileN = kTile;
  workspace.routing_expert_indexes = expert_indexes;
  workspace.permuted_idx_size = total_num_padded_tokens;
  workspace.expanded_idx_to_permuted_idx = expanded_idx_to_permuted_idx;
  workspace.permuted_idx_to_token_idx = permuted_idx_to_token_idx;
  workspace.expert_weights = expert_weights;
  workspace.cta_idx_xy_to_batch_idx = cta_idx_xy_to_batch_idx;
  workspace.cta_idx_xy_to_mn_limit = cta_idx_xy_to_mn_limit;
  workspace.num_non_exiting_ctas = num_non_exiting_ctas;
  workspace.gemm1_output = gemm1_output;
  workspace.gemm1_output_scale = gemm1_output_scale;
  workspace.activation_output = activation_output;
  workspace.activation_output_scale = activation_output_scale;
  workspace.gemm2_output = gemm2_output;
  workspace.gemm2_output_scale = nullptr;
  args.output = output;
  args.output_scale = nullptr;

  auto sizes = runner.getWorkspaceSizeInBytes(args, result.config_index);
  result.workspace_fc1 = std::get<0>(sizes);
  result.workspace_fc2 = std::get<1>(sizes);
  // at::empty of 0 bytes: no storage. Nonzero sizes are reported and rejected
  // by the compiler (the package allocates no GEMM workspace).
  workspace.bmm1_workspace = result.workspace_fc1 ? fake("workspace_fc1") : nullptr;
  workspace.bmm2_workspace = result.workspace_fc2 ? fake("workspace_fc2") : nullptr;
  runner.run(args, workspace, /*device=*/0, /*stream=*/nullptr, result.config_index);
  return result;
}

uint64_t pointer(const char* name) {
  for (auto& [n, address] : g_pointers) {
    if (n == name) return address;
  }
  return 0;  // buffers upstream passes as nullptr
}

// Bytes of a GEMM launch's KernelParams that KernelParamsSetup::setKernelParams
// leaves unset: rebuild it directly (BatchedGemmInterface::run's options and
// the transposed runner's operand swap, as in the recorded launch) into
// storage pre-filled with 0x00 and then 0xff. The rebuild must reproduce the
// recorded bytes everywhere else; the probe aborts otherwise.
std::vector<size_t> gemm_unset_bytes(const Launch& launch, int tokens) {
  auto const bmm = batchedGemm::batchedGemm::BatchedGemmInterface();
  auto const* configs = bmm.getBatchedGemmConfigs();
  const batchedGemm::batchedGemm::BatchedGemmConfig* config = nullptr;
  for (size_t i = 0; i < bmm.getNumBatchedGemmConfigs(); ++i) {
    if (launch.kernel == configs[i].mFunctionName) config = &configs[i];
  }
  if (config == nullptr) std::abort();
  bool const fc1 = config->mOptions.mRouteImpl != batchedGemm::batchedGemm::RouteImpl::NoRoute;
  int const n_gemm = fc1 ? 2 * kIntermediate : kHidden;  // the runner's n
  int const k_gemm = fc1 ? kHidden : kIntermediate;
  // BatchedGemmInterface::getOptionsFromConfigAndData for the runner's data.
  batchedGemm::batchedGemm::BatchedGemmOptions options = config->mOptions;
  options.mM = n_gemm;  // transposeMmaOutput: M = n, N = m (tokens)
  options.mN = tokens;
  options.mK = k_gemm;
  options.mBatchedM = {};
  options.mBatchedN = {};
  options.mBatchMode = batchedGemm::batchedGemm::BatchedGemmOptions::BatchMode::BatchN;
  options.mNumBatches = kLocalExperts;
  options.mNumTokens = tokens;
  int32_t const max_ctas =
      moe_ns::Routing::getMaxNumCtasInBatchDim(tokens, kTopK, kLocalExperts, kTile);
  auto p = [](const char* name) { return reinterpret_cast<void*>(pointer(name)); };
  // TrtllmGenBatchedGemmRunner::run swaps (a, sfA, perTokensSfA) with
  // (b, sfB, perTokensSfB); PermuteGemm1 passes the routing weights as
  // perTokensSfA, Gemm2 passes no per-token scales.
  void* a = fc1 ? p("gemm1_weights") : p("gemm2_weights");
  void* sf_a = fc1 ? p("gemm1_weights_scale") : p("gemm2_weights_scale");
  void* b = fc1 ? p("hidden_states") : p("activation_output");
  void* sf_b = fc1 ? p("hidden_states_scale") : p("activation_output_scale");
  void* c = fc1 ? p("gemm1_output") : p("gemm2_output");
  void* sf_c = fc1 ? p("gemm1_output_scale") : nullptr;
  void* per_token_b = fc1 ? p("expert_weights") : nullptr;
  auto* route = fc1 ? static_cast<int32_t const*>(p("permuted_idx_to_token_idx")) : nullptr;
  using Params = batchedGemm::KernelParams;
  alignas(Params) static unsigned char storage[2][sizeof(Params)];
  for (int run = 0; run < 2; ++run) {
    unsigned char fill = run ? 0xff : 0x00;
    std::memset(storage[run], fill, sizeof(Params));
    poison_stack(fill);
    new (storage[run]) Params(batchedGemm::batchedGemm::KernelParamsSetup::setKernelParams(
        options, /*batchM=*/false, a, b, c, sf_a, sf_b, /*perTokenSfA=*/nullptr, per_token_b,
        /*bias=*/nullptr, sf_c, /*scaleC=*/nullptr, /*scaleGate=*/nullptr,
        /*clampLimit=*/nullptr, /*swiGluAlpha=*/nullptr, /*swiGluBeta=*/nullptr, route,
        /*rowMax=*/nullptr, /*rowMaxBars=*/nullptr,
        static_cast<int32_t const*>(p("num_non_exiting_ctas")),
        static_cast<int32_t const*>(p("total_num_padded_tokens")),
        static_cast<int32_t const*>(p("cta_idx_xy_to_batch_idx")),
        static_cast<int32_t const*>(p("cta_idx_xy_to_mn_limit")), max_ctas));
  }
  g_pending_encodes.clear();
  std::vector<size_t> unset;
  for (size_t i = 0; i < sizeof(Params); ++i) {
    if (storage[0][i] != storage[1][i]) {
      unset.push_back(i);
    } else if (storage[0][i] != launch.params.at(i)) {
      std::fprintf(stderr, "moe_probe: setKernelParams rebuild differs at byte %zu\n", i);
      std::abort();
    }
  }
  return unset;
}

int run_mode(int argc, char** argv) {
  if (argc != 5) {
    std::fprintf(stderr, "usage: moe_probe run TOKENS LOCAL_EXPERT_OFFSET SCALE\n");
    return 1;
  }
  int tokens = std::atoi(argv[2]), offset = std::atoi(argv[3]);
  float scale = static_cast<float>(std::atof(argv[4]));
  register_kernels<true>();
  register_kernels<false>();
  std::vector<Launch> runs[2];
  RunResult result{};
  std::vector<std::string> events;
  for (int run = 0; run < 2; ++run) {
    g_launches.clear();
    g_events.clear();
    // g_function_names persists: upstream's global module cache reuses the
    // CUfunction handles of the first run.
    g_fill = run ? 0xff : 0x00;
    poison_stack(g_fill);
    result = run_once(tokens, offset, scale);
    runs[run] = g_launches;
    if (run == 0) events = g_events;
  }
  if (runs[0].size() != runs[1].size()) {
    std::fprintf(stderr, "moe_probe: the two runs launched differently\n");
    return 1;
  }
  std::printf("{\n  \"tokens\": %d,\n  \"local_expert_offset\": %d,\n", tokens, offset);
  std::printf("  \"routed_scaling_factor_hex\": \"%s\",\n", hex(&scale, 4).c_str());
  std::printf("  \"sm_count\": %d,\n", g_sm_count);
  std::printf("  \"moe_config_index\": %lld,\n", (long long)result.config_index);
  std::printf("  \"workspace_fc1\": %d,\n  \"workspace_fc2\": %d,\n", result.workspace_fc1,
              result.workspace_fc2);
  std::printf("  \"max_num_padded_tokens\": %d,\n  \"max_num_ctas\": %d,\n",
              result.max_num_padded_tokens, result.max_num_ctas);
  std::printf("  \"pointers\": {");
  for (size_t i = 0; i < g_pointers.size(); ++i) {
    std::printf("%s\"%s\": %llu", i ? ", " : "", g_pointers[i].first.c_str(),
                (unsigned long long)g_pointers[i].second);
  }
  std::printf("},\n  \"events\": [");
  for (size_t i = 0; i < events.size(); ++i) std::printf("%s%s", i ? ", " : "", events[i].c_str());
  std::printf("],\n  \"launches\": [\n");
  for (size_t n = 0; n < runs[0].size(); ++n) {
    auto& l = runs[0][n];
    auto& other = runs[1][n];
    std::vector<size_t> indeterminate;
    std::vector<size_t> unset;
    if (l.api == "driver") unset = gemm_unset_bytes(l, tokens);
    for (size_t i = 0; i < l.params.size(); ++i) {
      bool never_set = std::binary_search(unset.begin(), unset.end(), i);
      if (l.params[i] != other.params[i] || never_set) indeterminate.push_back(i);
    }
    std::printf(
        "    {\"api\": \"%s\", \"kernel\": \"%s\", \"grid\": [%u, %u, %u], \"block\": [%u, %u, "
        "%u], \"shared_mem\": %u, \"attrs\": %s, \"params_size\": %zu,\n",
        l.api.c_str(), l.kernel.c_str(), l.grid[0], l.grid[1], l.grid[2], l.block[0], l.block[1],
        l.block[2], l.shared_mem, l.attrs.c_str(), l.params.size());
    std::printf("     \"tensor_maps\": [");
    for (size_t i = 0; i < l.tensor_maps.size(); ++i) {
      auto& c = l.tensor_maps[i];
      std::printf(
          "%s{\"data_type\": %d, \"rank\": %d, \"address\": %llu, \"dims\": %s, \"strides\": %s, "
          "\"box\": %s, \"element_strides\": %s, \"interleave\": %d, \"swizzle\": %d, "
          "\"l2_promotion\": %d, \"oob_fill\": %d}",
          i ? ", " : "", c.data_type, c.rank, (unsigned long long)c.address,
          list(c.dims).c_str(), list(c.strides).c_str(), list(c.box).c_str(),
          list(c.element_strides).c_str(), c.interleave, c.swizzle, c.l2_promotion, c.oob_fill);
    }
    std::printf("],\n     \"indeterminate_bytes\": %s,\n", list(indeterminate).c_str());
    std::printf("     \"params_hex\": \"%s\"}%s\n", hex(l.params.data(), l.params.size()).c_str(),
                n + 1 < runs[0].size() ? "," : "");
  }
  std::printf("  ],\n  \"gemm_configs\": {\n");
  // Static options of every launched GEMM config (looked up by function name
  // in the full KernelMetaInfo list), for the Python Params/TMA builder.
  auto const bmm = batchedGemm::batchedGemm::BatchedGemmInterface();
  auto const* configs = bmm.getBatchedGemmConfigs();
  bool first = true;
  for (auto& l : runs[0]) {
    if (l.api != "driver") continue;
    for (size_t i = 0; i < bmm.getNumBatchedGemmConfigs(); ++i) {
      auto const& c = configs[i];
      if (l.kernel != c.mFunctionName) continue;
      auto const& o = c.mOptions;
      std::printf(
          "%s    \"%s\": {\"index\": %zu, \"sha256\": \"%s\", \"shared_mem\": %d, "
          "\"threads\": %d, \"tile_m\": %d, \"tile_n\": %d, \"tile_k\": %d, "
          "\"epilogue_tile_m\": %d, \"epilogue_tile_n\": %d, \"dtype_a_bits\": %d, "
          "\"dtype_b_bits\": %d, \"dtype_c_bits\": %d, \"dtype_a\": %d, \"dtype_b\": %d, "
          "\"dtype_c\": %d, \"use_tma_store\": %d, \"route_impl\": %d, \"layout_a\": %d, "
          "\"layout_b\": %d, \"mma_kind\": %d, \"cluster\": [%d, %d, %d], "
          "\"transpose_mma_output\": %d, \"use_deepseek_fp8\": %d, \"fused_act\": %d, "
          "\"use_tma_oob_opt\": %d, \"enables_early_exit\": %d, "
          "\"enables_delayed_early_exit\": %d, \"num_slices_for_split_k\": %d, "
          "\"is_static_batch\": %d, \"tile_scheduler\": %d, \"unroll_loop_2x\": %d, "
          "\"grid_wait_for_primary\": [%d, %d, %d], \"sf_reshape_factor\": %d}",
          first ? "" : ",\n", c.mFunctionName, i, c.mHash, c.mSharedMemSize,
          c.mNumThreadsPerCTA, o.mTileM, o.mTileN, o.mTileK, o.mEpilogueTileM, o.mEpilogueTileN,
          tg::dtypeGetNumBits(o.mDtypeA), tg::dtypeGetNumBits(o.mDtypeB),
          tg::dtypeGetNumBits(o.mDtypeC), int(o.mDtypeA), int(o.mDtypeB), int(o.mDtypeC),
          int(o.mUseTmaStore), int(o.mRouteImpl), int(o.mLayoutA), int(o.mLayoutB),
          int(o.mMmaKind), o.mClusterDimX, o.mClusterDimY, o.mClusterDimZ,
          int(o.mTransposeMmaOutput), int(o.mUseDeepSeekFp8), int(o.mFusedAct),
          int(o.mUseTmaOobOpt), int(o.mEnablesEarlyExit), int(o.mEnablesDelayedEarlyExit),
          o.mNumSlicesForSplitK, int(o.mIsStaticBatch), int(o.mTileScheduler),
          int(o.mUseUnrollLoop2xForMma), int(o.mGridWaitForPrimaryEarlyExit),
          int(o.mGridWaitForPrimaryA), int(o.mGridWaitForPrimaryB), o.mSfReshapeFactor);
      first = false;
    }
  }
  std::printf("\n  }\n}\n");
  return 0;
}

// ---- layout mode ----------------------------------------------------------------

std::string g_fields;
template <class S, class M>
void field(const char* name, const S& s, const M& member, const char* kind) {
  auto offset = reinterpret_cast<const char*>(&member) - reinterpret_cast<const char*>(&s);
  g_fields += std::string(g_fields.empty() ? "" : ",\n") + "      \"" + name +
              "\": {\"offset\": " + std::to_string(offset) +
              ", \"size\": " + std::to_string(sizeof(M)) + ", \"kind\": \"" + kind + "\"}";
}
#define FIELD(s, m, kind) field(#m, s, s.m, kind)

template <class S>
void print_struct(const char* name, const S& s, bool last) {
  std::printf("    \"%s\": {\"size\": %zu, \"align\": %zu, \"fields\": {\n%s\n    }}%s\n", name,
              sizeof(S), alignof(S), g_fields.c_str(), last ? "" : ",");
  g_fields.clear();
}

int layout_mode() {
  static_assert(sizeof(rd::KernelParams<float, __nv_bfloat16, true, true>) ==
                    sizeof(rd::KernelParams<float, __nv_bfloat16, true, false>),
                "UsePdl must not change the routing parameter layout");
  std::printf("{\n  \"structs\": {\n");
  {
    rd::KernelParams<float, __nv_bfloat16, true, false> p{};
    FIELD(p, mPtrExpertCounts, "ptr");
    FIELD(p, mPtrPermutedIdxSize, "ptr");
    FIELD(p, mPtrExpandedIdxToPermutedIdx, "ptr");
    FIELD(p, mPtrPermutedIdxToTokenIdx, "ptr");
    FIELD(p, mPtrCtaIdxXyToBatchIdx, "ptr");
    FIELD(p, mPtrCtaIdxXyToMnLimit, "ptr");
    FIELD(p, mPtrNumNonExitingCtas, "ptr");
    FIELD(p, mPtrExpertWeights, "ptr");
    FIELD(p, mPtrScores, "ptr");
    FIELD(p, mNumTokens, "i32");
    FIELD(p, mNumExperts, "i32");
    FIELD(p, mPaddingLog2, "i32");
    FIELD(p, mLocalExpertsStartIdx, "i32");
    FIELD(p, mLocalExpertsStrideLog2, "i32");
    FIELD(p, mNumLocalExperts, "i32");
    FIELD(p, mPtrExpertIdx, "ptr");
    FIELD(p, mPtrRoutingBias, "ptr");
    FIELD(p, mNumExpertGroups, "i32");
    FIELD(p, mNumExpertsPerGroup, "i32");
    FIELD(p, mNumLimitedGroups, "i32");
    FIELD(p, mTopK, "int_fast_div");
    FIELD(p, mRouteScale, "f32");
    print_struct("routing", p, false);
  }
  {
    act::KernelParams<cutlass::float_e4m3_t, false> p{};
    FIELD(p, inPtr, "ptr");
    FIELD(p, outPtr, "ptr");
    FIELD(p, inDqSfsPtr, "ptr");
    FIELD(p, outDqSfsPtr, "ptr");
    FIELD(p, innerDim, "i32");
    FIELD(p, numTokens, "i32");
    FIELD(p, topK, "i32");
    FIELD(p, expandedIdxToPermutedIdx, "ptr");
    FIELD(p, totalNumPaddedTokens, "ptr");
    print_struct("activation", p, false);
  }
  {
    fin::KernelParams<cutlass::bfloat16_t, cutlass::bfloat16_t, false> p{};
    FIELD(p, inPtr, "ptr");
    FIELD(p, expertWeightsPtr, "ptr");
    FIELD(p, outPtr, "ptr");
    FIELD(p, inDqSfsPtr, "ptr");
    FIELD(p, outDqSfsPtr, "ptr");
    FIELD(p, expandedIdxToPermutedIdx, "ptr");
    FIELD(p, hiddenDim, "i32");
    FIELD(p, hiddenDimPadded, "i32");
    FIELD(p, numTokens, "i32");
    FIELD(p, numExperts, "i32");
    FIELD(p, topK, "i32");
    FIELD(p, totalNumPaddedTokens, "ptr");
    print_struct("finalize", p, false);
  }
  {
    static batchedGemm::KernelParams p{};
    FIELD(p, tmaA, "tensor_map");
    FIELD(p, tmaB, "tensor_map");
    FIELD(p, tmaC, "tensor_map");
    FIELD(p, tmaSfA, "tensor_map");
    FIELD(p, tmaSfB, "tensor_map");
    FIELD(p, ptrA, "ptr");
    FIELD(p, strideInBytesA, "u64");
    FIELD(p, ptrB, "ptr");
    FIELD(p, strideInBytesB, "u64");
    FIELD(p, ptrC, "ptr");
    FIELD(p, ptrScaleC, "ptr");
    FIELD(p, ptrScaleGate, "ptr");
    FIELD(p, ptrClampLimit, "ptr");
    FIELD(p, ptrSwiGluAlpha, "ptr");
    FIELD(p, ptrSwiGluBeta, "ptr");
    FIELD(p, k, "i32");
    FIELD(p, nm, "i32");
    FIELD(p, tileStridePerBatch, "i32");
    FIELD(p, ptrDqSfsC, "ptr");
    FIELD(p, ptrSfA, "ptr");
    FIELD(p, ptrSfB, "ptr");
    FIELD(p, ptrPerTokenSfA, "ptr");
    FIELD(p, ptrPerTokenSfB, "ptr");
    FIELD(p, ptrBias, "ptr");
    FIELD(p, ptrSfC, "ptr");
    FIELD(p, ptrRouteMap, "ptr");
    FIELD(p, numTokens, "i32");
    FIELD(p, numBatches, "i32");
    FIELD(p, ptrNumNonExitingCtas, "ptr");
    FIELD(p, ptrTotalNumPaddedTokens, "ptr");
    FIELD(p, ptrCtaIdxXyToBatchIdx, "ptr");
    FIELD(p, ptrCtaIdxXyToMnLimit, "ptr");
    FIELD(p, totalNumPaddedTokens, "i32");
    FIELD(p, ctaIdxXyToBatchIdx, "bytes");
    FIELD(p, ctaIdxXyToMnLimit, "bytes");
    FIELD(p, rank, "i32");
    FIELD(p, tpGrpSize, "i32");
    FIELD(p, ptrPartialRowMax, "ptr");
    FIELD(p, ptrRowMaxCompletionBars, "ptr");
    print_struct("gemm", p, true);
  }
  std::printf("  },\n  \"int_fast_div\": {\"size\": %zu},\n",
              sizeof(trtllm::dev::IntFastDiv));
  // The GEMM configs upstream can select (static options the Python builder
  // reads; checked against the actual launches by the compiler).
  auto const bmm = batchedGemm::batchedGemm::BatchedGemmInterface();
  auto const* configs = bmm.getBatchedGemmConfigs();
  std::printf("  \"num_gemm_configs\": %zu\n}\n", bmm.getNumBatchedGemmConfigs());
  (void)configs;
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc >= 2 && std::string(argv[1]) == "layout") return layout_mode();
  if (argc >= 2 && std::string(argv[1]) == "run") return run_mode(argc, argv);
  std::fprintf(stderr, "usage: moe_probe layout | run TOKENS LOCAL_EXPERT_OFFSET SCALE\n");
  return 1;
}
