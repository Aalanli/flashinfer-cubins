// SPDX-License-Identifier: Apache-2.0
// Shared part of the host-only trtllm-gen probes (bmm_probe.cu, gemm_probe.cu):
// JSON helpers, the recording CUDA runtime/driver interposers and the
// deterministic CUtensorMap stand-in. Included once per probe executable.
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Every scalar option the Python side reads, as integers.
#define PROBE_COMMON_OPTIONS(X) \
  X(mAllReduceAlgo) \
  X(mBiasType) \
  X(mBlockK) \
  X(mClusterDimX) \
  X(mClusterDimY) \
  X(mClusterDimZ) \
  X(mCtaSwizzleType) \
  X(mDtypeAcc) \
  X(mDtypeA) \
  X(mDtypeB) \
  X(mDtypeC) \
  X(mDtypeMmaA) \
  X(mDtypeMmaB) \
  X(mEltwiseActType) \
  X(mEnablesEarlyExit) \
  X(mEnablesDelayedEarlyExit) \
  X(mEpilogueTileM) \
  X(mEpilogueTileN) \
  X(mGridWaitForPrimaryEarlyExit) \
  X(mGridWaitForPrimaryA) \
  X(mGridWaitForPrimaryB) \
  X(mLayoutA) \
  X(mLayoutB) \
  X(mMmaK) \
  X(mMmaKind) \
  X(mMmaM) \
  X(mMmaN) \
  X(mMmaTileK) \
  X(mNumSlicesForSplitK) \
  X(mNumSlicesForSliceK) \
  X(mSfBlockSizeA) \
  X(mSfBlockSizeB) \
  X(mSfBlockSizeC) \
  X(mSfLayoutA) \
  X(mSfLayoutB) \
  X(mSfLayoutC) \
  X(mSfReshapeFactor) \
  X(mSliceK) \
  X(mSparsityA) \
  X(mSplitK) \
  X(mTileK) \
  X(mTileM) \
  X(mTileN) \
  X(mTileScheduler) \
  X(mTransposeMmaOutput) \
  X(mUseDeepSeekFp8) \
  X(mUsePerTokenSfA) \
  X(mUsePerTokenSfB) \
  X(mUseShuffledMatrix) \
  X(mUseTmaStore) \
  X(mUseUnrollLoop2xForMma) \
  X(mUseCustomizedMma3xNvFp4)

namespace {

std::string json_string(const std::string& s) {
  std::string out = "\"";
  for (char ch : s) {
    if (ch == '"' || ch == '\\') {
      out += '\\';
      out += ch;
    } else if (static_cast<unsigned char>(ch) < 0x20) {
      out += ' ';
    } else {
      out += ch;
    }
  }
  return out + "\"";
}

// Sparse parameter bytes: [offset, hex] runs of nonzero bytes (gaps of fewer
// than 16 zero bytes stay inside a run).
std::string sparse_runs(const std::vector<unsigned char>& params);

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

std::string encode_json(const EncodeCall& c) {
  return "{\"data_type\": " + std::to_string(c.data_type) +
         ", \"address\": " + std::to_string(c.address) + ", \"dims\": " + list(c.dims) +
         ", \"strides\": " + list(c.strides) + ", \"box\": " + list(c.box) +
         ", \"element_strides\": " + list(c.element_strides) +
         ", \"interleave\": " + std::to_string(c.interleave) +
         ", \"swizzle\": " + std::to_string(c.swizzle) +
         ", \"l2_promotion\": " + std::to_string(c.l2_promotion) +
         ", \"oob_fill\": " + std::to_string(c.oob_fill) + "}";
}

struct Launch {
  std::string kernel;
  unsigned grid[3], block[3];
  unsigned shared_mem;
  std::string attrs;  // JSON array
  std::vector<unsigned char> params;
  std::vector<EncodeCall> tensor_maps;
};

std::vector<Launch> g_launches;
std::vector<EncodeCall> g_pending_encodes;
std::vector<std::string> g_events;
std::vector<std::string> g_function_names;
int g_sm_count = 148;
size_t g_params_size = 0;  // sizeof the probed KernelParams
unsigned char g_fill = 0;

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

}  // namespace

// ---- interposed CUDA runtime (host side of the shared cudart) ----------------
#ifndef __CUDA_ARCH__
extern "C" cudaError_t CUDARTAPI cudaDeviceGetAttribute(int* value, enum cudaDeviceAttr attr,
                                                        int device) {
  poison_stack(g_fill);
  g_events.push_back("{\"call\": \"cudaDeviceGetAttribute\", \"attribute\": " +
                     std::to_string(int(attr)) + ", \"device\": " + std::to_string(device) + "}");
  if (attr != cudaDevAttrMultiProcessorCount) return cudaErrorInvalidValue;
  *value = g_sm_count;
  return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaGetDevice(int* device) {
  *device = 0;
  return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaMemsetAsync(void* ptr, int value, size_t count,
                                                 cudaStream_t) {
  g_events.push_back("{\"call\": \"cudaMemsetAsync\", \"ptr\": " +
                     std::to_string(reinterpret_cast<uint64_t>(ptr)) + ", \"value\": " +
                     std::to_string(value) + ", \"count\": " + std::to_string(count) + "}");
  return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaMemcpyAsync(void*, const void*, size_t count,
                                                 cudaMemcpyKind, cudaStream_t) {
  g_events.push_back("{\"call\": \"cudaMemcpyAsync\", \"count\": " + std::to_string(count) + "}");
  return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaMallocHost(void** ptr, size_t size) {
  *ptr = std::malloc(size);
  g_events.push_back("{\"call\": \"cudaMallocHost\", \"size\": " + std::to_string(size) + "}");
  return cudaSuccess;
}

#endif  // __CUDA_ARCH__

// ---- interposed CUDA driver ------------------------------------------------------
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
  l.params.assign(bytes, bytes + g_params_size);
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
  *str = "probe";
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuGetErrorName(CUresult, const char** str) {
  *str = "probe";
  return CUDA_SUCCESS;
}


namespace {

std::string sparse_runs(const std::vector<unsigned char>& params) {
  std::string runs = "[";
  size_t b = 0;
  bool first = true;
  while (b < params.size()) {
    if (params[b] == 0) {
      ++b;
      continue;
    }
    size_t end = b, zeros = 0;
    for (size_t e = b; e < params.size(); ++e) {
      if (params[e] == 0) {
        if (++zeros >= 16) break;
      } else {
        zeros = 0;
        end = e + 1;
      }
    }
    runs += std::string(first ? "" : ", ") + "[" + std::to_string(b) + ", \"" +
            hex(params.data() + b, end - b) + "\"]";
    first = false;
    b = end;
  }
  return runs + "]";
}

std::string ranges_json(const std::vector<size_t>& indices) {
  std::vector<std::pair<size_t, size_t>> ranges;
  for (size_t u : indices) {
    if (!ranges.empty() && ranges.back().second == u) {
      ranges.back().second = u + 1;
    } else {
      ranges.push_back({u, u + 1});
    }
  }
  std::string out = "[";
  for (size_t r = 0; r < ranges.size(); ++r) {
    out += (r ? ", [" : "[") + std::to_string(ranges[r].first) + ", " +
           std::to_string(ranges[r].second) + "]";
  }
  return out + "]";
}

// One recorded launch as JSON; `undefined` lists byte offsets never compared
// (their recorded value is zeroed).
std::string launch_json(const Launch& l, const std::vector<size_t>& undefined) {
  std::vector<unsigned char> params = l.params;
  for (size_t u : undefined) params[u] = 0;
  std::string out = "{\"kernel\": " + json_string(l.kernel) + ", \"grid\": [" +
                    std::to_string(l.grid[0]) + ", " + std::to_string(l.grid[1]) + ", " +
                    std::to_string(l.grid[2]) + "], \"block\": [" + std::to_string(l.block[0]) +
                    ", " + std::to_string(l.block[1]) + ", " + std::to_string(l.block[2]) +
                    "], \"shared_mem\": " + std::to_string(l.shared_mem) +
                    ", \"attrs\": " + l.attrs +
                    ", \"params_size\": " + std::to_string(l.params.size()) +
                    ", \"params\": " + sparse_runs(params) +
                    ", \"indeterminate\": " + ranges_json(undefined) + ", \"tensor_maps\": [";
  for (size_t t = 0; t < l.tensor_maps.size(); ++t) {
    out += (t ? ", " : "") + encode_json(l.tensor_maps[t]);
  }
  return out + "]}";
}

}  // namespace
