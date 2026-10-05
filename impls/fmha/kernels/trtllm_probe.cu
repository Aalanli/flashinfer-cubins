// Host-only dispatch/parameter probe for the trtllm-gen FMHA kernels (compile stage only).
//
// Runs the unmodified upstream host code of FlashInfer v0.6.9
// (include/flashinfer/trtllm/fmha/fmhaKernels.cuh: TllmGenFmhaKernel::run, kernel selection,
// computeCtaAndClusterConfig, KernelParams::setKernelParams) over the trtllm-gen meta-info table
// (include/flashInferMetaInfo.h of artifact 55bba559.../fmha/trtllm-gen) with the CUDA driver API
// interposed: nothing is loaded or launched, no GPU is needed. For every case read from stdin it
// prints one JSON line with the kernel upstream selects, its launch configuration and the raw
// KernelParams bytes, in which every CUtensorMap holds a deterministic packing of the
// cuTensorMapEncodeTiled arguments (mirrored by harness.workloads.fmha.trtllm.fake_encode).
//
// Input: one case per line, whitespace-separated key=value pairs (see parse()); pointers are
// fake addresses 0x100000000000 * (role index + 1) for the roles listed in "ptrs"; "emit=0"
// leaves the KernelParams bytes out of that case's line (kernel selection sweeps).
// The first output line describes the KernelParams layout (sizeof and field offsets).
#include <cuda.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "flashinfer/trtllm/fmha/fmhaKernels.cuh"

namespace {

struct Recorder {
  std::string function;
  bool launched = false;
  unsigned grid[3]{}, block[3]{}, cluster[3]{1, 1, 1};
  unsigned smem = 0;
  int pdl = -1, policy = -1, nonPortable = 0, reductions = 0;
  long long maxDynSmem = -1;
  std::vector<unsigned char> params;
};
Recorder g_rec;
// cuOccupancyMaxActiveClusters: a fixed count, or (< 0) the count measured on a 148-SM B200
// for cluster sizes 1..16 at one CTA per SM, which every CgaSmemReduction kernel has (so the
// selection falls back to GmemReduction exactly where it does on that GPU).
int g_maxActiveClusters = -1;
constexpr int kB200Clusters[16] = {148, 74, 45, 33, 26, 22, 15, 15, 15, 11, 7, 7, 7, 7, 7, 7};

const char* kRoles[] = {"q",          "k",        "v",          "kv",         "qkv",
                        "kSf",        "vSf",      "customMask", "customMaskOffsets",
                        "firstSparse", "counter",  "seqLensKv",  "cumSeqLensQ", "cumSeqLensKv",
                        "pageIdx",    "outputScale", "scaleSoftmaxLog2", "kvSfScale",
                        "oSfScale",   "scratch",  "softmaxStats", "lse",       "sinks",
                        "o",          "oSf"};
constexpr int kNumRoles = sizeof(kRoles) / sizeof(kRoles[0]);

void* sentinel(int index) {
  return reinterpret_cast<void*>(uintptr_t(0x100000000000ull) * uintptr_t(index + 1));
}

std::string hex(const void* data, size_t size) {
  static const char digits[] = "0123456789abcdef";
  std::string out;
  auto* bytes = static_cast<const unsigned char*>(data);
  for (size_t i = 0; i < size; ++i) {
    out += digits[bytes[i] >> 4];
    out += digits[bytes[i] & 15];
  }
  return out;
}

}  // namespace

// ---- interposed CUDA driver API (the probe links no libcuda) ------------------------------------
extern "C" {
CUresult CUDAAPI cuModuleLoadData(CUmodule* module, const void*) {
  static int dummy;
  *module = reinterpret_cast<CUmodule>(&dummy);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuModuleGetFunction(CUfunction* hfunc, CUmodule, const char* name) {
  *hfunc = reinterpret_cast<CUfunction>(strdup(name));
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuFuncSetAttribute(CUfunction, CUfunction_attribute attrib, int value) {
  if (attrib == CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES) g_rec.maxDynSmem = value;
  if (attrib == CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED) g_rec.nonPortable = value;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuOccupancyMaxActiveClusters(int* numClusters, CUfunction,
                                              const CUlaunchConfig* config) {
  unsigned clusterX = 1;
  for (unsigned i = 0; i < config->numAttrs; ++i)
    if (config->attrs[i].id == CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION)
      clusterX = config->attrs[i].value.clusterDim.x;
  if (g_maxActiveClusters >= 0)
    *numClusters = g_maxActiveClusters;
  else if (clusterX >= 1 && clusterX <= 16)
    *numClusters = kB200Clusters[clusterX - 1];
  else
    return CUDA_ERROR_INVALID_CLUSTER_SIZE;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuLaunchKernelEx(const CUlaunchConfig* config, CUfunction f, void** kernelParams,
                                  void**) {
  g_rec.launched = true;
  g_rec.function = reinterpret_cast<const char*>(f);
  g_rec.grid[0] = config->gridDimX, g_rec.grid[1] = config->gridDimY,
  g_rec.grid[2] = config->gridDimZ;
  g_rec.block[0] = config->blockDimX, g_rec.block[1] = config->blockDimY,
  g_rec.block[2] = config->blockDimZ;
  g_rec.smem = config->sharedMemBytes;
  for (unsigned i = 0; i < config->numAttrs; ++i) {
    auto const& a = config->attrs[i];
    if (a.id == CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION) {
      g_rec.cluster[0] = a.value.clusterDim.x, g_rec.cluster[1] = a.value.clusterDim.y,
      g_rec.cluster[2] = a.value.clusterDim.z;
    } else if (a.id == CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION) {
      g_rec.pdl = a.value.programmaticStreamSerializationAllowed;
    } else if (a.id == CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE) {
      g_rec.policy = a.value.clusterSchedulingPolicyPreference;
    }
  }
  auto* bytes = static_cast<unsigned char*>(kernelParams[0]);
  g_rec.params.assign(bytes, bytes + sizeof(KernelParams));
  return CUDA_SUCCESS;
}
// Deterministic stand-in: packs the arguments into the 128-byte descriptor.
CUresult CUDAAPI cuTensorMapEncodeTiled(CUtensorMap* map, CUtensorMapDataType dataType,
                                        cuuint32_t rank, void* address, const cuuint64_t* dims,
                                        const cuuint64_t* strides, const cuuint32_t* box,
                                        const cuuint32_t* elementStrides,
                                        CUtensorMapInterleave interleave,
                                        CUtensorMapSwizzle swizzle,
                                        CUtensorMapL2promotion l2Promotion,
                                        CUtensorMapFloatOOBfill oobFill) {
  unsigned char* out = reinterpret_cast<unsigned char*>(map);
  std::memset(out, 0, 128);
  out[0] = static_cast<unsigned char>(dataType);
  out[1] = static_cast<unsigned char>(rank);
  out[2] = static_cast<unsigned char>(interleave);
  out[3] = static_cast<unsigned char>(swizzle);
  out[4] = static_cast<unsigned char>(l2Promotion);
  out[5] = static_cast<unsigned char>(oobFill);
  uint64_t addr = reinterpret_cast<uint64_t>(address);
  std::memcpy(out + 8, &addr, 8);
  std::memcpy(out + 16, dims, 8 * rank);
  std::memcpy(out + 56, strides, 8 * (rank - 1));
  std::memcpy(out + 88, box, 4 * rank);
  std::memcpy(out + 108, elementStrides, 4 * rank);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuGetErrorString(CUresult, const char** str) {
  *str = "probe";
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuGetErrorName(CUresult, const char** str) {
  *str = "probe";
  return CUDA_SUCCESS;
}
}  // extern "C"

namespace flashinfer::trtllm_cubin_loader {
std::string getCubin(const std::string&, const std::string&) { return "probe"; }
}  // namespace flashinfer::trtllm_cubin_loader

namespace tensorrt_llm::kernels {
// The separate reduction kernel (GmemReductionWithSeparateKernel) is not a single-kernel
// workload; the probe records that upstream would launch it.
void runFmhaReduction(TllmGenFmhaKernelMetaInfo const& meta, KernelParams const&, int32_t, bool,
                      cudaStream_t) {
  if (static_cast<MultiCtasKvMode>(meta.mMultiCtasKvMode) ==
      MultiCtasKvMode::GmemReductionWithSeparateKernel) {
    ++g_rec.reductions;
  }
}
}  // namespace tensorrt_llm::kernels

namespace {

using tensorrt_llm::kernels::sTllmGenFmhaKernelMetaInfos;

// ---- KernelParams layout -----------------------------------------------------------------------
template <class M>
void field(std::ostringstream& os, bool& first, const char* name, const char* kind,
           KernelParams const& p, M const& member) {
  os << (first ? "" : ", ") << "\"" << name << "\": [" << (reinterpret_cast<const char*>(&member) -
                                                          reinterpret_cast<const char*>(&p))
     << ", " << sizeof(M) << ", \"" << kind << "\"]";
  first = false;
}

void printLayout() {
  KernelParams p;
  std::ostringstream os;
  bool first = true;
#define F(name, kind) field(os, first, #name, kind, p, p.name)
  F(tmaQ_, "tma");
  F(tmaK_, "tma");
  F(tmaV_, "tma");
  F(tmaO_, "tma");
  F(tmaKSf_, "tma");
  F(tmaVSf_, "tma");
  F(logicalGridDimX, "i32");
  F(logicalGridDimY, "i32");
  F(logicalGridDimZ, "i32");
  F(ptrO, "ptr");
  F(ptrSfO, "ptr");
  F(ptrAttentionSinks, "ptr");
  F(ptrCumSeqLensQ, "ptr");
  F(ptrCumSeqLensKv, "ptr");
  F(ptrCustomMask, "ptr");
  F(ptrCustomMaskOffsets, "ptr");
  F(ptrDebugO, "ptr");
  F(ptrFirstSparseMaskOffsetsKv, "ptr");
  F(ptrMultiCtasKvCounter, "ptr");
  F(ptrOutputScale, "ptr");
  F(ptrPageIdxKv, "ptr");
  F(ptrPartialO, "ptr");
  F(ptrPartialStats, "ptr");
  F(ptrSageAttnSfsK, "ptr");
  F(ptrSageAttnSfsP, "ptr");
  F(ptrSageAttnSfsQ, "ptr");
  F(ptrSageAttnSfsV, "ptr");
  F(ptrScaleSoftmaxLog2, "ptr");
  F(ptrScaleSfKv, "ptr");
  F(ptrScaleSfO, "ptr");
  F(ptrSeqLensKv, "ptr");
  F(ptrSkipSoftmaxStats, "ptr");
  F(ptrSoftmaxStats, "ptr");
  F(mAttentionWindowSize, "i32");
  F(mBatchSize, "i32");
  F(mChunkedAttentionSizeLog2, "i32");
  F(mInflateMax, "f32");
  F(mLogNumEltsPerSageAttnBlkK, "i32");
  F(mLogNumEltsPerSageAttnBlkP, "i32");
  F(mLogNumEltsPerSageAttnBlkQ, "i32");
  F(mLogNumEltsPerSageAttnBlkV, "i32");
  F(mMaxSeqLenQ, "i32");
  F(mMaxSeqLenKv, "i32");
  F(mMaxNumCtasQ, "i32");
  F(mMaxNumCtasKv, "i32");
  F(mMaxNumPagesPerSeqKv, "i32");
  F(mNumHeadsKv, "i32");
  F(mNumHeadsQ, "i32");
  F(mNumHeadsQPerKv, "i32");
  F(mNumHeadsQPerKvDivisor, "fastmoddiv");
  F(mNumHiddenEltsO, "i64");
  F(mNumPagesInMemPool, "i32");
  F(mNumTokensPerCtaQ, "i32");
  F(mNumTokensPerPageLog2, "i32");
  F(mOutputScale, "f32");
  F(mScaleSoftmaxLog2, "f32");
  F(mScaleSfKv, "f32");
  F(mScaleSfO, "f32");
  F(mSkipSoftmaxThresholdScaleFactor, "f32");
  F(mStartTokenIdxSfO, "i32");
  F(mSumOfSeqLensQ, "i32");
  F(mSumOfSeqLensKv, "i32");
  F(mSparseMlaTopK, "i32");
  F(mUseBlockSparseAttention, "bool");
  F(mUsesSharedPagedKvIdx, "bool");
#undef F
  std::printf("{\"layout\": {\"size\": %zu, \"align\": %zu, \"fields\": {%s}}}\n",
              sizeof(KernelParams), alignof(KernelParams), os.str().c_str());
}

// ---- cases -------------------------------------------------------------------------------------
Data_type dtype(std::string const& s) {
  if (s == "fp16") return DATA_TYPE_FP16;
  if (s == "bf16") return DATA_TYPE_BF16;
  if (s == "e4m3") return DATA_TYPE_E4M3;
  if (s == "e2m1") return DATA_TYPE_E2M1;
  throw std::runtime_error("unknown dtype " + s);
}

struct Case {
  std::string id;
  Data_type q = DATA_TYPE_UNKNOWN, kv = DATA_TYPE_UNKNOWN, o = DATA_TYPE_UNKNOWN;
  bool emit = true;
  TllmGenFmhaRunnerParams p;
};

Case parse(std::string const& line) {
  Case c;
  auto& p = c.p;
  std::istringstream in(line);
  std::string token;
  while (in >> token) {
    auto eq = token.find('=');
    if (eq == std::string::npos) throw std::runtime_error("bad token " + token);
    std::string k = token.substr(0, eq), v = token.substr(eq + 1);
    auto i = [&] { return static_cast<int>(std::stoll(v)); };
    auto f = [&] { return std::strtof(v.c_str(), nullptr); };
    if (k == "id") c.id = v;
    else if (k == "dtq") c.q = dtype(v);
    else if (k == "dtkv") c.kv = dtype(v);
    else if (k == "dto") c.o = dtype(v);
    else if (k == "layout") p.mQkvLayout = static_cast<QkvLayout>(i());
    else if (k == "mask") p.mMaskType = static_cast<TrtllmGenAttentionMaskType>(i());
    else if (k == "ktype") p.mKernelType = static_cast<FmhaKernelType>(i());
    else if (k == "sched") p.mTileScheduler = static_cast<TileScheduler>(i());
    else if (k == "mcta") p.mMultiCtasKvMode = i() != 0;
    else if (k == "hdqk") p.mHeadDimQk = i();
    else if (k == "hdv") p.mHeadDimV = i();
    else if (k == "hq") p.mNumHeadsQ = i();
    else if (k == "hkv") p.mNumHeadsKv = i();
    else if (k == "batch") p.mBatchSize = i();
    else if (k == "max_q") p.mMaxSeqLenQ = i();
    else if (k == "max_kv") p.mMaxSeqLenKv = i();
    else if (k == "max_cache_kv") p.mMaxSeqLenCacheKv = i();
    else if (k == "sum_q") p.mSumOfSeqLensQ = i();
    else if (k == "sum_kv") p.mSumOfSeqLensKv = i();
    else if (k == "window") p.mAttentionWindowSize = i();
    else if (k == "chunk") p.mChunkedAttentionSize = i();
    else if (k == "tpp") p.mNumTokensPerPage = i();
    else if (k == "max_pages") p.mMaxNumPagesPerSeqKv = i();
    else if (k == "pool") p.mNumPagesInMemPool = i();
    else if (k == "sms") p.mMultiProcessorCount = i();
    else if (k == "skips") p.mSkipsSoftmaxWhenPossible = i() != 0;
    else if (k == "skip_thr") p.mSkipSoftmaxThresholdScaleFactor = f();
    else if (k == "sparse_topk") p.mSparseMla = (p.mSparseMlaTopK = i()) > 0;
    else if (k == "shared_idx") p.mUsesSharedPagedKvIdx = i() != 0;
    else if (k == "q_st") p.qStrideTokens = i();
    else if (k == "q_sh") p.qStrideHeads = i();
    else if (k == "k_skv") p.kStrideKeysValues = i();
    else if (k == "k_sh") p.kStrideHeads = i();
    else if (k == "k_sb") p.kStrideBatch = i();
    else if (k == "v_skv") p.vStrideKeysValues = i();
    else if (k == "v_sh") p.vStrideHeads = i();
    else if (k == "v_sb") p.vStrideBatch = i();
    else if (k == "ksf_sh") p.kSfStrideHeads = i();
    else if (k == "ksf_sb") p.kSfStrideBatch = i();
    else if (k == "vsf_sh") p.vSfStrideHeads = i();
    else if (k == "vsf_sb") p.vSfStrideBatch = i();
    else if (k == "scale_log2") p.scaleSoftmaxLog2 = f();
    else if (k == "out_scale") p.outputScale = f();
    else if (k == "sf_scale_kv") p.mScaleSfKv = f();
    else if (k == "sf_scale_o") p.mScaleSfO = f();
    else if (k == "sf_start") p.mSfStartTokenIdx = i();
    else if (k == "pdl") p.enable_pdl = i() != 0;
    else if (k == "occupancy") g_maxActiveClusters = i();
    else if (k == "emit") c.emit = i() != 0;
    else if (k == "ptrs") {
      std::istringstream roles(v);
      std::string role;
      while (std::getline(roles, role, ',')) {
        int idx = -1;
        for (int r = 0; r < kNumRoles; ++r)
          if (role == kRoles[r]) idx = r;
        if (idx < 0) throw std::runtime_error("unknown pointer role " + role);
        void* s = sentinel(idx);
        switch (idx) {
          case 0: p.qPtr = s; break;
          case 1: p.kPtr = s; break;
          case 2: p.vPtr = s; break;
          case 3: p.kvPtr = s; break;
          case 4: p.qkvPtr = s; break;
          case 5: p.kSfBasePtr = s; break;
          case 6: p.vSfBasePtr = s; break;
          case 7: p.customMaskPtr = static_cast<uint32_t const*>(s); break;
          case 8: p.customMaskOffsetsPtr = static_cast<int64_t const*>(s); break;
          case 9: p.firstSparseMaskOffsetsKvPtr = static_cast<int32_t const*>(s); break;
          case 10: p.multiCtasKvCounterPtr = static_cast<int32_t*>(s); break;
          case 11: p.seqLensKvPtr = static_cast<int const*>(s); break;
          case 12: p.cumSeqLensQPtr = static_cast<int const*>(s); break;
          case 13: p.cumSeqLensKvPtr = static_cast<int const*>(s); break;
          case 14: p.kvPageIdxPtr = static_cast<int const*>(s); break;
          case 15: p.outputScalePtr = static_cast<float const*>(s); break;
          case 16: p.scaleSoftmaxLog2Ptr = static_cast<float const*>(s); break;
          case 17: p.kvSfScalePtr = static_cast<float const*>(s); break;
          case 18: p.oSfScalePtr = static_cast<float const*>(s); break;
          case 19: p.multiCtasKvScratchPtr = s; break;
          case 20: p.softmaxStatsPtr = static_cast<float2*>(s); break;
          case 21: p.lsePtr = static_cast<float*>(s); break;
          case 22: p.ptrAttentionSinks = static_cast<float const*>(s); break;
          case 23: p.oPtr = s; break;
          case 24: p.oSfPtr = s; break;
        }
      }
    } else {
      throw std::runtime_error("unknown key " + k);
    }
  }
  p.mNumHeadsQPerKv = p.mNumHeadsKv ? p.mNumHeadsQ / p.mNumHeadsKv : 0;
  return c;
}

}  // namespace

int main() {
  printLayout();
  std::map<std::tuple<int, int, int>, std::unique_ptr<TllmGenFmhaKernel>> kernels;
  unsigned const count = sizeof(sTllmGenFmhaKernelMetaInfos) / sizeof(sTllmGenFmhaKernelMetaInfos[0]);
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;
    std::string id;
    try {
      Case c = parse(line);
      id = c.id;
      auto key = std::make_tuple(int(c.q), int(c.kv), int(c.o));
      auto& kernel = kernels[key];
      if (!kernel) {
        // B200 (sm_100): loadKernels keeps the kSM_100 and kSM_100f entries.
        kernel = std::make_unique<TllmGenFmhaKernel>(sTllmGenFmhaKernelMetaInfos, count, c.q,
                                                     c.kv, c.o, kSM_100);
        kernel->loadKernels();
      }
      g_rec = Recorder{};
      kernel->run(c.p);
      if (!g_rec.launched) throw std::runtime_error("no launch");
      std::printf(
          "{\"id\": \"%s\", \"kernel\": \"%s\", \"grid\": [%u, %u, %u], \"block\": [%u, %u, %u], "
          "\"cluster\": [%u, %u, %u], \"smem\": %u, \"max_dyn_smem\": %lld, \"pdl\": %d, "
          "\"policy\": %d, \"non_portable\": %d, \"reductions\": %d, \"params\": \"%s\"}\n",
          id.c_str(), g_rec.function.c_str(), g_rec.grid[0], g_rec.grid[1], g_rec.grid[2],
          g_rec.block[0], g_rec.block[1], g_rec.block[2], g_rec.cluster[0], g_rec.cluster[1],
          g_rec.cluster[2], g_rec.smem, g_rec.maxDynSmem, g_rec.pdl, g_rec.policy,
          g_rec.nonPortable, g_rec.reductions,
          c.emit ? hex(g_rec.params.data(), g_rec.params.size()).c_str() : "");
    } catch (std::exception const& e) {
      std::string what = e.what();
      for (auto& ch : what)
        if (ch == '"' || ch == '\\' || ch == '\n') ch = ' ';
      std::printf("{\"id\": \"%s\", \"error\": \"%s\"}\n", id.c_str(), what.c_str());
    }
    std::fflush(stdout);
  }
  return 0;
}
