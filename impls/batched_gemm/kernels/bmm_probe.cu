// SPDX-License-Identifier: Apache-2.0
// Compile-time, host-only probe of FlashInfer v0.6.9's trtllm-gen batched-GEMM
// host code as its fused MoE drives it: PermuteGemm1::Runner / Gemm2::Runner /
// MoE::Runner (csrc/trtllm_fused_moe_runner.cu) on top of
// TrtllmGenBatchedGemmRunner (csrc/trtllm_batched_gemm_runner.cu) and the
// pinned trtllmGen_bmm_export headers with their 1,405-config
// flashinferMetaInfo.h, plus the launchers' tile selection
// (computeSelectedTileN and the supported tile ladders of
// csrc/trtllm_fused_moe_kernel_launcher.cu, copied verbatim in
// bmm_moe_probe.cu, the probe's second translation unit; see moe_bridge.h).
//
// Built and run by impls/batched_gemm/compiler.py; never part of a benchmark
// run. It needs neither a GPU nor libcuda: every CUDA runtime/driver entry
// point the upstream host code reaches is interposed and only recorded.
//
//   bmm_probe layout OUT
//       sizeof/offsetof of every KernelParams field and the config count.
//   bmm_probe configs OUT
//       Per config: function name, shared memory, threads, sha256 and the
//       options the TrtllmGenBatchedGemmRunner constructor filter sees.
//   bmm_probe moe OUT < CASES
//       One MoE problem per input line (see read_case). Per line one JSON
//       line: the launcher that serves the config's dtypes, its tile ladder
//       and computeSelectedTileN's candidate tiles for the problem; whether
//       the PermuteGemm1 (routed config) / Gemm2 (unrouted) runner lists the
//       config as passing and valid (isValidConfigIndex), whether
//       MoE::Runner::getValidConfigIndices pairs it with a valid FC1/FC2
//       config (what FlashInfer's getValidConfigs offers the autotuner) and
//       whether it is the default (tactic -1) pair; the runner-level m, n,
//       k and maxNumCtasInBatchDim the MoE runner passes; with RECORD = 1 also
//       the launch PermuteGemm1/Gemm2::Runner::run makes with fake device
//       addresses (grid, block, dynamic smem, launch attributes, KernelParams
//       bytes, the bytes upstream leaves indeterminate, cuTensorMapEncodeTiled
//       calls and other recorded calls).
//
// Output goes to OUT: upstream's option checks print their reasons to stdout.
//
// Indeterminate bytes: KernelParams members without default initializers that
// setKernelParams does not set (unused descriptors, the static-batch CTA
// tables of a dynamic-batch launch) are indeterminate. Every case is run twice
// with the stack poisoned by 0x00 and then 0xff in the interposed
// cudaDeviceGetAttribute, which TrtllmGenBatchedGemmRunner::run calls right
// before BatchedGemmInterface::run builds KernelParams below it; bytes that
// differ are reported. The probe also rebuilds KernelParams directly from the
// runner-level problem it reports (m, n, k, maxNumCtasInBatchDim) and aborts
// unless that rebuild reproduces the launched bytes: the reported dimension
// mapping is the one the MoE runner used. Struct padding is never read.
//
// Tensor maps: cuTensorMapEncodeTiled is interposed and fills each 128-byte
// descriptor with a deterministic packing of its arguments (fake_tensor_map;
// harness/workloads/batched_gemm/trtllm.py has the identical Python packing).
#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>

#include "csrc/trtllm_batched_gemm_runner.cu"

#include "moe_bridge.h"
#include "probe_common.cuh"

// ---- other upstream link dependencies -------------------------------------------
namespace flashinfer::trtllm_cubin_loader {
// loadCubinData fetches the cubin by name; the module load is interposed.
std::string getCubin(const std::string&, const std::string&) { return std::string(); }
}  // namespace flashinfer::trtllm_cubin_loader

namespace {

namespace bg = batchedGemm::batchedGemm;
namespace gm = batchedGemm::gemm;
namespace tg = batchedGemm::trtllm::gen;
namespace tk = tensorrt_llm::kernels;

// Fake device address of each buffer: 1 GiB aligned, distinct, fixed by name
// (harness/workloads/batched_gemm/trtllm.py BMM_PROBE_BUFFERS).
const char* const kBuffers[] = {
    "a",        "sf_a",          "b",           "sf_b",          "c",
    "sf_c",     "scale_c",       "scale_gate",  "bias",          "alpha",
    "beta",     "clamp_limit",   "route_map",   "total_num_padded_tokens",
    "cta_idx_xy_to_batch_idx",   "cta_idx_xy_to_mn_limit",      "num_non_exiting_ctas",
    "per_token_sf_a",            "per_token_sf_b",              "workspace"};

uint64_t fake_address(const std::string& name) {
  for (size_t i = 0; i < sizeof(kBuffers) / sizeof(kBuffers[0]); ++i) {
    if (name == kBuffers[i]) return 0x7F0000000000ull + (uint64_t(i + 1) << 30);
  }
  std::fprintf(stderr, "bmm_probe: unknown buffer %s\n", name.c_str());
  std::abort();
}

tk::TrtllmGenBatchedGemmRunnerOptions runner_options(const bg::BatchedGemmOptions& o) {
  tk::TrtllmGenBatchedGemmRunnerOptions r{};
  r.dtypeA = o.mDtypeA;
  r.dtypeB = o.mDtypeB;
  r.dtypeC = o.mDtypeC;
  r.actType = static_cast<tk::ActType>(o.mActType);
  r.eltwiseActType = static_cast<tk::EltwiseActType>(o.mEltwiseActType);
  r.deepSeekFp8 = o.mUseDeepSeekFp8;
  r.fusedAct = o.mFusedAct;
  r.routeAct = !bg::doesRouteImplUseNoRoute(o.mRouteImpl);
  r.staticBatch = o.mIsStaticBatch;
  r.transposeMmaOutput = o.mTransposeMmaOutput;
  r.tileSize = o.mTransposeMmaOutput ? o.mTileN : o.mTileM;
  r.epilogueTileM = o.mEpilogueTileM;
  r.useShuffledMatrix = o.mUseShuffledMatrix;
  r.weightLayout = o.mLayoutA;
  return r;
}

void write_layout(std::ostream& out) {
  using KP = batchedGemm::KernelParams;
  out << "{\"size\": " << sizeof(KP) << ", \"max_num_ctas\": " << KP::MaxNumCtas
      << ", \"num_configs\": " << bg::BatchedGemmInterface().getNumBatchedGemmConfigs()
      << ", \"fields\": {";
  bool first = true;
  auto field = [&](const char* name, size_t offset, size_t size, const char* kind) {
    out << (first ? "" : ", ") << "\"" << name << "\": {\"offset\": " << offset
        << ", \"size\": " << size << ", \"kind\": \"" << kind << "\"}";
    first = false;
  };
#define FIELD(name, kind) field(#name, offsetof(KP, name), sizeof(KP::name), kind)
  FIELD(tmaA, "tensor_map");
  FIELD(tmaB, "tensor_map");
  FIELD(tmaC, "tensor_map");
  FIELD(tmaSfA, "tensor_map");
  FIELD(tmaSfB, "tensor_map");
  FIELD(tmaSparsityInfoA, "tensor_map");
  FIELD(ptrA, "ptr");
  FIELD(strideInBytesA, "u64");
  FIELD(ptrB, "ptr");
  FIELD(strideInBytesB, "u64");
  FIELD(ptrC, "ptr");
  FIELD(ptrScaleC, "ptr");
  FIELD(ptrScaleAct, "ptr");
  FIELD(ptrScaleGate, "ptr");
  FIELD(ptrClampLimit, "ptr");
  FIELD(ptrGatedActAlpha, "ptr");
  FIELD(ptrGatedActBeta, "ptr");
  FIELD(k, "i32");
  FIELD(nm, "i32");
  FIELD(tileStridePerBatch, "i32");
  FIELD(ptrDqSfsC, "ptr");
  FIELD(ptrSfA, "ptr");
  FIELD(ptrSfB, "ptr");
  FIELD(ptrPerTokenSfA, "ptr");
  FIELD(ptrPerTokenSfB, "ptr");
  FIELD(ptrBias, "ptr");
  FIELD(ptrSfC, "ptr");
  FIELD(ptrRouteMap, "ptr");
  FIELD(numTokens, "i32");
  FIELD(numBatches, "i32");
  FIELD(ptrNumNonExitingCtas, "ptr");
  FIELD(ptrTotalNumPaddedTokens, "ptr");
  FIELD(ptrCtaIdxXyToBatchIdx, "ptr");
  FIELD(ptrCtaIdxXyToMnLimit, "ptr");
  FIELD(totalNumPaddedTokens, "i32");
  FIELD(totalNumOutputPaddedTokens, "i32");
  FIELD(ctaIdxXyToBatchIdx, "i32_array");
  FIELD(ctaIdxXyToMnLimit, "i32_array");
  FIELD(ctasInTokenDimPerBatch, "i32");
  FIELD(batchStrideInCtas, "i32");
  FIELD(rank, "i32");
  FIELD(tpGrpSize, "i32");
  FIELD(ptrPartialRowMax, "ptr");
  FIELD(ptrRowMaxCompletionBars, "ptr");
  FIELD(ptrDynamicTileCounter, "ptr");
#undef FIELD
  out << "}}\n";
}

// Every scalar option the Python side reads, as integers.
#define BMM_OPTIONS(X) \
  X(mBiasDtype) \
  X(mActType) \
  X(mClampBeforeAct) \
  X(mBatchMode) \
  X(mBatchStrideInTokens) \
  X(mFusedAct) \
  X(mGridWaitForPrimaryRouting) \
  X(mIsStaticBatch) \
  X(mIsUniformNumTokensPerBatch) \
  X(mNumBatches) \
  X(mNumStagesA) \
  X(mNumStagesB) \
  X(mNumStagesMma) \
  X(mNumTokens) \
  X(mRouteImpl) \
  X(mUseTmaOobOpt)

template <class Options>
std::string options_json(Options const& o) {
  std::string out = "{";
  bool first = true;
  auto add = [&](const char* name, long long value) {
    out += std::string(first ? "" : ", ") + "\"" + name + "\": " + std::to_string(value);
    first = false;
  };
#define X(f) add(#f, static_cast<long long>(o.f));
  PROBE_COMMON_OPTIONS(X)
  BMM_OPTIONS(X)
#undef X
  add("mRouteSfsImpl", o.mRouteSfsImpl.has_value() ? static_cast<long long>(*o.mRouteSfsImpl) : -1);
  return out + "}";
}

void write_configs(std::ostream& out) {
  auto const bmm = bg::BatchedGemmInterface();
  auto const* configs = bmm.getBatchedGemmConfigs();
  size_t const n = bmm.getNumBatchedGemmConfigs();
  for (size_t i = 0; i < n; ++i) {
    auto const& c = configs[i];
    auto const& o = c.mOptions;
    // Reachability: constructing the runner with the options derived from the
    // config must list it among the passing configs.
    bool passing = false;
    std::string error;
    try {
      tk::TrtllmGenBatchedGemmRunner runner(runner_options(o));
      auto const idx = runner.getPassingConfigIndices();
      passing = std::find(idx.begin(), idx.end(), int64_t(i)) != idx.end();
    } catch (std::exception const& e) {
      error = e.what();
    }
    out << "{\"index\": " << i << ", \"function\": \"" << c.mFunctionName
        << "\", \"shared_mem\": " << c.mSharedMemSize << ", \"threads\": " << c.mNumThreadsPerCTA
        << ", \"sha256\": \"" << c.mHash << "\", \"sm\": " << int(c.mSm)
        << ", \"runner_passing\": " << (passing ? "true" : "false")
        << ", \"options\": " << options_json(o) << "}\n";
    if (!error.empty()) std::fprintf(stderr, "config %zu: %s\n", i, error.c_str());
  }
}

// ---- MoE problems --------------------------------------------------------------------

struct Case {
  std::string id;
  std::string act;  // "-": the routed config's own; FC2: the paired FC1 activation
  int config, tokens, top_k, experts, hidden, inter, sm_count, record;
  std::vector<std::string> buffers;  // requested non-null optional buffers (kernel roles)
};

// One line: ID CONFIG ACT TOKENS TOP_K EXPERTS HIDDEN INTER SM_COUNT BUFFERS
// RECORD, where TOKENS .. INTER are the MoE problem (local experts), BUFFERS a
// comma list of the optional buffers passed non-null or "-", RECORD 1 to
// record the launch.
bool read_case(std::istream& in, Case& c) {
  std::string buffers;
  if (!(in >> c.id >> c.config >> c.act >> c.tokens >> c.top_k >> c.experts >> c.hidden >>
        c.inter >> c.sm_count >> buffers >> c.record)) {
    return false;
  }
  c.buffers.clear();
  if (buffers != "-") {
    std::stringstream ss(buffers);
    std::string item;
    while (std::getline(ss, item, ',')) c.buffers.push_back(item);
  }
  return true;
}

moe_bridge::Query query_of(const Case& c) {
  auto const& o = bg::BatchedGemmInterface().getBatchedGemmConfigs()[c.config].mOptions;
  moe_bridge::Query q;
  q.config = c.config;
  q.fc1 = !bg::doesRouteImplUseNoRoute(o.mRouteImpl);
  q.dtype_act = static_cast<uint32_t>(o.mDtypeB);
  q.dtype_weights = static_cast<uint32_t>(o.mDtypeA);
  q.deepseek = o.mUseDeepSeekFp8;
  q.fused_act = o.mFusedAct;
  q.shuffled = o.mUseShuffledMatrix;
  q.act_type = static_cast<int>(o.mActType);
  q.eltwise_act_type = static_cast<int>(o.mEltwiseActType);
  q.layout = static_cast<int>(o.mLayoutA);
  q.tile = o.mTileN;
  q.pair_act = c.act;
  q.top_k = c.top_k;
  q.hidden = c.hidden;
  q.inter = c.inter;
  q.experts = c.experts;
  q.tokens = c.tokens;
  return q;
}

// The optional buffers each MoE runner forwards to TrtllmGenBatchedGemmRunner::
// run (kernel roles): PermuteGemm1::Runner::run all of these, Gemm2::Runner::
// run null for the gate scale, the gated-activation parameters, the routing
// scales and the route map.
std::vector<std::string> forwarded(const Case& c, bool fc1) {
  static const std::set<std::string> fc1_buffers = {
      "sf_a", "sf_b", "sf_c", "scale_c", "scale_gate", "bias", "alpha", "beta",
      "clamp_limit", "route_map", "per_token_sf_b"};
  static const std::set<std::string> fc2_buffers = {"sf_a", "sf_b", "sf_c", "scale_c", "bias"};
  std::vector<std::string> out;
  for (auto const& b : c.buffers) {
    if ((fc1 ? fc1_buffers : fc2_buffers).count(b)) out.push_back(b);
  }
  return out;
}

struct Result {
  std::string error;
  std::vector<Launch> launches;
  std::vector<std::string> events;
};

__attribute__((noinline)) Result run_case(const Case& c, const moe_bridge::Query& q,
                                          const std::vector<std::string>& buffers,
                                          size_t workspace, unsigned char fill) {
  Result r;
  g_fill = fill;
  g_launches.clear();
  g_events.clear();
  g_pending_encodes.clear();
  g_sm_count = c.sm_count;
  auto ptr = [&](const char* name) -> void* {
    return reinterpret_cast<void*>(fake_address(name));
  };
  auto opt = [&](const char* name) -> void* {
    return std::find(buffers.begin(), buffers.end(), name) != buffers.end() ? ptr(name)
                                                                            : nullptr;
  };
  moe_bridge::Pointers p{};
  p.a = ptr("a");
  p.b = ptr("b");
  p.c = ptr("c");
  p.sf_a = opt("sf_a");
  p.sf_b = opt("sf_b");
  p.sf_c = opt("sf_c");
  p.scale_c = opt("scale_c");
  p.scale_gate = opt("scale_gate");
  p.bias = opt("bias");
  p.alpha = opt("alpha");
  p.beta = opt("beta");
  p.clamp_limit = opt("clamp_limit");
  p.route_map = opt("route_map");
  p.per_token_sf_b = opt("per_token_sf_b");
  p.total_num_padded_tokens = ptr("total_num_padded_tokens");
  p.cta_idx_xy_to_batch_idx = ptr("cta_idx_xy_to_batch_idx");
  p.cta_idx_xy_to_mn_limit = ptr("cta_idx_xy_to_mn_limit");
  p.num_non_exiting_ctas = ptr("num_non_exiting_ctas");
  p.workspace = workspace ? ptr("workspace") : nullptr;
  try {
    moe_bridge::run(q, p);
  } catch (std::exception const& e) {
    r.error = e.what();
  }
  r.launches = g_launches;
  r.events = g_events;
  return r;
}

// Bytes of the launch's KernelParams that KernelParamsSetup::setKernelParams
// leaves unset: rebuild it directly (BatchedGemmInterface::run's options for
// the runner-level problem the probe reports, the same operands) into storage
// pre-filled with 0x00 and then 0xff. The rebuild must reproduce the recorded
// bytes everywhere else; the probe aborts otherwise.
std::vector<bool> unset_bytes(const Case& c, const moe_bridge::Answer& p,
                              const std::vector<std::string>& buffers, const Launch& launch) {
  auto const bmm = bg::BatchedGemmInterface();
  auto const& config = bmm.getBatchedGemmConfigs()[c.config];
  // TrtllmGenBatchedGemmRunner::run's BatchedGemmData (transposeMmaOutput).
  bg::BatchedGemmData data{};
  data.mProblemDimensions.mNumBatches = c.experts;
  data.mProblemDimensions.mNumTokens = c.tokens;
  data.mProblemDimensions.mBatchM = false;
  data.mProblemDimensions.mBatchedN = {};
  data.mProblemDimensions.mM = p.n;
  data.mProblemDimensions.mN = p.m;
  data.mProblemDimensions.mK = p.k;
  data.mProblemDimensions.mValidM = p.n;
  data.mProblemDimensions.mValidN = p.m;
  data.mProblemDimensions.mValidK = p.k;
  data.mProblemDimensions.mMaxNumCtasInTokenDim = p.max_num_ctas;
  auto options = bmm.getOptionsFromConfigAndData(config, data);
  auto [numCtaBatch, numCtaTile, numCtaInner] = bmm.getGridDim(options, p.max_num_ctas);
  (void)numCtaTile;
  (void)numCtaInner;
  if (options.mUseDeepSeekFp8 && options.mFusedAct) {
    std::fprintf(stderr, "bmm_probe: DeepSeek FP8 with fused activation is not rebuilt\n");
    std::abort();
  }
  auto ptr = [&](const char* name) -> void* {
    return reinterpret_cast<void*>(fake_address(name));
  };
  auto opt = [&](const char* name) -> void* {
    return std::find(buffers.begin(), buffers.end(), name) != buffers.end() ? ptr(name)
                                                                            : nullptr;
  };
  auto f32 = [&](const char* name) { return static_cast<float const*>(opt(name)); };
  auto i32 = [&](const char* name) { return static_cast<int32_t const*>(ptr(name)); };
  using Params = batchedGemm::KernelParams;
  alignas(Params) static unsigned char storage[2][sizeof(Params)];
  for (int run = 0; run < 2; ++run) {
    unsigned char fill = run ? 0xff : 0x00;
    std::memset(storage[run], fill, sizeof(Params));
    poison_stack(fill);
    new (storage[run]) Params(bg::KernelParamsSetup::setKernelParams(
        options, /*batchM=*/false, ptr("a"), ptr("b"), ptr("c"), opt("sf_a"), opt("sf_b"),
        opt("per_token_sf_a"), opt("per_token_sf_b"), /*sparsityInfoA=*/nullptr, opt("bias"),
        opt("sf_c"), f32("scale_c"), /*scaleAct=*/f32("scale_gate"), f32("scale_gate"),
        f32("clamp_limit"), f32("alpha"), f32("beta"),
        static_cast<int32_t const*>(opt("route_map")),
        /*rowMax=*/nullptr, /*rowMaxBars=*/nullptr, i32("num_non_exiting_ctas"),
        i32("total_num_padded_tokens"), i32("cta_idx_xy_to_batch_idx"),
        i32("cta_idx_xy_to_mn_limit"), numCtaBatch, /*dynamicTileCounter=*/nullptr));
  }
  g_pending_encodes.clear();
  std::vector<bool> unset(sizeof(Params));
  for (size_t i = 0; i < sizeof(Params); ++i) {
    if (storage[0][i] != storage[1][i]) {
      unset[i] = true;
    } else if (storage[0][i] != launch.params.at(i)) {
      std::fprintf(stderr, "bmm_probe: %s: setKernelParams rebuild differs at byte %zu\n",
                   c.id.c_str(), i);
      std::abort();
    }
  }
  return unset;
}

std::string strings(const std::vector<std::string>& values) {
  std::string out = "[";
  for (size_t i = 0; i < values.size(); ++i) out += (i ? ", " : "") + json_string(values[i]);
  return out + "]";
}

void write_case(std::ostream& out, const Case& c) {
  auto const q = query_of(c);
  auto const buffers = forwarded(c, q.fc1);
  moe_bridge::Answer a;
  std::string error;
  try {
    moe_bridge::query(q, a);
  } catch (std::exception const& e) {
    error = e.what();
  }
  out << "{\"id\": " << json_string(c.id) << ", \"config\": " << c.config
      << ", \"launcher\": " << json_string(a.launcher) << ", \"ladder\": " << list(a.ladder)
      << ", \"selected\": " << list(a.selected) << ", \"activation\": " << a.activation
      << ", \"passing\": " << (a.passing ? "true" : "false")
      << ", \"valid\": " << (a.valid ? "true" : "false")
      << ", \"moe_valid\": " << (a.moe_valid ? "true" : "false")
      << ", \"default\": " << (a.is_default ? "true" : "false") << ", \"m\": " << a.m
      << ", \"n\": " << a.n << ", \"k\": " << a.k << ", \"max_num_ctas\": " << a.max_num_ctas
      << ", \"buffers\": " << strings(buffers) << ", \"workspace\": " << a.workspace;
  if (c.record && error.empty() && a.passing && a.valid) {
    Result zero = run_case(c, q, buffers, a.workspace, 0x00);
    Result ones = run_case(c, q, buffers, a.workspace, 0xff);
    if (!zero.error.empty()) error = zero.error;
    out << ", \"events\": [";
    for (size_t i = 0; i < zero.events.size(); ++i) out << (i ? ", " : "") << zero.events[i];
    out << "], \"launches\": [";
    if (zero.launches.size() != ones.launches.size()) {
      std::fprintf(stderr, "bmm_probe: launch count differs between runs\n");
      std::abort();
    }
    for (size_t i = 0; i < zero.launches.size(); ++i) {
      auto const& l = zero.launches[i];
      auto const& w = ones.launches[i];
      auto const unset = unset_bytes(c, a, buffers, l);
      std::vector<size_t> undefined;
      for (size_t b = 0; b < l.params.size(); ++b) {
        if (l.params[b] != w.params[b] || unset[b]) undefined.push_back(b);
      }
      out << (i ? ", " : "") << launch_json(l, undefined);
    }
    out << "]";
  }
  if (!error.empty()) out << ", \"error\": " << json_string(error);
  out << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: bmm_probe layout|configs|moe OUT\n");
    return 2;
  }
  std::string mode = argv[1];
  g_params_size = sizeof(batchedGemm::KernelParams);
  std::ofstream out(argv[2]);
  if (mode == "layout") {
    write_layout(out);
  } else if (mode == "configs") {
    write_configs(out);
  } else if (mode == "moe") {
    Case c;
    while (read_case(std::cin, c)) write_case(out, c);
  } else {
    return 2;
  }
  return out.good() ? 0 : 1;
}
