// SPDX-License-Identifier: Apache-2.0
// Compile-time, host-only probe of FlashInfer v0.6.9's dense trtllm-gen GEMM
// host code: TrtllmGenGemmRunner (csrc/trtllm_gemm_runner.cu) and
// TrtllmLowLatencyGemmRunner (csrc/trtllm_low_latency_gemm_runner.cu), copied
// verbatim in gemm_runners.cuh, over the pinned trtllmGen_gemm_export headers
// and their 74-config flashinferMetaInfo.h.
//
// Built and run by impls/batched_gemm/compiler.py; never part of a benchmark
// run. Like bmm_probe.cu it needs neither a GPU nor libcuda (probe_common.cuh).
//
//   gemm_probe layout OUT     sizeof/offsetof of every KernelParams field.
//   gemm_probe configs OUT    per config: name, smem, threads, sha256, the
//                             runner that reaches it and its passing check.
//   gemm_probe run OUT < CASES
//       One case per line: ID CONFIG M N K SM_COUNT BUFFERS RECORD, with M, N, K
//       the runner's (m = activation rows, n = weight rows), BUFFERS a comma
//       list of optional non-null buffers or "-" and RECORD 1 to launch. Per
//       case one JSON line with the runner's validity check, its valid
//       tactics (the autotuner's list), its tactic -1 heuristic and (RECORD)
//       the recorded launch.
//
// Indeterminate bytes are found as in bmm_probe.cu: two runs with the stack
// poisoned by 0x00 / 0xff (in cudaDeviceGetAttribute, called by both runners
// right before GemmInterface::run) plus a direct setKernelParams rebuild into
// pre-filled storage.
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

#include "gemm_runners.cuh"
#include "probe_common.cuh"

namespace flashinfer::trtllm_cubin_loader {
// loadCubinData fetches the cubin by name; the module load is interposed.
std::string getCubin(const std::string&, const std::string&) { return std::string(); }
}  // namespace flashinfer::trtllm_cubin_loader

namespace {

namespace gg = gemm::gemm;
namespace tg = gemm::trtllm::gen;

const char* const kBuffers[] = {"a",     "sf_a",           "b",
                                "sf_b",  "c",              "sf_c",
                                "scale_c", "workspace"};

uint64_t fake_address(const std::string& name) {
  for (size_t i = 0; i < sizeof(kBuffers) / sizeof(kBuffers[0]); ++i) {
    if (name == kBuffers[i]) return 0x7F0000000000ull + (uint64_t(i + 1) << 30);
  }
  std::fprintf(stderr, "gemm_probe: unknown buffer %s\n", name.c_str());
  std::abort();
}

// The low-latency runner owns the BlockMajorK + shuffled E4m3 -> Bf16 configs
// (its constructor filter); every other config is TrtllmGenGemmRunner's.
bool low_latency(const gg::GemmOptions& o) {
  return o.mLayoutA == gg::MatrixLayout::BlockMajorK && o.mUseShuffledMatrix;
}

flashinfer::TrtllmGenGemmRunnerOptions runner_options(const gg::GemmOptions& o) {
  return flashinfer::TrtllmGenGemmRunnerOptions{
      .eltType = o.mDtypeA,
      .outputType = o.mDtypeC,
      .transposeMmaOutput = o.mTransposeMmaOutput,
      .sfLayoutB = o.mSfLayoutB,
      .layoutA = o.mLayoutA,
  };
}

void write_layout(std::ostream& out) {
  using KP = gg::KernelParams;
  out << "{\"size\": " << sizeof(KP)
      << ", \"num_configs\": " << gg::GemmInterface().getNumGemmConfigs() << ", \"fields\": {";
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
  FIELD(ptrC, "ptr");
  FIELD(ptrSfA, "ptr");
  FIELD(ptrSfB, "ptr");
  FIELD(ptrBias, "ptr");
  FIELD(ptrPerTokenSfA, "ptr");
  FIELD(ptrPerTokenSfB, "ptr");
  FIELD(ptrSfC, "ptr");
  FIELD(ptrScaleC, "ptr");
  FIELD(ptrScaleAct, "ptr");
  FIELD(m, "i32");
  FIELD(n, "i32");
  FIELD(k, "i32");
  FIELD(rank, "i32");
  FIELD(tpGrpSize, "i32");
  FIELD(multimemC, "ptr");
  FIELD(ptrTileBars, "ptr");
  FIELD(multimemTileBars, "ptr");
  FIELD(ptrCompletionBars, "ptr");
  FIELD(multimemCompletionBars, "ptr");
  FIELD(ptrSplitKCompletionBars, "ptr");
  FIELD(ptrPartialSumsForSplitK, "ptr");
  FIELD(ptrNumNonExitingCtas, "ptr");
#undef FIELD
  out << "}}\n";
}

std::string options_json(gg::GemmOptions const& o) {
  std::string out = "{";
  bool first = true;
  auto add = [&](const char* name, long long value) {
    out += std::string(first ? "" : ", ") + "\"" + name + "\": " + std::to_string(value);
    first = false;
  };
#define X(f) add(#f, static_cast<long long>(o.f));
  PROBE_COMMON_OPTIONS(X)
  X(mNumStages)
  X(mNumStagesMma)
#undef X
  return out + "}";
}

void write_configs(std::ostream& out) {
  auto const gemm = gg::GemmInterface();
  auto const* configs = gemm.getGemmConfigs();
  for (size_t i = 0; i < gemm.getNumGemmConfigs(); ++i) {
    auto const& c = configs[i];
    out << "{\"index\": " << i << ", \"function\": \"" << c.mFunctionName
        << "\", \"shared_mem\": " << c.mSharedMemSize << ", \"threads\": " << c.mNumThreadsPerCTA
        << ", \"sha256\": \"" << c.mHash << "\", \"sm\": " << int(c.mSm) << ", \"runner\": \""
        << (low_latency(c.mOptions) ? "low_latency" : "gemm")
        << "\", \"options\": " << options_json(c.mOptions) << "}\n";
  }
}

struct Case {
  std::string id;
  int config, m, n, k, sm_count, record;
  std::vector<std::string> buffers;
};

// One line: ID CONFIG M N K SM_COUNT BUFFERS RECORD (the runners' m, n, k;
// BUFFERS a comma list or "-"; RECORD 0 skips the launch).
bool read_case(std::istream& in, Case& c) {
  std::string buffers;
  if (!(in >> c.id >> c.config >> c.m >> c.n >> c.k >> c.sm_count >> buffers >> c.record)) {
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

struct Result {
  bool valid = false;
  std::vector<int64_t> tactics;  // the runner's getValidTactics (autotuner list)
  int64_t heuristic = -1;        // the tactic its tactic = -1 path selects
  size_t workspace = 0;
  std::string error;
  std::vector<Launch> launches;
  std::vector<std::string> events;
};

// The runners' GemmData (transposeMmaOutput: M = n, N = m).
gg::GemmData gemm_data(const Case& c) {
  gg::GemmData data{};
  data.mProblemDimensions.mM = c.n;
  data.mProblemDimensions.mN = c.m;
  data.mProblemDimensions.mK = c.k;
  data.mProblemDimensions.mValidM = c.n;
  data.mProblemDimensions.mValidN = c.m;
  data.mProblemDimensions.mValidK = c.k;
  data.mProblemDimensions.mRank = 0;
  data.mProblemDimensions.mWorldSize = 1;
  return data;
}

__attribute__((noinline)) Result run_case(const Case& c, unsigned char fill) {
  Result r;
  g_fill = fill;
  g_launches.clear();
  g_events.clear();
  g_pending_encodes.clear();
  g_sm_count = c.sm_count;
  auto const gemm = gg::GemmInterface();
  auto const& config = gemm.getGemmConfigs()[c.config];
  auto p = [&](const char* name) -> void* {
    return reinterpret_cast<void*>(fake_address(name));
  };
  auto opt = [&](const char* name) -> void* {
    return std::find(c.buffers.begin(), c.buffers.end(), name) != c.buffers.end() ? p(name)
                                                                                    : nullptr;
  };
  try {
    r.valid = gemm.isValidConfig(config, gemm_data(c));
    if (low_latency(config.mOptions)) {
      flashinfer::TrtllmLowLatencyGemmRunner runner(flashinfer::TrtllmLowLatencyGemmRunnerOptions{
          .eltType = config.mOptions.mDtypeA, .outputType = config.mOptions.mDtypeC});
      r.tactics = runner.getValidTactics(c.m, c.n, c.k);
      try {
        r.heuristic = flashinfer::select_kernel(c.m, c.n, c.k, gemm);
      } catch (std::exception const&) {
      }
    } else {
      flashinfer::TrtllmGenGemmRunner runner(runner_options(config.mOptions));
      r.tactics = runner.getValidTactics(c.m, c.n, c.k);
      try {
        r.heuristic = runner.selectHeuristic(c.m, c.n, c.k);
      } catch (std::exception const&) {
      }
    }
    if (!r.valid || !c.record) return r;
    if (low_latency(config.mOptions)) {
      r.workspace = flashinfer::getWorkspaceSizeInBytes(c.m, c.n, c.k, c.config);
      flashinfer::TrtllmLowLatencyGemmRunner runner(flashinfer::TrtllmLowLatencyGemmRunnerOptions{
          .eltType = config.mOptions.mDtypeA, .outputType = config.mOptions.mDtypeC});
      // run(m, n, k, a, b, c, cScale, ...): it passes b (the weights) as the
      // kernel's A and a (the activations) as B.
      runner.run(c.m, c.n, c.k, /*a=*/p("b"), /*b=*/p("a"), p("c"), opt("scale_c"),
                 r.workspace ? p("workspace") : nullptr, /*stream=*/nullptr,
                 /*device_index=*/0, c.config);
    } else {
      flashinfer::TrtllmGenGemmRunner runner(runner_options(config.mOptions));
      r.workspace = runner.getWorkspaceSizeInBytes(c.m, c.n, c.k, c.config);
      // run(m, n, k, a, aScale, b, bScale, c, cScale, cScalePtr, ...): with
      // transposeMmaOutput the weights b become the kernel's A.
      runner.run(c.m, c.n, c.k, /*a=*/p("b"), /*aScale=*/opt("sf_b"), /*b=*/p("a"),
                 /*bScale=*/opt("sf_a"), p("c"), opt("scale_c"), opt("sf_c"),
                 r.workspace ? p("workspace") : nullptr, /*stream=*/nullptr,
                 /*device_index=*/0, c.config);
    }
  } catch (std::exception const& e) {
    r.error = e.what();
  }
  r.launches = g_launches;
  r.events = g_events;
  return r;
}

std::vector<bool> unset_bytes(const Case& c, const Launch& launch) {
  auto const gemm = gg::GemmInterface();
  auto const& config = gemm.getGemmConfigs()[c.config];
  auto options = gemm.getOptionsFromConfigAndData(config, gemm_data(c));
  if (gg::doesSplitKUseGmem(options.mSplitK)) {
    std::fprintf(stderr, "gemm_probe: global-memory split-K is not rebuilt\n");
    std::abort();
  }
  auto p = [&](const char* name) -> void* {
    return reinterpret_cast<void*>(fake_address(name));
  };
  // The low-latency runner passes no scale-factor buffers.
  bool const ll = low_latency(config.mOptions);
  auto opt = [&](const char* name) -> void* {
    bool const sf = std::string(name).rfind("sf_", 0) == 0;
    return !(ll && sf) && std::find(c.buffers.begin(), c.buffers.end(), name) != c.buffers.end()
               ? p(name)
               : nullptr;
  };
  using Params = gg::KernelParams;
  alignas(Params) static unsigned char storage[2][sizeof(Params)];
  for (int run = 0; run < 2; ++run) {
    unsigned char fill = run ? 0xff : 0x00;
    std::memset(storage[run], fill, sizeof(Params));
    poison_stack(fill);
    new (storage[run]) Params(gg::KernelParamsSetup::setKernelParams(
        options, p("a"), opt("sf_a"), /*perTokenSfA=*/nullptr, p("b"), opt("sf_b"),
        /*perTokenSfB=*/nullptr, /*sparsityInfoA=*/nullptr, /*bias=*/nullptr, p("c"),
        opt("sf_c"), /*multimemC=*/nullptr, static_cast<float*>(opt("scale_c")),
        /*scaleAct=*/nullptr, /*partialSumsForSplitK=*/nullptr, /*tileBars=*/nullptr,
        /*multimemTileBars=*/nullptr, /*completionBars=*/nullptr,
        /*multimemCompletionBars=*/nullptr, /*splitKCompletionBars=*/nullptr,
        /*numNonExitingCtas=*/nullptr, /*rank=*/0, /*worldSize=*/1));
  }
  g_pending_encodes.clear();
  std::vector<bool> unset(sizeof(Params));
  for (size_t i = 0; i < sizeof(Params); ++i) {
    if (storage[0][i] != storage[1][i]) {
      unset[i] = true;
    } else if (storage[0][i] != launch.params.at(i)) {
      std::fprintf(stderr, "gemm_probe: %s: setKernelParams rebuild differs at byte %zu\n",
                   c.id.c_str(), i);
      std::abort();
    }
  }
  return unset;
}

void write_case(std::ostream& out, const Case& c) {
  Result zero = run_case(c, 0x00);
  Result ones = c.record ? run_case(c, 0xff) : zero;
  out << "{\"id\": " << json_string(c.id) << ", \"config\": " << c.config
      << ", \"valid\": " << (zero.valid ? "true" : "false")
      << ", \"tactics\": " << list(zero.tactics) << ", \"heuristic\": " << zero.heuristic
      << ", \"workspace\": " << zero.workspace;
  if (!zero.error.empty()) out << ", \"error\": " << json_string(zero.error);
  out << ", \"events\": [";
  for (size_t i = 0; i < zero.events.size(); ++i) out << (i ? ", " : "") << zero.events[i];
  out << "], \"launches\": [";
  if (zero.launches.size() != ones.launches.size()) {
    std::fprintf(stderr, "gemm_probe: launch count differs between runs\n");
    std::abort();
  }
  for (size_t i = 0; i < zero.launches.size(); ++i) {
    auto const& l = zero.launches[i];
    auto const& o = ones.launches[i];
    auto const unset = unset_bytes(c, l);
    std::vector<size_t> undefined;
    for (size_t b = 0; b < l.params.size(); ++b) {
      if (l.params[b] != o.params[b] || unset[b]) undefined.push_back(b);
    }
    out << (i ? ", " : "") << launch_json(l, undefined);
  }
  out << "]}\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: gemm_probe layout|configs|run OUT\n");
    return 2;
  }
  std::string mode = argv[1];
  g_params_size = sizeof(gemm::gemm::KernelParams);
  std::ofstream out(argv[2]);
  if (mode == "layout") {
    write_layout(out);
  } else if (mode == "configs") {
    write_configs(out);
  } else if (mode == "run") {
    Case c;
    while (read_case(std::cin, c)) write_case(out, c);
  } else {
    return 2;
  }
  return out.good() ? 0 : 1;
}
