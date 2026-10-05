// SPDX-License-Identifier: Apache-2.0
// Second translation unit of bmm_probe (see bmm_probe.cu): FlashInfer v0.6.9's
// MoE runners (csrc/trtllm_fused_moe_runner.cu, compiled unmodified) and the
// fused-MoE launchers' tile selection, copied verbatim from
// csrc/trtllm_fused_moe_kernel_launcher.cu (VERBATIM blocks, checked line for
// line by impls/batched_gemm/compiler.py). The launchers themselves need TVM
// FFI and are not compiled: which launcher serves which (activation, weight)
// dtypes is FlashInfer's dispatch (trtllm_*_moe entry points), restated in
// launcher_ladder below.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <map>
#include <memory>
#include <set>
#include <tuple>
#include <sstream>

#include "csrc/trtllm_fused_moe_runner.cu"

#include "moe_bridge.h"

// The routing, activation, permute, conversion and finalize kernels of the MoE
// runner are never launched by PermuteGemm1/Gemm2::Runner::run.
namespace moe::dev {
namespace routing {
namespace routingDeepSeek {
void run(Data&, void*) { std::abort(); }
}  // namespace routingDeepSeek
namespace routingLlama4 {
void run(Data const&, void*) { std::abort(); }
}  // namespace routingLlama4
namespace routingCustom {
void run(Data const&, void*) { std::abort(); }
}  // namespace routingCustom
}  // namespace routing
namespace activation {
void run(Data const&, void*) { std::abort(); }
}  // namespace activation
namespace convertsf {
void run(Data const&, void*) { std::abort(); }
}  // namespace convertsf
namespace permute {
void run(Data const&, void*) { std::abort(); }
}  // namespace permute
namespace finalize {
void run(Data const&, void*) { std::abort(); }
}  // namespace finalize
}  // namespace moe::dev

namespace probe_launcher {
namespace btg = batchedGemm::trtllm::gen;

// TVM_FFI_ICHECK of the launcher source: abort with the streamed message.
struct CheckFailure {
  std::ostringstream message;
  template <class T>
  CheckFailure& operator<<(T const& value) {
    message << value;
    return *this;
  }
  ~CheckFailure() {
    std::fprintf(stderr, "bmm_probe: check failed: %s\n", message.str().c_str());
    std::abort();
  }
};
#define TVM_FFI_ICHECK(condition) \
  if (condition) {                \
  } else                          \
    ::probe_launcher::CheckFailure()

// VERBATIM BEGIN csrc/trtllm_fused_moe_kernel_launcher.cu:61-66
enum class Fp8QuantizationType {
  NoneFp8,
  DeepSeekFp8,
  MxFp8,
  PerTensorFp8,
};
// VERBATIM END

// VERBATIM BEGIN csrc/trtllm_fused_moe_kernel_launcher.cu:89-106
inline int32_t nextPowerOfTwo(float value) {
  int32_t n = static_cast<int32_t>(std::ceil(value));
  if (n <= 1) return 1;

  // If n is already a power of 2, return it
  if ((n & (n - 1)) == 0) return n;

  // Find the next power of 2
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n++;

  return n;
}
// VERBATIM END

// VERBATIM BEGIN csrc/trtllm_fused_moe_kernel_launcher.cu:108-144
std::set<int32_t> computeSelectedTileN(std::vector<int32_t> const& supported_tile_nums,
                                       int64_t const num_tokens, int64_t const top_k,
                                       int64_t const num_local_experts) {
  TVM_FFI_ICHECK(!supported_tile_nums.empty()) << "supported_tile_nums must not be empty.";
  float const avg_tokens_per_expert = static_cast<float>(num_tokens * top_k) / num_local_experts;
  // NOTE: This differs from Python AutoTuner bucketing:
  // - AutoTuner maps raw num_tokens with last_positive_power_of_2 (round-down).
  // - Here we map derived avg_tokens_per_expert and use nextPowerOfTwo (round-up).
  // Because they round different quantities in different directions, cache bucket and runtime
  // tile candidates can diverge; launcher-side tactic resolution handles that mismatch.
  // assume supported_tile_nums is sorted
  int32_t tile_tokens_dim = std::clamp(nextPowerOfTwo(avg_tokens_per_expert),
                                       supported_tile_nums.front(), supported_tile_nums.back());
  auto it = std::find(supported_tile_nums.begin(), supported_tile_nums.end(), tile_tokens_dim);
  FLASHINFER_CHECK(
      it != supported_tile_nums.end(), "computeSelectedTileN expected exact tile ", tile_tokens_dim,
      " in supported_tile_nums (size=", supported_tile_nums.size(),
      "). Please keep supported_tile_nums as a dense power-of-2 ladder for this launcher.");

  // Candidate tile set centered on the heuristic tile.
  // This function returns nearby candidates (not a single final tile):
  //   center, +1, +2, and -1 neighbors when available.
  // Final tile choice is made later (autotuner-provided tile if valid, otherwise fallback policy).
  std::set<int32_t> selected_tile_nums;
  selected_tile_nums.insert(tile_tokens_dim);
  if (std::next(it) != supported_tile_nums.end()) {
    selected_tile_nums.insert(*std::next(it));
    if (std::next(std::next(it)) != supported_tile_nums.end()) {
      selected_tile_nums.insert(*std::next(std::next(it)));
    }
  }
  if (it != supported_tile_nums.begin()) {
    selected_tile_nums.insert(*std::prev(it));
  }

  return selected_tile_nums;
}
// VERBATIM END

struct Bf16MoeLauncher {
// VERBATIM BEGIN csrc/trtllm_fused_moe_kernel_launcher.cu:550-550
  static constexpr std::array<int32_t, 5> mSupportedTileNums = {8, 16, 32, 64, 128};
// VERBATIM END
};

struct Fp8PerTensorLauncher {
// VERBATIM BEGIN csrc/trtllm_fused_moe_kernel_launcher.cu:696-696
  static constexpr std::array<int32_t, 5> mSupportedTileNums = {8, 16, 32, 64, 128};
// VERBATIM END
};

struct Fp8BlockScaleLauncher {
// VERBATIM BEGIN csrc/trtllm_fused_moe_kernel_launcher.cu:895-903
  static constexpr std::array<int32_t, 5> mBaseSupportedTileNums = {8, 16, 32, 64, 128};

  static std::vector<int32_t> getSupportedTileNums(Fp8QuantizationType quantization_type) {
    std::vector<int32_t> tiles(mBaseSupportedTileNums.begin(), mBaseSupportedTileNums.end());
    if (quantization_type == Fp8QuantizationType::MxFp8) {
      tiles.push_back(256);
    }
    return tiles;
  }
// VERBATIM END
};

struct MxInt4BlockScaleLauncher {
// VERBATIM BEGIN csrc/trtllm_fused_moe_kernel_launcher.cu:1300-1300
  static constexpr std::array<int32_t, 5> mSupportedTileNums = {8, 16, 32, 64, 128};
// VERBATIM END
};

struct FP4BlockScaleLauncher {
// VERBATIM BEGIN csrc/trtllm_fused_moe_kernel_launcher.cu:1460-1469
  static constexpr std::array<int32_t, 4> mBaseSupportedTileNums = {8, 16, 32, 64};

  static std::vector<int32_t> getSupportedTileNums(btg::Dtype dtype_act) {
    std::vector<int32_t> tiles(mBaseSupportedTileNums.begin(), mBaseSupportedTileNums.end());
    if (dtype_act != btg::Dtype::Bfloat16) {
      tiles.push_back(128);
      tiles.push_back(256);
    }
    return tiles;
  }
// VERBATIM END
};

#undef TVM_FFI_ICHECK
}  // namespace probe_launcher

namespace {

namespace tg = batchedGemm::trtllm::gen;
namespace gm = batchedGemm::gemm;
namespace tmoe = tensorrt_llm::kernels::trtllmgen_moe;
using tmoe::MoE::ActivationType;

// The MoE activation whose PermuteGemm1 options (getOptions) build a routed
// config: DeepSeek FP8 runners are always built with SwiGlu.
ActivationType config_activation(moe_bridge::Query const& q) {
  if (q.deepseek) return ActivationType::Swiglu;
  if (q.fused_act) {
    return q.act_type == static_cast<int>(tensorrt_llm::kernels::ActType::SwiGlu)
               ? ActivationType::Swiglu
               : ActivationType::Geglu;
  }
  return q.eltwise_act_type == static_cast<int>(tensorrt_llm::kernels::EltwiseActType::Relu2)
             ? ActivationType::Relu2
             : ActivationType::Identity;
}

ActivationType activation_of(moe_bridge::Query const& q) {
  if (q.fc1 || q.pair_act == "-") return config_activation(q);
  if (q.pair_act == "Swiglu") return ActivationType::Swiglu;
  if (q.pair_act == "Geglu") return ActivationType::Geglu;
  if (q.pair_act == "Relu2") return ActivationType::Relu2;
  if (q.pair_act == "Identity") return ActivationType::Identity;
  std::fprintf(stderr, "bmm_probe: unknown activation %s\n", q.pair_act.c_str());
  std::abort();
}

// The launcher whose MoE::Runner is built with the config's (activation,
// weight) dtypes: trtllm_bf16_moe, trtllm_fp8_per_tensor_scale_moe,
// trtllm_fp8_block_scale_moe (DeepSeek FP8 / MxFP8), trtllm_mxint4_block_scale_moe
// and trtllm_fp4_block_scale_moe (NvFP4 / MxFP4 weights), and its tile ladder.
std::pair<std::string, std::vector<int32_t>> launcher_ladder(moe_bridge::Query const& q) {
  namespace pl = probe_launcher;
  auto const act = static_cast<tg::Dtype>(q.dtype_act);
  auto const weights = static_cast<tg::Dtype>(q.dtype_weights);
  auto vec = [](auto const& a) { return std::vector<int32_t>(a.begin(), a.end()); };
  if (weights == tg::Dtype::Bfloat16 && act == tg::Dtype::Bfloat16) {
    return {"Bf16MoeLauncher", vec(pl::Bf16MoeLauncher::mSupportedTileNums)};
  }
  if (weights == tg::Dtype::E4m3 && act == tg::Dtype::E4m3) {
    if (q.deepseek) {
      return {"Fp8BlockScaleLauncher/DeepSeekFp8",
              pl::Fp8BlockScaleLauncher::getSupportedTileNums(pl::Fp8QuantizationType::DeepSeekFp8)};
    }
    return {"Fp8PerTensorLauncher", vec(pl::Fp8PerTensorLauncher::mSupportedTileNums)};
  }
  if (weights == tg::Dtype::MxE4m3 && act == tg::Dtype::MxE4m3) {
    return {"Fp8BlockScaleLauncher/MxFp8",
            pl::Fp8BlockScaleLauncher::getSupportedTileNums(pl::Fp8QuantizationType::MxFp8)};
  }
  if (weights == tg::Dtype::MxInt4) {
    return {"MxInt4BlockScaleLauncher", vec(pl::MxInt4BlockScaleLauncher::mSupportedTileNums)};
  }
  if (weights == tg::Dtype::E2m1 || weights == tg::Dtype::MxE2m1) {
    return {"FP4BlockScaleLauncher", pl::FP4BlockScaleLauncher::getSupportedTileNums(act)};
  }
  return {"none", {}};
}

// Runners are built once per option set (their constructors scan all 1,405
// configs).
using RunnerKey = std::tuple<uint32_t, uint32_t, bool, int, int64_t, bool, int>;

RunnerKey key_of(moe_bridge::Query const& q, ActivationType act) {
  return {q.dtype_act, q.dtype_weights, q.deepseek, q.tile, static_cast<int64_t>(act),
          q.shuffled, q.layout};
}

tmoe::PermuteGemm1::Runner& gemm1(moe_bridge::Query const& q, ActivationType act) {
  static std::map<RunnerKey, std::unique_ptr<tmoe::PermuteGemm1::Runner>> cache;
  auto& slot = cache[key_of(q, act)];
  if (!slot) {
    slot = std::make_unique<tmoe::PermuteGemm1::Runner>(
        static_cast<tg::Dtype>(q.dtype_act), static_cast<tg::Dtype>(q.dtype_weights), q.deepseek,
        q.tile, act, q.shuffled, static_cast<gm::MatrixLayout>(q.layout));
  }
  return *slot;
}

tmoe::Gemm2::Runner& gemm2(moe_bridge::Query const& q) {
  static std::map<RunnerKey, std::unique_ptr<tmoe::Gemm2::Runner>> cache;
  auto& slot = cache[key_of(q, ActivationType::InvalidType)];
  if (!slot) {
    slot = std::make_unique<tmoe::Gemm2::Runner>(
        static_cast<tg::Dtype>(q.dtype_act), static_cast<tg::Dtype>(q.dtype_weights),
        tg::Dtype::Bfloat16, q.deepseek, q.tile, q.shuffled,
        static_cast<gm::MatrixLayout>(q.layout));
  }
  return *slot;
}

// MoE::Runner, or nullptr where its constructor throws (no compatible pair).
tmoe::MoE::Runner* moe_runner(moe_bridge::Query const& q, ActivationType act) {
  static std::map<RunnerKey, std::unique_ptr<tmoe::MoE::Runner>> cache;
  static std::set<RunnerKey> failed;
  auto const key = key_of(q, act);
  if (failed.count(key)) return nullptr;
  auto& slot = cache[key];
  if (!slot) {
    try {
      slot = std::make_unique<tmoe::MoE::Runner>(
          static_cast<tg::Dtype>(q.dtype_act), static_cast<tg::Dtype>(q.dtype_weights),
          q.deepseek, q.tile, act, q.shuffled, static_cast<gm::MatrixLayout>(q.layout));
    } catch (std::exception const&) {
      failed.insert(key);
      cache.erase(key);
      return nullptr;
    }
  }
  return slot.get();
}

std::vector<int64_t> const& passing(moe_bridge::Query const& q, ActivationType act, bool fc1) {
  static std::map<std::pair<RunnerKey, bool>, std::vector<int64_t>> cache;
  auto const key = std::make_pair(key_of(q, fc1 ? act : ActivationType::InvalidType), fc1);
  auto it = cache.find(key);
  if (it == cache.end()) {
    auto indices =
        fc1 ? gemm1(q, act).getPassingConfigIndices() : gemm2(q).getPassingConfigIndices();
    it = cache.emplace(key, std::move(indices)).first;
  }
  return it->second;
}

}  // namespace

namespace moe_bridge {

void query(Query const& q, Answer& a) {
  auto const act = activation_of(q);
  a.activation = static_cast<int64_t>(act);
  auto [launcher, ladder] = launcher_ladder(q);
  a.launcher = launcher;
  a.ladder = ladder;
  if (!ladder.empty()) {
    auto const s = probe_launcher::computeSelectedTileN(ladder, q.tokens, q.top_k, q.experts);
    a.selected.assign(s.begin(), s.end());
  }
  // PermuteGemm1::Runner::run: run(numTokens, factor * intermediateSize,
  // hiddenSize, ...); Gemm2::Runner::run: run(numTokens, hiddenSize,
  // intermediateSize, ...); both with Routing::getMaxNumCtasInBatchDim.
  int const factor = tmoe::MoE::isGatedActivation(act) ? 2 : 1;
  a.m = q.tokens;
  a.n = q.fc1 ? factor * q.inter : q.hidden;
  a.k = q.fc1 ? q.hidden : q.inter;
  a.max_num_ctas = tmoe::Routing::getMaxNumCtasInBatchDim(q.tokens, q.top_k, q.experts, q.tile);
  if (q.fc1) {
    auto const& runner = gemm1(q, act);
    auto const& idx = passing(q, act, true);
    a.passing = std::find(idx.begin(), idx.end(), int64_t(q.config)) != idx.end();
    a.valid = runner.isValidConfigIndex(q.config, q.top_k, q.hidden, q.inter, q.experts,
                                        q.tokens);
    if (a.passing && a.valid) {
      a.workspace = runner.getWorkspaceSizeInBytes(q.top_k, q.hidden, q.inter, q.experts,
                                                   q.tokens, q.config);
    }
  } else {
    auto const& runner = gemm2(q);
    auto const& idx = passing(q, act, false);
    a.passing = std::find(idx.begin(), idx.end(), int64_t(q.config)) != idx.end();
    a.valid = runner.isValidConfigIndex(q.config, q.top_k, q.hidden, q.inter, q.experts,
                                        q.tokens);
    if (a.passing && a.valid) {
      a.workspace = runner.getWorkspaceSizeInBytes(q.top_k, q.hidden, q.inter, q.experts,
                                                   q.tokens, q.config);
    }
  }
  // MoE::Runner's mPassingConfigs is the cartesian product of the PermuteGemm1
  // and Gemm2 passing configs (its constructor); getValidConfigIndices (the
  // autotuner's list) and getDefaultValidConfigIndex (tactic -1) index it.
  if (auto* runner = moe_runner(q, act)) {
    auto const& i1 = passing(q, act, true);
    auto const& i2 = passing(q, act, false);
    auto mine = [&](int64_t index) {
      auto const c = q.fc1 ? i1.at(index / i2.size()) : i2.at(index % i2.size());
      return c == q.config;
    };
    for (auto index :
         runner->getValidConfigIndices(q.top_k, q.hidden, q.inter, q.experts, q.tokens)) {
      a.moe_valid = a.moe_valid || mine(index);
    }
    try {
      a.is_default = mine(
          runner->getDefaultValidConfigIndex(q.top_k, q.hidden, q.inter, q.experts, q.tokens));
    } catch (std::exception const&) {
    }
  }
}

void run(Query const& q, Pointers const& p) {
  auto const act = activation_of(q);
  auto f32 = [](void* ptr) { return static_cast<float*>(ptr); };
  auto i32 = [](void* ptr) { return static_cast<int32_t*>(ptr); };
  if (q.fc1) {
    // The runner's hidden states are the kernel's B (transposeMmaOutput),
    // the weights its A; expertWeights (routing scales on the input) become
    // the kernel's per-token B scales.
    gemm1(q, act).run(p.b, p.sf_b, p.a, p.sf_a, p.per_token_sf_b, f32(p.scale_c),
                      f32(p.scale_gate), f32(p.bias), f32(p.alpha), f32(p.beta),
                      f32(p.clamp_limit), p.c, p.sf_c, q.top_k, q.hidden, q.inter, q.experts,
                      q.tokens, i32(p.route_map), i32(p.num_non_exiting_ctas),
                      i32(p.total_num_padded_tokens), i32(p.cta_idx_xy_to_batch_idx),
                      i32(p.cta_idx_xy_to_mn_limit), p.workspace,
                      /*useRoutingScalesOnInput=*/p.per_token_sf_b != nullptr, /*device=*/0,
                      /*stream=*/nullptr, q.config, /*enable_pdl=*/true);
  } else {
    gemm2(q).run(p.b, p.sf_b, p.a, p.sf_a, f32(p.scale_c), f32(p.bias), p.c, p.sf_c, q.top_k,
                 q.hidden, q.inter, q.experts, q.tokens, i32(p.num_non_exiting_ctas),
                 i32(p.total_num_padded_tokens), i32(p.cta_idx_xy_to_batch_idx),
                 i32(p.cta_idx_xy_to_mn_limit), p.workspace, /*device=*/0, /*stream=*/nullptr,
                 q.config, /*enable_pdl=*/true);
  }
}

}  // namespace moe_bridge
