// SPDX-License-Identifier: Apache-2.0
// FlashInfer v0.6.9's two dense trtllm-gen GEMM runners, copied verbatim
// (VERBATIM blocks; impls/batched_gemm/compiler.py checks them byte for byte
// against the pinned sources) without their TVM-FFI entry points, so the
// host-only gemm_probe.cu can run them with fake device addresses. The TVM-FFI
// check macros they use are replaced by exception-throwing stand-ins.
#pragma once
#include <cuda.h>

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "flashinfer/exception.h"
#include "flashinfer/trtllm/common.h"
#include "flashinfer/trtllm/gemm/trtllmGen_gemm_export/Enums.h"
#include "flashinfer/trtllm/gemm/trtllmGen_gemm_export/GemmInterface.h"
#include "flashinfer/trtllm/gemm/trtllmGen_gemm_export/trtllm/gen/DtypeDecl.h"
#include "flashinfer/trtllm/gemm/trtllmGen_gemm_export/trtllm/gen/SfLayoutDecl.h"

namespace probe_ffi {
// Collects the streamed message and throws when the failed check's full
// expression ends.
struct Fail {
  std::ostringstream message;
  template <class T>
  Fail& operator<<(T const& value) {
    message << value;
    return *this;
  }
  ~Fail() noexcept(false) { throw std::runtime_error(message.str()); }
};
}  // namespace probe_ffi
#define TVM_FFI_ICHECK(cond) \
  if (cond) {                \
  } else                     \
    ::probe_ffi::Fail()
#define TVM_FFI_LOG_AND_THROW(kind) ::probe_ffi::Fail()

// VERBATIM BEGIN csrc/trtllm_gemm_runner.cu:30-275
namespace {
static thread_local gemm::gemm::GemmInterface::ModuleCache globalTrtllmGenGemmModuleCache;
}  // namespace

namespace flashinfer {

struct TrtllmGenGemmRunnerOptions {
  gemm::trtllm::gen::Dtype eltType;
  gemm::trtllm::gen::Dtype outputType;
  bool transposeMmaOutput{false};
  gemm::trtllm::gen::SfLayout sfLayoutB;
  gemm::gemm::MatrixLayout layoutA{gemm::gemm::MatrixLayout::MajorK};
};

int64_t select_kernel_fp8(int32_t M, int32_t N, int32_t K,
                          const gemm::gemm::GemmInterface& interface) {
  static constexpr const char* KERNEL_NAME_HIGH_N_K_RATIO =
      "gemm_Bfloat16_E4m3E4m3_Fp32_t128x8x128u2_s6_et64x8_m64x8x32_c1x1x1_16dp256b_rM_TN_"
      "transOut_"
      "noShflA_dsFp8_schPd2x2x1x3_sm100f";

  static constexpr const char* KERNEL_NAME_LOW_N_K_RATIO =
      "gemm_Bfloat16_E4m3E4m3_Fp32_t128x32x128u2_s6_et64x32_m64x32x32_c1x1x1_16dp256b_rM_TN_"
      "transOut_noShflA_dsFp8_schedS_sm100f";

  static constexpr const char* KERNEL_NAME_LARGE_N =
      "gemm_Bfloat16_E4m3E4m3_Fp32_t128x32x128u2_s6_et64x32_m64x32x32_c1x1x1_16dp256b_rM_TN_"
      "transOut_noShflA_dsFp8_schPd2x2x1x3_sm100f";

  static constexpr const char* KERNEL_NAME_DEFAULT =
      "gemm_Bfloat16_E4m3E4m3_Fp32_t128x16x128u2_s6_et64x16_m64x16x32_c1x1x1_16dp256b_rM_TN_"
      "transOut_noShflA_dsFp8_schedS_sm100f";

  double const n_k_ratio = static_cast<double>(N) / static_cast<double>(K);

  std::string kernel_name;
  if (n_k_ratio >= 32) {
    kernel_name = KERNEL_NAME_HIGH_N_K_RATIO;
  } else if (n_k_ratio <= 2.0) {
    kernel_name = KERNEL_NAME_LOW_N_K_RATIO;
  } else if (N >= 20000) {
    kernel_name = KERNEL_NAME_LARGE_N;
  } else {
    kernel_name = KERNEL_NAME_DEFAULT;
  }

  auto const& configs = interface.getGemmConfigs();
  size_t const num_configs = interface.getNumGemmConfigs();

  for (size_t i = 0; i < num_configs; ++i) {
    if (std::string(configs[i].mFunctionName) == kernel_name) {
      return static_cast<int64_t>(i);
    }
  }

  TVM_FFI_ICHECK(false) << "Kernel not found";
}

class TrtllmGenGemmRunner {
 public:
  explicit TrtllmGenGemmRunner(TrtllmGenGemmRunnerOptions const& options) : mOptions(options) {
    // Select a GEMM kernel config to use
    auto const gemm = gemm::gemm::GemmInterface();
    auto const configs = gemm.getGemmConfigs();

    mPassingConfigIndices.clear();

    for (size_t i = 0; i < gemm.getNumGemmConfigs(); ++i) {
      auto const options = configs[i].mOptions;

      if (options.mDtypeA == mOptions.eltType && options.mDtypeC == mOptions.outputType &&
          options.mTransposeMmaOutput == mOptions.transposeMmaOutput &&
          options.mSfLayoutB == mOptions.sfLayoutB &&
          options.mLayoutA == mOptions.layoutA) {  // FIXME(siyuanf): expose matrix layout to user
        mPassingConfigIndices.push_back(i);
      }
    }

    FLASHINFER_CHECK(mPassingConfigIndices.size() > 0,
                     "No valid tactic found for the given options",
                     "mDtypeA: ", gemm::trtllm::gen::dtypeToString(mOptions.eltType),
                     "mDtypeC: ", gemm::trtllm::gen::dtypeToString(mOptions.outputType),
                     "mTransposeMmaOutput: ", mOptions.transposeMmaOutput,
                     "mSfLayoutB: ", gemm::trtllm::gen::sfLayoutToString(mOptions.sfLayoutB));
  }

  int64_t getWorkspaceSizeInBytes(int64_t m, int64_t n, int64_t k, int64_t tactic) {
    auto gemm = gemm::gemm::GemmInterface();
    auto const configs = gemm.getGemmConfigs();
    FLASHINFER_CHECK(tactic >= 0 && tactic < gemm.getNumGemmConfigs(),
                     "Invalid tactic in getWorkspaceSizeInBytes");
    auto const config = configs[tactic];

    gemm::gemm::GemmData gemmData;
    gemmData.mProblemDimensions.mM = mOptions.transposeMmaOutput ? n : m;
    gemmData.mProblemDimensions.mN = mOptions.transposeMmaOutput ? m : n;
    gemmData.mProblemDimensions.mK = k;
    gemmData.mProblemDimensions.mValidM = gemmData.mProblemDimensions.mM;
    gemmData.mProblemDimensions.mValidN = gemmData.mProblemDimensions.mN;
    gemmData.mProblemDimensions.mValidK = gemmData.mProblemDimensions.mK;
    gemmData.mProblemDimensions.mRank = 0;
    gemmData.mProblemDimensions.mWorldSize = 1;

    return gemm.getWorkspaceSizeInBytes(config, gemmData);
  }

  void run(int64_t m, int64_t n, int64_t k, void const* a, void const* aScale, void const* b,
           void const* bScale, void* c, void* cScale, void* cScalePtr, void* workspace,
           CUstream stream, int32_t device_index, int64_t tactic) {
    auto gemm = gemm::gemm::GemmInterface();
    auto const configs = gemm.getGemmConfigs();
    TVM_FFI_ICHECK(tactic >= 0 && tactic < gemm.getNumGemmConfigs()) << "Invalid tactic id in run";
    auto const& config = configs[tactic];
    TVM_FFI_ICHECK(config.mOptions.mSfLayoutB == mOptions.sfLayoutB) << "Invalid sf layout in run";

    gemm::gemm::GemmData gemmData;
    // Dims
    gemmData.mProblemDimensions.mM = mOptions.transposeMmaOutput ? n : m;
    gemmData.mProblemDimensions.mN = mOptions.transposeMmaOutput ? m : n;
    gemmData.mProblemDimensions.mK = k;
    gemmData.mProblemDimensions.mValidM = gemmData.mProblemDimensions.mM;
    gemmData.mProblemDimensions.mValidN = gemmData.mProblemDimensions.mN;
    gemmData.mProblemDimensions.mValidK = gemmData.mProblemDimensions.mK;
    gemmData.mProblemDimensions.mRank = 0;
    gemmData.mProblemDimensions.mWorldSize = 1;

    gemmData.mProblemDimensions.mValidM = gemmData.mProblemDimensions.mM;
    gemmData.mProblemDimensions.mValidN = gemmData.mProblemDimensions.mN;
    gemmData.mProblemDimensions.mValidK = gemmData.mProblemDimensions.mK;

    // Inputs
    gemmData.mInputBuffers.mPtrA = mOptions.transposeMmaOutput ? b : a;
    gemmData.mInputBuffers.mPtrSfA = mOptions.transposeMmaOutput ? bScale : aScale;
    gemmData.mInputBuffers.mPtrB = mOptions.transposeMmaOutput ? a : b;
    gemmData.mInputBuffers.mPtrSfB = mOptions.transposeMmaOutput ? aScale : bScale;
    gemmData.mInputBuffers.mPtrScaleC = cScale;

    // Outputs
    gemmData.mOutputBuffers.mPtrC = c;
    gemmData.mOutputBuffers.mPtrSfC = cScalePtr;

    TVM_FFI_ICHECK(gemm.isValidConfig(config, gemmData)) << "unsupported tactic id in run";

    const int32_t multiProcessorCount = [device_index]() {
      static thread_local int32_t cached_multi_processor_count = -1;
      static thread_local int cached_device_index = -1;

      if (device_index == cached_device_index && cached_multi_processor_count != -1) {
        return cached_multi_processor_count;
      } else {
        int32_t count;
        cudaError_t cudaStatus =
            cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device_index);
        TVM_FFI_ICHECK(cudaStatus == cudaSuccess)
            << "Failed to get device attribute: " << cudaGetErrorString(cudaStatus);
        cached_multi_processor_count = count;
        cached_device_index = device_index;
        return count;
      }
    }();

    TVM_FFI_ICHECK(gemm.run(config, workspace, gemmData, static_cast<void*>(stream),
                            multiProcessorCount, true, globalTrtllmGenGemmModuleCache) == 0)
        << "Error occurred when running GEMM!";
  }

  std::vector<int64_t> getValidTactics(int64_t m, int64_t n, int64_t k) const {
    auto const gemm = gemm::gemm::GemmInterface();
    auto const configs = gemm.getGemmConfigs();

    gemm::gemm::GemmData gemmData;
    // Dims
    gemmData.mProblemDimensions.mM = mOptions.transposeMmaOutput ? n : m;
    gemmData.mProblemDimensions.mN = mOptions.transposeMmaOutput ? m : n;
    gemmData.mProblemDimensions.mK = k;
    gemmData.mProblemDimensions.mValidM = gemmData.mProblemDimensions.mM;
    gemmData.mProblemDimensions.mValidN = gemmData.mProblemDimensions.mN;
    gemmData.mProblemDimensions.mValidK = gemmData.mProblemDimensions.mK;
    gemmData.mProblemDimensions.mRank = 0;
    gemmData.mProblemDimensions.mWorldSize = 1;

    gemmData.mProblemDimensions.mValidM = gemmData.mProblemDimensions.mM;
    gemmData.mProblemDimensions.mValidN = gemmData.mProblemDimensions.mN;
    gemmData.mProblemDimensions.mValidK = gemmData.mProblemDimensions.mK;

    std::vector<int64_t> sortedIndices = mPassingConfigIndices;
    std::sort(sortedIndices.begin(), sortedIndices.end(), [&configs](int64_t idx0, int64_t idx1) {
      auto const& optionsA = configs[idx0].mOptions;
      auto const& optionsB = configs[idx1].mOptions;

      // Sort by tileK sizes first
      if (optionsA.mTileK != optionsB.mTileK) {
        return optionsA.mTileK > optionsB.mTileK;
      }

      // Then by splitK sizes
      if (optionsA.mNumSlicesForSplitK != optionsB.mNumSlicesForSplitK) {
        return optionsA.mNumSlicesForSplitK > optionsB.mNumSlicesForSplitK;
      }

      // Then by unroll loop 2x for mma
      if (optionsA.mUseUnrollLoop2xForMma != optionsB.mUseUnrollLoop2xForMma) {
        return optionsA.mUseUnrollLoop2xForMma;
      }

      return false;
    });

    bool findLoop2xMma = false;
    std::vector<int64_t> validTactics;
    for (auto const& configIndex : sortedIndices) {
      auto const& config = configs[configIndex];
      if (gemm.isValidConfig(config, gemmData)) {
        validTactics.push_back(configIndex);

        // when loop2x mma is found, only add the tactic that has loop2x mma
        if (!findLoop2xMma) {
          if (config.mOptions.mUseUnrollLoop2xForMma) {
            findLoop2xMma = true;
          }
        } else {
          if (!config.mOptions.mUseUnrollLoop2xForMma) {
            break;
          }
        }
      }
    }
    return validTactics;
  }

  int64_t selectHeuristic(int64_t m, int64_t n, int64_t k) const {
    if (mOptions.eltType == gemm::trtllm::gen::Dtype::E4m3) {
      return select_kernel_fp8(m, n, k, gemm::gemm::GemmInterface());
    } else {
      auto sortedIndices = getValidTactics(m, n, k);
      TVM_FFI_ICHECK(!sortedIndices.empty()) << "No valid tactic found";

      // the getValidTactics is sorted by priority, so the first one is the best one
      return sortedIndices[0];
    }
  }

 private:
  TrtllmGenGemmRunnerOptions mOptions;
  std::vector<int64_t> mPassingConfigIndices;
};
// VERBATIM END
}  // namespace flashinfer

// VERBATIM BEGIN csrc/trtllm_low_latency_gemm_runner.cu:31-36
namespace {
static thread_local gemm::gemm::GemmInterface::ModuleCache globalTrtllmLowLatencyGemmModuleCache;
}  // namespace

namespace flashinfer {

// VERBATIM END
// VERBATIM BEGIN csrc/trtllm_low_latency_gemm_runner.cu:40-222
struct TrtllmLowLatencyGemmRunnerOptions {
  gemm::trtllm::gen::Dtype eltType;
  gemm::trtllm::gen::Dtype outputType;
};

gemm::gemm::GemmData createGemmData(int64_t m, int64_t n, int64_t k) {
  gemm::gemm::GemmData gemmData{};

  // Dims
  gemmData.mProblemDimensions.mM = n;
  gemmData.mProblemDimensions.mN = m;
  gemmData.mProblemDimensions.mK = k;
  gemmData.mProblemDimensions.mValidM = gemmData.mProblemDimensions.mM;
  gemmData.mProblemDimensions.mValidN = gemmData.mProblemDimensions.mN;
  gemmData.mProblemDimensions.mValidK = gemmData.mProblemDimensions.mK;
  gemmData.mProblemDimensions.mRank = 0;
  gemmData.mProblemDimensions.mWorldSize = 1;

  return gemmData;
}

/**
 * Very rough heuristic for selecting a kernel. Prefer using auto-tuning.
 */
int64_t select_kernel(int32_t m, int32_t n, int32_t k, const gemm::gemm::GemmInterface& interface) {
  static constexpr const char* KERNEL_MMAN_8_TILEK_128 =
      "gemm_Bfloat16_E4m3E4m3_Fp32_t128x8x128_s7_et128x8_m128x8x32_c1x1x1_16dp256b_rM_BN_"
      "transOut_schedS_sm100f";
  static constexpr const char* KERNEL_MMAN_8_TILEK_256 =
      "gemm_Bfloat16_E4m3E4m3_Fp32_t128x8x256_s4_et128x8_m128x8x32_c1x1x1_16dp256b_rM_BN_"
      "transOut_schedS_sm100f";
  static constexpr const char* KERNEL_MMAN_16_TILEK_128 =
      "gemm_Bfloat16_E4m3E4m3_Fp32_t128x64x128_s7_et128x32_m128x64x32_c1x1x1_16dp256b_rM_BN_"
      "transOut_schedS_sm100f";
  static constexpr const char* KERNEL_MMAN_16_TILEK_256 =
      "gemm_Bfloat16_E4m3E4m3_Fp32_t128x64x256_s3_et128x32_m128x64x32_c1x1x1_16dp256b_rM_BN_"
      "transOut_schedS_sm100f";
  static constexpr const char* KERNEL_MMAN_32_TILEK_128 =
      "gemm_Bfloat16_E4m3E4m3_Fp32_t128x32x128_s9_et128x32_m128x32x32_c1x1x1_16dp256b_rM_BN_"
      "transOut_schedS_sm100f";
  static constexpr const char* KERNEL_MMAN_32_TILEK_256 =
      "gemm_Bfloat16_E4m3E4m3_Fp32_t128x32x256_s5_et128x32_m128x32x32_c1x1x1_16dp256b_rM_BN_"
      "transOut_schedS_sm100f";
  static constexpr const char* KERNEL_MMAN_64_TILEK_128 =
      "gemm_Bfloat16_E4m3E4m3_Fp32_t128x16x128_s7_et128x16_m128x16x32_c1x1x1_16dp256b_rM_BN_"
      "transOut_schedS_sm100f";
  static constexpr const char* KERNEL_MMAN_64_TILEK_256 =
      "gemm_Bfloat16_E4m3E4m3_Fp32_t128x16x256_s5_et128x16_m128x16x32_c1x1x1_16dp256b_rM_BN_"
      "transOut_schedS_sm100f";

  std::string kernel_name;
  if (m <= 8) {
    kernel_name = KERNEL_MMAN_8_TILEK_128;
  } else if (m <= 16) {
    kernel_name = KERNEL_MMAN_16_TILEK_128;
  } else if (m <= 32) {
    kernel_name = KERNEL_MMAN_32_TILEK_128;
  } else {
    kernel_name = KERNEL_MMAN_64_TILEK_128;
  }

  auto const& configs = interface.getGemmConfigs();
  size_t const num_configs = interface.getNumGemmConfigs();

  for (size_t i = 0; i < num_configs; ++i) {
    if (std::string(configs[i].mFunctionName) == kernel_name) {
      return static_cast<int64_t>(i);
    }
  }

  TVM_FFI_LOG_AND_THROW(RuntimeError)
      << "No kernel was found heuristically for the given problem size";
}

int64_t getWorkspaceSizeInBytes(int64_t m, int64_t n, int64_t k, int64_t tactic) {
  auto gemm = gemm::gemm::GemmInterface();

  if (tactic == -1) {
    tactic = select_kernel(m, n, k, gemm);
  }

  auto const configs = gemm.getGemmConfigs();
  FLASHINFER_CHECK(tactic >= 0 && tactic < gemm.getNumGemmConfigs(),
                   "Invalid tactic in getWorkspaceSizeInBytes");
  auto const config = configs[tactic];

  auto const gemmData = createGemmData(m, n, k);

  return gemm.getWorkspaceSizeInBytes(config, gemmData);
}

class TrtllmLowLatencyGemmRunner {
 public:
  explicit TrtllmLowLatencyGemmRunner(TrtllmLowLatencyGemmRunnerOptions const& options)
      : mOptions(options) {
    // Select a GEMM kernel config to use
    auto const gemm = gemm::gemm::GemmInterface();
    auto const configs = gemm.getGemmConfigs();

    mPassingConfigIndices.clear();

    for (size_t i = 0; i < gemm.getNumGemmConfigs(); ++i) {
      auto const configOptions = configs[i].mOptions;

      if (configOptions.mDtypeA == mOptions.eltType &&
          configOptions.mDtypeC == mOptions.outputType &&
          configOptions.mTransposeMmaOutput == true &&
          configOptions.mLayoutA == gemm::gemm::MatrixLayout::BlockMajorK &&
          configOptions.mUseShuffledMatrix) {
        mPassingConfigIndices.push_back(i);
      }
    }

    FLASHINFER_CHECK(
        mPassingConfigIndices.size() > 0,
        "No valid low latency TRTLLM-GEN GEMM kernel was found for the given data types.");
  }

  void run(int64_t m, int64_t n, int64_t k, void const* a, void const* b, void* c, void* cScale,
           void* workspace, CUstream stream, int32_t device_index, int64_t tactic) {
    auto gemm = gemm::gemm::GemmInterface();
    auto const configs = gemm.getGemmConfigs();
    TVM_FFI_ICHECK(tactic >= 0 && tactic < gemm.getNumGemmConfigs()) << "Invalid tactic id in run";
    auto const& config = configs[tactic];

    gemm::gemm::GemmData gemmData = createGemmData(m, n, k);

    // Inputs
    gemmData.mInputBuffers.mPtrA = b;
    gemmData.mInputBuffers.mPtrB = a;
    gemmData.mInputBuffers.mPtrScaleC = cScale;

    // Outputs
    gemmData.mOutputBuffers.mPtrC = c;

    TVM_FFI_ICHECK(gemm.isValidConfig(config, gemmData))
        << "The selected tactic points to a TRTLLM-GEN low latency GEMM kernel that is not valid "
           "for "
           "the given problem size.";

    int32_t const multiProcessorCount = [device_index]() {
      static thread_local int32_t cached_multi_processor_count = -1;
      static thread_local int cached_device_index = -1;

      if (device_index == cached_device_index && cached_multi_processor_count != -1) {
        return cached_multi_processor_count;
      } else {
        int32_t count;
        cudaError_t cudaStatus =
            cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device_index);
        TVM_FFI_ICHECK(cudaStatus == cudaSuccess)
            << "Failed to get device attribute: " << cudaGetErrorString(cudaStatus);
        cached_multi_processor_count = count;
        cached_device_index = device_index;
        return count;
      }
    }();

    TVM_FFI_ICHECK(gemm.run(config, workspace, gemmData, static_cast<void*>(stream),
                            multiProcessorCount, true, globalTrtllmLowLatencyGemmModuleCache) == 0)
        << "Error occurred when running low latency TRTLLM-GEN GEMM!";
  }

  std::vector<int64_t> getValidTactics(int64_t m, int64_t n, int64_t k) const {
    auto const gemm = gemm::gemm::GemmInterface();
    auto const configs = gemm.getGemmConfigs();

    auto const gemmData = createGemmData(m, n, k);

    std::vector<int64_t> validTactics{};
    for (auto const& configIndex : mPassingConfigIndices) {
      auto const& config = configs[configIndex];
      if (gemm.isValidConfig(config, gemmData)) {
        validTactics.push_back(configIndex);
      }
    }
    return validTactics;
  }

 private:
  TrtllmLowLatencyGemmRunnerOptions mOptions;
  std::vector<int64_t> mPassingConfigIndices;
};
// VERBATIM END
}  // namespace flashinfer
