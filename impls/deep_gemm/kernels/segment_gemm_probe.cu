// SPDX-License-Identifier: Apache-2.0
// Compile-time probe for the sm_86 deep_gemm workloads: FlashInfer's CUTLASS
// segment GEMM (include/flashinfer/gemm/group_gemm.cuh, CutlassSegmentGEMMRun)
// with column-major ("NT", K-major) weights.
//
// Built and run by impls/deep_gemm/compiler.py; never part of a benchmark run.
// The executable launches nothing and needs no GPU. It instantiates
// cutlass::Kernel<GemmKernel> for the four column-major-weight kernels exactly
// as group_gemm.cuh's DefaultGemmGrouped (DType in {bf16, fp16}, NUM_STAGES in
// {2, 4}); the device code embedded in the executable (-gencode sm_80) lets
// the compiler confirm that the mangled symbols equal the kernels stripped
// from the FlashInfer JIT-cache cubin. Running it prints one JSON document:
// per kernel, sizeof/alignof/offsetof of every GemmGrouped::Params field, the
// launch constants (kThreadCount, sizeof(SharedStorage), stages, workspace)
// and reference Params bytes built the way GemmGrouped::initialize builds
// them (BaseKernel::Params(args, workspace) for the kDeviceOnly schedule)
// from Arguments filled as CutlassSegmentGEMMRun fills them, with fake
// (never dereferenced) device addresses.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>
#include <string>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/epilogue/thread/linear_combination.h"

namespace {

template <typename DType, int NUM_STAGES>
using SegmentGemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    DType, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,  //
    DType, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,  //
    DType, cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<DType, 8, float, float>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    NUM_STAGES>::GemmKernel;

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

#define FIELD_OFFSET(member)                                                     \
  reinterpret_cast<size_t>(&reinterpret_cast<char const volatile&>(              \
      reinterpret_cast<Params*>(0)->member))
#define FIELD_SIZE(member) sizeof(reinterpret_cast<Params*>(0)->member)

// Every GemmGrouped::Params field the Python builder fills.
#define PARAMS_FIELDS(F)                                                         \
  F("problem_visitor.problem_sizes", problem_visitor.problem_sizes)               \
  F("problem_visitor.problem_count", problem_visitor.problem_count)               \
  F("problem_visitor.workspace", problem_visitor.workspace)                       \
  F("problem_visitor.tile_count", problem_visitor.tile_count)                     \
  F("threadblock_count", threadblock_count)                                       \
  F("output_op.alpha", output_op.alpha)                                           \
  F("output_op.beta", output_op.beta)                                             \
  F("output_op.alpha_ptr", output_op.alpha_ptr)                                   \
  F("output_op.beta_ptr", output_op.beta_ptr)                                     \
  F("output_op.alpha_ptr_array", output_op.alpha_ptr_array)                       \
  F("output_op.beta_ptr_array", output_op.beta_ptr_array)                         \
  F("ptr_A", ptr_A)                                                               \
  F("ptr_B", ptr_B)                                                               \
  F("ptr_C", ptr_C)                                                               \
  F("ptr_D", ptr_D)                                                               \
  F("lda", lda)                                                                   \
  F("ldb", ldb)                                                                   \
  F("ldc", ldc)                                                                   \
  F("ldd", ldd)

#define PRINT_FIELD(name, member)                                                \
  std::printf("%s\"%s\": [%zu, %zu]", first ? "" : ", ", name,                    \
              FIELD_OFFSET(member), FIELD_SIZE(member));                         \
  first = false;
#define COVER_FIELD(name, member)                                                \
  std::memset(covered + FIELD_OFFSET(member), 1, FIELD_SIZE(member));

template <typename DType, int NUM_STAGES>
void probe(const char* name, bool last) {
  using GemmKernel = SegmentGemmKernel<DType, NUM_STAGES>;
  using Params = typename GemmKernel::Params;
  using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;
  static_assert(GemmKernel::kGroupScheduleMode ==
                    cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly,
                "segment GEMM uses the device-only group schedule");
  static_assert(!GemmKernel::ProblemVisitor::kRequiresPrecomputation,
                "no host precomputation");
  // Reference the kernel so that its device code is instantiated.
  volatile auto kernel = &cutlass::Kernel<GemmKernel>;
  (void)kernel;

  std::printf("%s\"%s\": {\"params_size\": %zu, \"params_align\": %zu, ", last ? "" : "",
              name, sizeof(Params), alignof(Params));
  std::printf("\"threads\": %d, \"shared_storage\": %zu, \"stages\": %d, ",
              GemmKernel::kThreadCount, sizeof(typename GemmKernel::SharedStorage),
              GemmKernel::Mma::kStages);
  std::printf("\"gemm_coord_size\": %zu, \"fields\": {", sizeof(cutlass::gemm::GemmCoord));
  bool first = true;
  PARAMS_FIELDS(PRINT_FIELD)
  std::printf("}, \"examples\": [");
  // The constructor copy-assigns sub-structs (problem visitor, epilogue
  // params) from temporaries, which carries their indeterminate padding into
  // Params. Zero every byte outside a field so the fixtures are reproducible
  // (the tests compare fields only).
  bool covered[sizeof(Params)] = {};
  PARAMS_FIELDS(COVER_FIELD)

  // CutlassSegmentGEMMRun: Arguments(all_problems, batch_size,
  // threadblock_count = 4, epilogue_op(1.0, 1.0), x, w, y, y, x_ld, w_ld,
  // y_ld, y_ld), then GemmGrouped::initialize(args, nullptr, stream).
  const int counts[] = {1, 3, 64};
  for (int e = 0; e < 3; ++e) {
    const uintptr_t base = 0x7f0000000000ull + 0x1000000ull * e;
    typename GemmKernel::EpilogueOutputOp::Params epilogue_op(1.0, 1.0);
    typename GemmGrouped::Arguments args(
        reinterpret_cast<cutlass::gemm::GemmCoord*>(base + 0x000), counts[e],
        /*threadblock_count=*/4, epilogue_op, reinterpret_cast<DType**>(base + 0x1000),
        reinterpret_cast<DType**>(base + 0x2000), reinterpret_cast<DType**>(base + 0x3000),
        reinterpret_cast<DType**>(base + 0x3000), reinterpret_cast<int64_t*>(base + 0x4000),
        reinterpret_cast<int64_t*>(base + 0x5000), reinterpret_cast<int64_t*>(base + 0x6000),
        reinterpret_cast<int64_t*>(base + 0x6000));
    size_t workspace = GemmGrouped::get_workspace_size(args);
    alignas(Params) unsigned char storage[sizeof(Params)];
    std::memset(storage, 0, sizeof(storage));
    Params* params = new (storage) Params(args, nullptr);
    unsigned char bytes[sizeof(Params)];
    std::memcpy(bytes, storage, sizeof(Params));
    for (size_t i = 0; i < sizeof(Params); ++i) {
      if (!covered[i]) bytes[i] = 0;
    }
    std::printf("%s{\"base\": %llu, \"problem_count\": %d, \"workspace\": %zu, \"bytes\": \"%s\"}",
                e ? ", " : "", static_cast<unsigned long long>(base), counts[e], workspace,
                hex(bytes, sizeof(Params)).c_str());
    params->~Params();
  }
  std::printf("]}%s", last ? "" : ", ");
}

}  // namespace

int main() {
  std::printf("{");
  probe<cutlass::bfloat16_t, 2>("bf16_pipelined", false);
  probe<cutlass::bfloat16_t, 4>("bf16_multistage", false);
  probe<cutlass::half_t, 2>("fp16_pipelined", false);
  probe<cutlass::half_t, 4>("fp16_multistage", true);
  std::printf("}\n");
  return 0;
}
