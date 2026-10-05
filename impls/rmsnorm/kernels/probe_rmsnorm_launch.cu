// Host-only probe of FlashInfer's norm::RMSNorm host launcher (no GPU needed).
//
// norm::RMSNorm<__nv_bfloat16> (include/flashinfer/norm.cuh, unmodified) is
// compiled with cudaFuncSetAttribute and cudaLaunchKernelEx redirected to the
// recorders below, so running it with fake device pointers records exactly the
// launch the upstream host code performs: the selected kernel instantiation
// (VEC_SIZE), grid, block, dynamic shared memory, launch attributes (PDL), the
// max-dynamic-smem attribute it sets, and the kernel parameter buffer bytes
// (each argument converted to the kernel's parameter type at its ABI offset).
//
// usage: probe_rmsnorm_launch BATCH HIDDEN STRIDE_INPUT STRIDE_OUTPUT ENABLE_PDL EPS
// Fake pointers: input/weight/output = 0x100000000000 * (1, 2, 3).
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace probe {
struct Launch {
  int called = 0;
  const void* kernel = nullptr;
  cudaLaunchConfig_t config{};
  std::vector<cudaLaunchAttribute> attrs;
  std::vector<unsigned char> params;
  std::vector<int> param_offsets;
  std::vector<int> param_sizes;
};
struct Attribute {
  const void* kernel;
  int attr;
  int value;
};
Launch launch;
std::vector<Attribute> attributes;

template <typename P, typename A>
void append(std::vector<unsigned char>& buffer, A&& value) {
  P converted = static_cast<P>(value);
  size_t offset = (buffer.size() + alignof(P) - 1) / alignof(P) * alignof(P);
  buffer.resize(offset + sizeof(P));
  std::memcpy(buffer.data() + offset, &converted, sizeof(P));
  launch.param_offsets.push_back(static_cast<int>(offset));
  launch.param_sizes.push_back(static_cast<int>(sizeof(P)));
}
}  // namespace probe

template <typename T>
cudaError_t probe_cudaFuncSetAttribute(T* entry, cudaFuncAttribute attr, int value) {
  probe::attributes.push_back({reinterpret_cast<const void*>(entry), int(attr), value});
  return cudaSuccess;
}

template <typename... KernelArgs, typename... Args>
cudaError_t probe_cudaLaunchKernelEx(const cudaLaunchConfig_t* config,
                                     void (*kernel)(KernelArgs...), Args&&... args) {
  static_assert(sizeof...(KernelArgs) == sizeof...(Args), "argument count");
  probe::launch.called += 1;
  probe::launch.kernel = reinterpret_cast<const void*>(kernel);
  probe::launch.config = *config;
  probe::launch.attrs.assign(config->attrs, config->attrs + config->numAttrs);
  (probe::append<KernelArgs>(probe::launch.params, std::forward<Args>(args)), ...);
  return cudaSuccess;
}

#define cudaFuncSetAttribute probe_cudaFuncSetAttribute
#define cudaLaunchKernelEx probe_cudaLaunchKernelEx
#include <flashinfer/norm.cuh>
#undef cudaLaunchKernelEx
#undef cudaFuncSetAttribute

using flashinfer::norm::RMSNormKernel;
using T = __nv_bfloat16;

int main(int argc, char** argv) {
  if (argc != 7) {
    std::fprintf(stderr, "usage: %s BATCH HIDDEN STRIDE_IN STRIDE_OUT PDL EPS\n", argv[0]);
    return 2;
  }
  uint32_t batch = std::strtoul(argv[1], nullptr, 10);
  uint32_t hidden = std::strtoul(argv[2], nullptr, 10);
  uint32_t stride_input = std::strtoul(argv[3], nullptr, 10);
  uint32_t stride_output = std::strtoul(argv[4], nullptr, 10);
  bool enable_pdl = std::atoi(argv[5]) != 0;
  float eps = std::strtof(argv[6], nullptr);
  const uintptr_t sentinel = 0x100000000000ull;
  T* input = reinterpret_cast<T*>(sentinel * 1);
  T* weight = reinterpret_cast<T*>(sentinel * 2);
  T* output = reinterpret_cast<T*>(sentinel * 3);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(uintptr_t(0x5157));

  cudaError_t status = flashinfer::norm::RMSNorm<T>(input, weight, output, batch, hidden,
                                                    stride_input, stride_output, eps,
                                                    enable_pdl, stream);
  if (status != cudaSuccess || probe::launch.called != 1) {
    std::fprintf(stderr, "RMSNorm status %d, %d launches\n", int(status), probe::launch.called);
    return 1;
  }
  const void* candidates[] = {
      reinterpret_cast<const void*>(&RMSNormKernel<1, T>),
      reinterpret_cast<const void*>(&RMSNormKernel<2, T>),
      reinterpret_cast<const void*>(&RMSNormKernel<4, T>),
      reinterpret_cast<const void*>(&RMSNormKernel<8, T>),
      reinterpret_cast<const void*>(&RMSNormKernel<16, T>),
  };
  const int vec_sizes[] = {1, 2, 4, 8, 16};
  int vec_size = 0;
  for (int i = 0; i < 5; ++i) {
    if (candidates[i] == probe::launch.kernel) vec_size = vec_sizes[i];
  }
  const auto& c = probe::launch.config;
  std::printf("{\"batch\": %u, \"hidden\": %u, \"stride_input\": %u, \"stride_output\": %u,",
              batch, hidden, stride_input, stride_output);
  std::printf(" \"enable_pdl\": %s, \"eps\": %.9g,", enable_pdl ? "true" : "false", eps);
  std::printf(" \"pointers\": {\"input\": %llu, \"weight\": %llu, \"output\": %llu},",
              (unsigned long long)(uintptr_t)input, (unsigned long long)(uintptr_t)weight,
              (unsigned long long)(uintptr_t)output);
  std::printf(" \"vec_size\": %d, \"grid\": [%u, %u, %u], \"block\": [%u, %u, %u],", vec_size,
              c.gridDim.x, c.gridDim.y, c.gridDim.z, c.blockDim.x, c.blockDim.y, c.blockDim.z);
  std::printf(" \"shared_mem\": %zu, \"stream_passed\": %s,", c.dynamicSmemBytes,
              c.stream == stream ? "true" : "false");
  std::printf(" \"launch_attributes\": [");
  for (size_t i = 0; i < probe::launch.attrs.size(); ++i) {
    const auto& a = probe::launch.attrs[i];
    int value = a.id == cudaLaunchAttributeProgrammaticStreamSerialization
                    ? int(a.val.programmaticStreamSerializationAllowed)
                    : -1;
    std::printf("%s{\"id\": %d, \"value\": %d}", i ? ", " : "", int(a.id), value);
  }
  std::printf("], \"func_attributes\": [");
  for (size_t i = 0; i < probe::attributes.size(); ++i) {
    const auto& a = probe::attributes[i];
    std::printf("%s{\"attr\": %d, \"value\": %d, \"same_kernel\": %s}", i ? ", " : "",
                a.attr, a.value, a.kernel == probe::launch.kernel ? "true" : "false");
  }
  std::printf("], \"params\": [");
  for (size_t i = 0; i < probe::launch.param_offsets.size(); ++i) {
    std::printf("%s{\"offset\": %d, \"size\": %d}", i ? ", " : "",
                probe::launch.param_offsets[i], probe::launch.param_sizes[i]);
  }
  std::printf("], \"params_hex\": \"");
  for (unsigned char byte : probe::launch.params) std::printf("%02x", byte);
  std::printf("\"}\n");
  return 0;
}
