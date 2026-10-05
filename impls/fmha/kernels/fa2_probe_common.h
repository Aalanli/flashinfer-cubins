// Host-only probe helpers for the fmha FA2 (sm_86) layout probes; compile stage only.
//
// A probe constructs a kernel's by-value parameter struct with upstream host code (fake,
// distinct device pointers) and prints JSON: field offsets/sizes/kinds plus the raw bytes,
// so Python can rebuild the struct at run time and tests can compare byte for byte.
//
// The planners and launchers query the device (cudaGetDevice / cudaDeviceGetAttribute). The
// probes run without a GPU on a fixed sm_86 (RTX 3090) model: these two calls are redirected
// to the functions below before any FlashInfer header is included.
#pragma once
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

namespace fa2_probe {

// RTX 3090 (GA102, sm_86): 82 SMs, 100 KiB shared memory per SM, 99 KiB opt-in per block.
constexpr int kNumSms = 82;
constexpr int kMaxSmemPerSm = 102400;
constexpr int kMaxSmemPerBlockOptin = 101376;

inline cudaError_t get_device(int* device) {
  *device = 0;
  return cudaSuccess;
}

inline cudaError_t device_get_attribute(int* value, cudaDeviceAttr attr, int) {
  switch (attr) {
    case cudaDevAttrMultiProcessorCount: *value = kNumSms; return cudaSuccess;
    case cudaDevAttrMaxSharedMemoryPerMultiprocessor: *value = kMaxSmemPerSm; return cudaSuccess;
    case cudaDevAttrMaxSharedMemoryPerBlockOptin: *value = kMaxSmemPerBlockOptin; return cudaSuccess;
    case cudaDevAttrComputeCapabilityMajor: *value = 8; return cudaSuccess;
    case cudaDevAttrComputeCapabilityMinor: *value = 6; return cudaSuccess;
    default: std::fprintf(stderr, "unexpected device attribute %d\n", int(attr)); std::abort();
  }
}

struct Field {
  std::string name, kind;
  size_t offset, size;
};

inline std::vector<Field>& fields() {
  static std::vector<Field> f;
  return f;
}

template <class Base, class Member>
size_t offset_of(Base const& base, Member const& member) {
  return size_t(reinterpret_cast<const char*>(&member) - reinterpret_cast<const char*>(&base));
}

template <class Base, class Member>
void field(std::string const& name, const char* kind, Base const& base, Member const& member) {
  fields().push_back({name, kind, offset_of(base, member), sizeof(Member)});
}

template <class T>
constexpr const char* scalar_kind() {
  if constexpr (std::is_pointer_v<T>) return "ptr";
  else if constexpr (std::is_same_v<T, bool>) return "bool";
  else if constexpr (std::is_same_v<T, float>) return "f32";
  else if constexpr (std::is_same_v<T, double>) return "f64";
  else if constexpr (std::is_same_v<T, int32_t>) return "i32";
  else if constexpr (std::is_same_v<T, uint32_t>) return "u32";
  else if constexpr (std::is_same_v<T, int64_t>) return "i64";
  else return nullptr;
}

template <class Base, class Member>
void scalar(std::string const& name, Base const& base, Member const& member) {
  constexpr const char* kind = scalar_kind<Member>();
  static_assert(kind != nullptr, "unsupported scalar field type");
  field(name, kind, base, member);
}

inline std::string hex(const void* data, size_t size) {
  static const char digits[] = "0123456789abcdef";
  std::string out;
  auto* bytes = static_cast<const unsigned char*>(data);
  for (size_t i = 0; i < size; ++i) {
    out += digits[bytes[i] >> 4];
    out += digits[bytes[i] & 15];
  }
  return out;
}

inline void print_fields(const char* indent = "    ") {
  std::printf("{\n");
  for (size_t i = 0; i < fields().size(); ++i) {
    auto& f = fields()[i];
    std::printf("%s  \"%s\": {\"offset\": %zu, \"size\": %zu, \"kind\": \"%s\"}%s\n", indent,
                f.name.c_str(), f.offset, f.size, f.kind.c_str(),
                i + 1 < fields().size() ? "," : "");
  }
  std::printf("%s}", indent);
  fields().clear();
}

template <class T>
inline void print_vec(std::vector<T> const& v) {
  std::printf("[");
  for (size_t i = 0; i < v.size(); ++i) std::printf("%s%lld", i ? ", " : "", (long long)v[i]);
  std::printf("]");
}

// Distinct, 256-byte aligned fake device addresses; never dereferenced.
constexpr uint64_t kSentinelStride = 0x100000000000ull;
inline void* sentinel(int index) {
  return reinterpret_cast<void*>(uintptr_t(kSentinelStride) * uintptr_t(index + 1));
}

}  // namespace fa2_probe

#define cudaGetDevice fa2_probe::get_device
#define cudaDeviceGetAttribute fa2_probe::device_get_attribute
