// Host-only layout probe helpers (compile stage only; never part of a cubin).
// A probe constructs a kernel's by-value parameter struct with upstream host code and prints
// JSON describing field offsets/sizes/kinds plus the raw bytes, so Python can rebuild the
// struct at run time without native host code and tests can compare it byte for byte.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

namespace gqa_probe {

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
  else if constexpr (std::is_same_v<T, int32_t>) return "i32";
  else if constexpr (std::is_same_v<T, uint32_t>) return "u32";
  else if constexpr (std::is_same_v<T, int64_t>) return "i64";
  else return nullptr;
}

// A scalar member; its kind is derived from its type.
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

inline void print_fields() {
  std::printf("  \"fields\": {\n");
  for (size_t i = 0; i < fields().size(); ++i) {
    auto& f = fields()[i];
    std::printf("    \"%s\": {\"offset\": %zu, \"size\": %zu, \"kind\": \"%s\"}%s\n",
                f.name.c_str(), f.offset, f.size, f.kind.c_str(),
                i + 1 < fields().size() ? "," : "");
  }
  std::printf("  }");
}

// Distinct, 256-byte aligned fake device addresses; never dereferenced.
constexpr uint64_t kSentinelStride = 0x100000000000ull;
inline void* sentinel(int index) {
  return reinterpret_cast<void*>(uintptr_t(kSentinelStride) * uintptr_t(index + 1));
}

}  // namespace gqa_probe
