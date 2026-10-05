// Host-only layout probe helpers (compile stage only; never part of a cubin).
// A probe constructs a kernel's by-value parameter struct with upstream host code and
// prints JSON describing field offsets/sizes plus the raw bytes, so Python can rebuild
// the struct at run time without native host code.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace dsa_probe {

struct Field {
  std::string name, kind;
  size_t offset, size;
};

inline std::vector<Field>& fields() {
  static std::vector<Field> f;
  return f;
}

template <class Base, class Member>
void field(const char* name, const char* kind, Base const& base, Member const& member) {
  fields().push_back({name, kind,
                      size_t(reinterpret_cast<const char*>(&member) -
                             reinterpret_cast<const char*>(&base)),
                      sizeof(Member)});
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
inline void* sentinel(int index) {
  return reinterpret_cast<void*>(uintptr_t(0x100000000000ull) * uintptr_t(index + 1));
}

}  // namespace dsa_probe
