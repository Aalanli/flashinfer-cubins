// Host-only launch probe for the GDN chunk (dvsplit) kernel (compile stage only;
// never part of a cubin, never touches a GPU).
//
// impls/gdn/compiler.py extracts, verbatim from the pinned upstream TVM-FFI
// host shim csrc/gdn/cake/host/cake_gdn_prefill_dvsplit_initial_373d87a0efe5.cc,
//   * the four EncodeTma_{Q,K,V,O} descriptor builders and
//   * the part of Run() that turns its arguments into `void* kargs[]`,
// into "chunk_upstream.inc" (included below). This file supplies the minimum the
// excerpt needs: a TensorView with fake device addresses, TVM_FFI_CHECK, and a
// recording cuTensorMapEncodeTiled. It then calls the excerpt with the arguments
// FlashInfer's flashinfer/gdn_prefill.py (_run_cake_gdn_prefill) passes for
// <tokens> tokens in <seqs> ragged sequences (decode: tokens == seqs, B
// one-token sequences), Hq = 4, Hv = 8, and prints JSON:
//   * "tma_calls": every cuTensorMapEncodeTiled argument, the address as
//     {role, offset} of the fake pointer it came from;
//   * "bytes": the packed kernel parameter buffer (each kargs[i] copied to the
//     cubin's EIATTR_KPARAM_INFO offset/size given on the command line), with each
//     CUtensorMap replaced by a marker (TMA_MARKER | call index, then zeros).
//
// usage: probe <tokens> <seqs> <scale> <sm_count> <offset:size,offset:size,...>
#include <cuda.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace probe {

constexpr uint64_t kSentinelStride = 0x100000000000ull;
constexpr uint64_t kTmaMarker = 0x7E5A000000000000ull;

const char* const kRoles[] = {"Q",
                              "K",
                              "V",
                              "O",
                              "gate",
                              "beta",
                              "cu_seqlens",
                              "state_indices",
                              "initial_state",
                              "output_state",
                              "checkpoint_state",
                              "cu_checkpoints",
                              "tensormap_workspace"};
constexpr int kNumRoles = sizeof(kRoles) / sizeof(kRoles[0]);

inline void* sentinel(int role) {
  return reinterpret_cast<void*>(uintptr_t(kSentinelStride) * uintptr_t(role + 1));
}

[[noreturn]] inline void fail(const std::string& what) {
  std::fprintf(stderr, "probe: %s\n", what.c_str());
  std::exit(2);
}

struct CheckFailure {
  std::ostringstream message;
  const char* kind;
  explicit CheckFailure(const char* k) : kind(k) {}
  template <class T>
  CheckFailure& operator<<(const T& value) {
    message << value;
    return *this;
  }
  [[noreturn]] ~CheckFailure() { fail(std::string(kind) + ": " + message.str()); }
};

}  // namespace probe

// Contiguous-by-construction view with a fake data pointer (never dereferenced).
class TensorView {
 public:
  TensorView(void* data, std::vector<int64_t> shape) : data_(data), shape_(std::move(shape)) {
    strides_.assign(shape_.size(), 1);
    for (int i = int(shape_.size()) - 2; i >= 0; --i)
      strides_[i] = strides_[i + 1] * shape_[i + 1];
  }
  int ndim() const { return int(shape_.size()); }
  int64_t size(int i) const { return shape_[index(i)]; }
  int64_t stride(int i) const { return strides_[index(i)]; }
  void* data_ptr() const { return data_; }

 private:
  size_t index(int i) const {
    int j = i < 0 ? i + ndim() : i;
    if (j < 0 || j >= ndim()) probe::fail("TensorView index out of range");
    return size_t(j);
  }
  void* data_;
  std::vector<int64_t> shape_, strides_;
};

#define TVM_FFI_CHECK(cond, kind) \
  if (cond) {                     \
  } else                          \
    probe::CheckFailure(#kind)

namespace probe {

struct TmaCall {
  int data_type, rank, role;
  uint64_t offset;
  std::vector<uint64_t> dims, strides;
  std::vector<uint32_t> box, element_strides;
  int interleave, swizzle, l2_promotion, oob_fill;
};

std::vector<TmaCall>& tma_calls() {
  static std::vector<TmaCall> calls;
  return calls;
}

}  // namespace probe

// Recording stub: the probe is linked without libcuda.
CUresult CUDAAPI cuTensorMapEncodeTiled(CUtensorMap* map, CUtensorMapDataType data_type,
                                        cuuint32_t rank, void* address, const cuuint64_t* dims,
                                        const cuuint64_t* strides, const cuuint32_t* box,
                                        const cuuint32_t* element_strides,
                                        CUtensorMapInterleave interleave,
                                        CUtensorMapSwizzle swizzle,
                                        CUtensorMapL2promotion l2_promotion,
                                        CUtensorMapFloatOOBfill oob_fill) {
  uint64_t a = uint64_t(reinterpret_cast<uintptr_t>(address));
  probe::TmaCall call;
  call.data_type = int(data_type);
  call.rank = int(rank);
  call.role = int(a / probe::kSentinelStride) - 1;
  call.offset = a % probe::kSentinelStride;
  if (call.role < 0 || call.role >= probe::kNumRoles) probe::fail("unknown TMA address");
  call.dims.assign(dims, dims + rank);
  call.strides.assign(strides, strides + (rank - 1));
  call.box.assign(box, box + rank);
  call.element_strides.assign(element_strides, element_strides + rank);
  call.interleave = int(interleave);
  call.swizzle = int(swizzle);
  call.l2_promotion = int(l2_promotion);
  call.oob_fill = int(oob_fill);
  std::memset(map, 0, sizeof(*map));
  uint64_t marker = probe::kTmaMarker | uint64_t(probe::tma_calls().size());
  std::memcpy(map, &marker, sizeof(marker));
  probe::tma_calls().push_back(call);
  return CUDA_SUCCESS;
}

namespace probe {

struct Param {
  size_t offset, size;
};
std::vector<Param> params;
std::vector<unsigned char> buffer;

// Called by the upstream excerpt with its kargs array.
void pack(void** kargs, size_t count) {
  if (count != params.size()) fail("kargs count differs from the cubin parameters");
  size_t total = 0;
  for (auto& p : params) total = p.offset + p.size > total ? p.offset + p.size : total;
  buffer.assign(total, 0);
  for (size_t i = 0; i < count; ++i)
    std::memcpy(buffer.data() + params[i].offset, kargs[i], params[i].size);
}

}  // namespace probe

#include "chunk_upstream.inc"

namespace probe {

template <class T>
void print_list(const std::vector<T>& values) {
  std::printf("[");
  for (size_t i = 0; i < values.size(); ++i)
    std::printf("%s%llu", i ? ", " : "", (unsigned long long)values[i]);
  std::printf("]");
}

}  // namespace probe

int main(int argc, char** argv) {
  using namespace probe;
  if (argc != 6) fail("usage: probe <tokens> <seqs> <scale> <sm_count> <offset:size,...>");
  const int64_t tokens = std::atoll(argv[1]);
  const int64_t seqs = std::atoll(argv[2]);
  const double scale = std::strtod(argv[3], nullptr);
  const int64_t sm_count = std::atoll(argv[4]);
  for (std::stringstream list(argv[5]); list.good();) {
    std::string item;
    std::getline(list, item, ',');
    size_t colon = item.find(':');
    if (colon == std::string::npos) fail("bad parameter layout");
    params.push_back({size_t(std::stoull(item.substr(0, colon))),
                      size_t(std::stoull(item.substr(colon + 1)))});
  }
  if (tokens < 1 || seqs < 1 || sm_count < 1)
    fail("tokens, seqs and sm_count must be positive");

  // flashinfer/gdn_prefill.py _run_cake_gdn_prefill: q/k [T, 4, 128], v/output
  // [T, 8, 128] (contiguous), FP32 g/beta [T, 8], int32 cu_seqlens [N + 1],
  // FP32 initial/output state [N, 8, 128, 128], one-element dummies for the
  // unused state_indices / checkpoint buffers, and a tensormap workspace of
  // grid_x * 512 bytes.
  const int64_t hq = 4, hv = 8, num_o_heads = hv;
  const int64_t total_tiles = seqs * num_o_heads * 2;  // dvsplit route
  const int64_t grid_x = sm_count < total_tiles ? sm_count : total_tiles;
  TensorView q(sentinel(0), {tokens, hq, 128}), k(sentinel(1), {tokens, hq, 128}),
      v(sentinel(2), {tokens, hv, 128}), o(sentinel(3), {tokens, num_o_heads, 128}),
      gate(sentinel(4), {tokens, num_o_heads}), beta(sentinel(5), {tokens, num_o_heads}),
      cu_seqlens(sentinel(6), {seqs + 1}), state_indices(sentinel(7), {1}),
      initial_state(sentinel(8), {seqs, num_o_heads, 128, 128}),
      output_state(sentinel(9), {seqs, num_o_heads, 128, 128}),
      checkpoint_state(sentinel(10), {1}), cu_checkpoints(sentinel(11), {1}),
      workspace(sentinel(12), {grid_x * 512});
  upstream_kargs(q, k, v, o, gate, beta, cu_seqlens, state_indices, initial_state,
                 output_state, checkpoint_state, cu_checkpoints, workspace,
                 initial_state.stride(0), output_state.stride(0),
                 /*checkpoint_every_n_tokens=*/0, scale, seqs, hq, hv, total_tiles, grid_x,
                 1, 1);

  std::printf("{\n  \"tokens\": %lld,\n  \"seqs\": %lld,\n  \"scale\": %.17g,\n"
              "  \"sm_count\": %lld,\n",
              (long long)tokens, (long long)seqs, scale, (long long)sm_count);
  std::printf("  \"grid\": [%lld, 1, 1],\n  \"workspace_bytes\": %lld,\n",
              (long long)grid_x, (long long)(grid_x * 512));
  std::printf("  \"tma_calls\": [\n");
  for (size_t i = 0; i < tma_calls().size(); ++i) {
    const TmaCall& c = tma_calls()[i];
    std::printf("    {\"data_type\": %d, \"rank\": %d, \"address\": {\"role\": \"%s\", "
                "\"offset\": %llu}, \"dims\": ",
                c.data_type, c.rank, kRoles[c.role], (unsigned long long)c.offset);
    print_list(c.dims);
    std::printf(", \"strides\": ");
    print_list(c.strides);
    std::printf(", \"box\": ");
    print_list(c.box);
    std::printf(", \"element_strides\": ");
    print_list(c.element_strides);
    std::printf(", \"interleave\": %d, \"swizzle\": %d, \"l2_promotion\": %d, "
                "\"oob_fill\": %d}%s\n",
                c.interleave, c.swizzle, c.l2_promotion, c.oob_fill,
                i + 1 < tma_calls().size() ? "," : "");
  }
  std::printf("  ],\n  \"bytes\": \"");
  for (unsigned char byte : buffer) std::printf("%02x", byte);
  std::printf("\"\n}\n");
  return 0;
}
