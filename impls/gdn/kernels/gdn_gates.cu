// Project-local glue of the GDN pipeline (not FlashInfer code).
//
// FlashInfer's Blackwell GDN prefill kernel takes the forget gate g and the
// update gate beta as contiguous FP32 [tokens, Hv] tensors and int32
// cu_seqlens; the official definitions (resources/gdn_decode.json,
// resources/gdn_prefill.json) supply the raw projections instead:
//
//   gate[t, h] = exp(-exp(A_log[h]) * softplus(a[t, h] + dt_bias[h]))
//   beta[t, h] = sigmoid(b[t, h])
//   offsets    = int32(cu) for prefill's int64 cu_seqlens, or 0..seqs when cu
//                is null (decode: B one-token sequences)
//
// softplus uses torch's default threshold (x > 20 returns x); IEEE expf/log1pf
// (built without fast math). One thread per (token, head) element; the first
// seqs + 1 threads also write the offsets. Hv = 8.
// Launch: grid = ceil(max(tokens * 8, seqs + 1) / 256), block = 256.
//
// The kernel body is the former gdn_decode package's prepare_gates (copied
// unchanged from the previous package's impls/templates/gdn_prefill.cu; same
// SASS), the superset of the former gdn_prefill package's copy, which lacked
// the cu == nullptr branch. Both had this symbol and parameter layout.
#include <cstdint>
#include <cuda_bf16.h>

__global__ void prepare_gates(const float* alog, const __nv_bfloat16* a, const float* bias,
    const __nv_bfloat16* b, const int64_t* cu, float* gate, float* beta, int* offsets, int tokens, int seqs) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < tokens * 8) {
    int h = i % 8;
    float x = float(a[i]) + bias[h];
    float sp = x > 20.f ? x : log1pf(expf(x));
    gate[i] = expf(-expf(alog[h]) * sp);
    beta[i] = 1.f / (1.f + expf(-float(b[i])));
  }
  if (i <= seqs) offsets[i] = cu ? int(cu[i]) : i;
}
