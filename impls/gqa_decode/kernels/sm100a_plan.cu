// GQA paged decode, sm_100a stage 1/4: host-precomputed tile-scheduler metadata.
//
// Project-local adapter kernel (moved verbatim from impls/templates/gqa_decode_sm100a.cu); not
// upstream code. Writes, for the CUTLASS FMHA HostPrecomputedTileScheduler, the per-request
// query offsets (one query token per request), ragged KV offsets (empty requests keep one dummy
// row) and a balanced split of the batch * 32 (query head, request) work items over `sms` CTAs.
// The kernel is deliberately serial (<<<1, 1>>>): it is metadata preparation only.
__global__ void gqa_plan(const int* indptr, int* qo, int* kv, int* work,
                         int* tiles, int* heads, int* batches, int batch, int sms) {
  if (threadIdx.x == 0) {
    kv[0] = 0;
    for (int b = 0; b < batch; ++b) {
      qo[b] = b;
      kv[b+1] = kv[b] + max(1, indptr[b+1]-indptr[b]);
    }
    qo[batch] = batch;
    // FwdRunner indexes the flattened (GQA group, KV head) axis: all 32 Q heads.
    for (int i = 0; i <= sms; ++i) work[i] = int(int64_t(i)*batch*32/sms);
    for (int i = 0; i < batch*32; ++i) { tiles[i] = 0; heads[i] = i%32; batches[i] = i/32; }
  }
}
