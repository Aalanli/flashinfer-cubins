// GQA paged decode, sm_86: upstream FlashInfer BatchDecodeWithPagedKVCacheKernel.
//
// The kernel body is unmodified upstream FlashInfer (include/flashinfer/attention/decode.cuh,
// Apache-2.0, see ../LICENSE), included from the pinned checkout under resources/. This
// translation unit holds exactly one explicit instantiation: the configuration the previous
// native host code launched (impls/templates/gqa_decode.cu): no positional encoding, 2 shared
// memory stages, tile_size_per_bdx 1, vec_size 8, bdx 16, bdy 8 (GQA group), bdz 1, default
// attention variant (no custom mask / sliding window / soft cap / alibi), BF16 Q/KV/O, int32
// page IDs, non-partitioned (partition_kv = false). Launch: grid = (batch, 4 KV heads),
// block = (16, 8, 1), dynamic smem = GQA_DECODE_SMEM_BYTES (recorded by the probe).
#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/attention/decode.cuh>

namespace flashinfer {
using GqaDecodeParams = BatchDecodeParams<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, int>;
template __global__ void BatchDecodeWithPagedKVCacheKernel<
    PosEncodingMode::kNone, 2, 1, 8, 16, 8, 1, DefaultAttention<false, false, false, false>,
    GqaDecodeParams>(const __grid_constant__ GqaDecodeParams params);
}  // namespace flashinfer
