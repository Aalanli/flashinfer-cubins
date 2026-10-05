// DSA attention, sm_86 stage 2/3: upstream FlashInfer MLA paged decode.
//
// The kernel body is unmodified upstream FlashInfer
// (include/flashinfer/attention/decode.cuh, Apache-2.0, see ../LICENSE), included
// from the pinned checkout under resources/. This translation unit contains exactly
// one explicit instantiation: the configuration the previous host code launched
// (BatchDecodeWithPagedKVCacheKernelMLA<2, 16, 2, 32, 8, 1, 2, ...>, BF16, int32 IDs,
// non-partitioned, no RoPE/soft-cap/sliding window). Launch: grid = (batch, 1, 1),
// block = (32, 8, 1), dynamic smem = DSA_DECODE_SMEM_BYTES.
#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/attention/decode.cuh>

namespace flashinfer {
using DsaDecodeParams = BatchDecodeParamsMLA<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, int>;
template __global__ void BatchDecodeWithPagedKVCacheKernelMLA<
    2, 16, 2, 32, 8, 1, 2, DefaultAttention<false, false, false, false>, DsaDecodeParams>(
    DsaDecodeParams params);
}  // namespace flashinfer
