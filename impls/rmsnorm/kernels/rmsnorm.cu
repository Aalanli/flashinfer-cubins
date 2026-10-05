// RMSNorm (hidden 7168, BF16): upstream FlashInfer norm::RMSNormKernel.
//
// The kernel body is unmodified upstream FlashInfer (include/flashinfer/norm.cuh,
// Apache-2.0, see ../LICENSE), included from the pinned checkout under resources/.
// This translation unit contains exactly one explicit instantiation: the
// configuration FlashInfer's host norm::RMSNorm selects for d = 7168 and BF16
// (vec_size = gcd(16 / sizeof(bf16), 7168) = 8 -> RMSNormKernel<8, __nv_bfloat16>).
// Launch (mirrored in harness/workloads/rmsnorm.py, checked against
// probe_rmsnorm_launch.cu): grid = (batch, 1, 1), block = (32, 28, 1),
// dynamic smem = 28 * sizeof(float) = 112 bytes, weight_bias = 0, eps = 1e-6,
// programmatic stream serialization (PDL) enabled on sm_90+ as in
// flashinfer.norm.rmsnorm(enable_pdl=None).
#include <flashinfer/norm.cuh>

namespace flashinfer {
namespace norm {
template __global__ void RMSNormKernel<8u, __nv_bfloat16>(
    __nv_bfloat16* __restrict__ input, __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output, const uint32_t d, const uint32_t stride_input,
    const uint32_t stride_output, float weight_bias, float eps);
}  // namespace norm
}  // namespace flashinfer
