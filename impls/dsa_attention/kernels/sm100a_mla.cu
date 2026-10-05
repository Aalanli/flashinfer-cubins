// DSA attention, sm_100a stage 2/3: upstream CUTLASS SM100 MLA decode (QK and PV UMMA).
//
// The kernel is unmodified upstream code: FlashInfer include/flashinfer/attention/
// cutlass_mla.cuh (Apache-2.0, see ../LICENSE), which wraps CUTLASS example
// 77_blackwell_fmha Sm100FmhaMlaKernelTmaWarpspecialized (BSD-3-Clause, see
// ../CUTLASS_LICENSE), included from the pinned checkouts under resources/.
// This translation unit explicitly instantiates exactly one kernel:
// cutlass::device_kernel<MlaSm100<bfloat16_t>::FmhaKernel> (persistent tile scheduler,
// cp.async page-table loads). The split-KV reduction kernel of cutlass::fmha::device::MLA
// is intentionally not instantiated: the workload always runs with split_kv = 1.
#include <flashinfer/attention/cutlass_mla.cuh>

using DsaMlaKernel = flashinfer::attention::MlaSm100<cutlass::bfloat16_t>::FmhaKernel;
template __global__ void cutlass::device_kernel<DsaMlaKernel>(
    CUTLASS_GRID_CONSTANT DsaMlaKernel::Params const);
