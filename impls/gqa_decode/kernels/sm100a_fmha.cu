// GQA paged decode, sm_100a stage 3/4: upstream CUTLASS SM100 FMHA forward (QK and PV UMMA).
//
// The kernel is unmodified upstream code (see sm100a_fmha.cuh). This translation unit
// explicitly instantiates exactly one kernel, cutlass::device_kernel<gqa_decode::Kernel>, and
// contains no host launch code; FwdRunner::run's host logic is reproduced in Python from the
// layout the host-only probe (probe_sm100a_fmha_params.cu) records.
#include "sm100a_fmha.cuh"

template __global__ void cutlass::device_kernel<gqa_decode::Kernel>(
    CUTLASS_GRID_CONSTANT gqa_decode::Kernel::Params const);
