// GQA paged decode, sm_100a: the CUTLASS SM100 FMHA kernel type FlashInfer instantiates.
//
// flashinfer::FwdRunner (include/flashinfer/attention/blackwell/fmha_cutlass_sm100.cuh,
// Apache-2.0, see ../LICENSE) over CUTLASS example 77_blackwell_fmha
// Sm100FmhaFwdKernelTmaWarpspecialized (BSD-3-Clause, see ../CUTLASS_LICENSE), with the
// template arguments the previous native host code used (impls/templates/gqa_decode_sm100a.cu):
// BF16 in/out, int32 offsets, QK and PV tiles 256x128x128, ResidualMask, varlen Q/KV problem
// shape and the host-precomputed tile scheduler. Shared by the kernel TU and the host probe.
#pragma once
#include <flashinfer/attention/blackwell/fmha_cutlass_sm100.cuh>

namespace gqa_decode {
using Element = cutlass::bfloat16_t;
using Runner = flashinfer::FwdRunner<Element, Element, int, cute::Shape<cute::_256, cute::_128, cute::_128>,
                                     cute::Shape<cute::_256, cute::_128, cute::_128>,
                                     cutlass::fmha::collective::ResidualMask>;
using Operation = Runner::Operation;
using Kernel = Operation::Kernel;
}  // namespace gqa_decode
