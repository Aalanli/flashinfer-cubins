"""Compile the routed FP8 MoE package into single-kernel cubins.

Workload: ``moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048``
(``resources/moe.json``): DeepSeek-V3 routing (256 experts, top-8, 8 groups /
4 kept), 32 local experts, hidden 7168, intermediate 2048, FP8 E4M3 with
128-element block scales, BF16 output. Upstream: FlashInfer v0.2.10
``trtllm_fp8_block_scale_moe`` (``csrc/trtllm_fused_moe_kernel_launcher.cu``)
with the Python defaults ``tile_tokens_dim = 8``, ``use_shuffled_weight =
False``, ``weight_layout = 0`` (MajorK) the official baseline uses.

Upstream dispatch (host code of the pinned revision, recorded by the probe)
---------------------------------------------------------------------------

``Routing::Runner::run`` -> ``routingDeepSeek::run``::

    routingMainKernel                      <<<T, 256>>>                 always
    routingIndicesClusterKernel            <<<8, 256>>> (8-CTA cluster)  T <= 1024
    routingIndicesCoopKernel               <<<128, 256>>> cooperative    1024 < T <= 262144
    routingIndicesHistogram/OffsetsKernel                               T > 262144

``MoE::Runner::run``::

    PermuteGemm1 (trtllm-gen batched GEMM, routed FC1, FP8 out + DS scales)
    activationDeepSeekKernel               <<<(32, 8, T), 128>>>
    Gemm2 (trtllm-gen batched GEMM, FC2, BF16 out)
    finalizeKernel                         <<<(28, min(T, 8192)), 256>>> 28 * min(T, 8192) < 1184
    finalizeKernelVecLoad                  <<<T, 256>>>                 otherwise

The GEMM configs are those ``TrtllmGenBatchedGemmRunner::getValidConfigIndices``
selects from the full 408-config ``KernelMetaInfo.h``. Its last tie-breaker
prefers the persistent tile scheduler when ``ceil(M/128) * ceil(T/8) > 148``,
so each GEMM has two live cubins (FC1: M = 4096, persistent for T >= 33;
FC2: M = 7168, persistent for T >= 17). The probe runs that selection for
every fixture and the compiler checks the result against ``GEMMS``.

Kernel inventory and the live/dead decision
-------------------------------------------

==============================================  ======  ==================================
kernel                                          status  workload / reason
==============================================  ======  ==================================
routingMainKernel<KP<f32,bf16,groups,Pdl=0>>    live    moe_routing_main (sm_86, sm_100a)
routingIndicesClusterKernel<...>                live    moe_routing_cluster (sm_100a); its
                                                        sm_86 build is an assert stub
routingIndicesCoopKernel<...>                   live    moe_routing_coop (sm_100a); needs a
                                                        cooperative launch; sm_86 is a stub
routingIndicesHistogram/OffsetsKernel           dead    only for T > 262144
bmm_E4m3_..._t128x8x128u2_..._schedS_..._sm100a live    moe_gemm1 (T <= 32)
bmm_E4m3_..._t128x8x128u2_..._schedP_..._sm100a live    moe_gemm1_persistent (T >= 33)
bmm_Bfloat16_..._t128x8x128u2_..._schedS_...    live    moe_gemm2 (T <= 16)
bmm_Bfloat16_..._t128x8x128u2_..._schedP_...    live    moe_gemm2_persistent (T >= 17)
``<gemm>GetSmemSize`` (in each GEMM cubin)      dead    never launched (the host uses
                                                        config.mSharedMemSize); stripped
non-``u2`` t128x8x128 FC1/FC2 cubins            dead    the previous package's two cubins:
                                                        upstream never selects them here
activationDeepSeekKernel<KP<e4m3,Pdl=0>>        live    moe_activation (sm_86, sm_100a)
finalizeKernel<KP<bf16,bf16,Pdl=0>>             live    moe_finalize (sm_86, sm_100a), T<=42
finalizeKernelVecLoad<KP<bf16,bf16,Pdl=0>>      live    moe_finalize_vec (sm_86, sm_100a)
activationKernel, permuteKernel,                dead    non-DeepSeek-FP8 paths of the same
convertSf*Kernel, finalizeDeepSeekKernel                launchers
``UsePdl = true`` instantiations                dead    see "PDL" below
cub::detail::EmptyKernel<void>                  dead    emitted into every TU by cub's
                                                        PtxVersionUncached; never launched;
                                                        stripped
other routing methods (Llama4, Renormalize)     dead    DeepSeekV3 only
==============================================  ======  ==================================

PDL: upstream sets ``mUsePdl = true`` for routing/activation/finalize (kernels
templated ``UsePdl = true``, launched with programmatic stream serialization)
while the GEMMs launch without it (``TRTLLM_ENABLE_PDL`` unset). As in the
previous package, this package builds the ``UsePdl = false`` instantiations and
launches without PDL (``USE_PDL``); the parameter layout is identical and the
probe's recordings show the upstream attribute that is not applied.

One kernel per cubin
--------------------

``kernels/moe_*.cuh`` copy the upstream kernel templates verbatim (marked
blocks that ``verify_sources`` compares byte for byte with the pinned files);
each ``kernels/moe_<kernel>.cu`` holds one explicit instantiation and no host
code. nvcc still emits cub's ``EmptyKernel<void>`` into every TU, and each
official GEMM cubin holds a ``GetSmemSize`` helper kernel. ``strip_cubin``
(``harness/cubin_strip.py``, shared with other packages) removes those dead
kernels at the ELF level. It removes their
sections, symbols, relocations, ``.nv.info``/callgraph entries, frame
descriptors and line sequences, renumbers everything consistently and keeps
every other byte. That includes the kept kernel's code, constant bank,
metadata and Mercury capsule; only the capsule's leading word changes, since it
holds the index of its SASS ``.text`` section. ``compile`` checks these
invariants for every cubin. ``tests/test_cubin_strip.py`` shows the technique
on sm_86: it builds a multi-kernel cubin, strips each kernel in turn, and the
stripped cubins load and run with identical results. The four stripped
official sm_100a GEMM cubins loaded and validated on a B200 (VALIDATION.md,
October 3, 2026).

Host-side setup without native code
-----------------------------------

``kernels/moe_probe.cu`` (host-only, compile time only) includes the four
pinned upstream host sources and repeats the official launcher's call sequence
with fake device addresses, interposing the CUDA runtime/driver entry points.
It records every launch (kernel, grid, block, shared memory, attributes, the
by-value parameter bytes, bytes upstream leaves indeterminate, and every
``cuTensorMapEncodeTiled`` call) and the static options of the selected GEMM
configs. ``harness/workloads/moe.py`` rebuilds all parameters in Python;
``tests/test_moe.py`` compares them byte for byte with these recordings.

Cases, upstream tests and fixtures
----------------------------------

``FIXTURE_CASES`` is every (tokens, local_expert_offset) of the workloads'
smoke and throughput cases (``harness/workloads/moe.py``): the probe runs
upstream's host code for each, so the kernel choice (routing cluster/coop,
static/persistent GEMM config, finalize variant) of every case is upstream's
own, and each launch is a byte-compared fixture. ``tests/test_moe.py``
requires every case to be probed. The cases map the upstream sources as
follows (enforced by ``tests/test_moe.py::UpstreamCoverage``):

* ``resources/moe.json`` (official definition): its constant axes are the
  constants here; ``tests/test_official_references.py`` chains the stage
  references against its ``run``.
* ``resources/benchmark_workloads/moe.jsonl``: all 19 rows (T = 1, 7, 14, 15,
  16, 32, 52-59, 62, 80, 901, 11948, 14107 with their recorded offsets) are
  throughput cases of the kernels upstream dispatches them to.
* FlashInfer v0.2.10 ``tests/test_trtllm_gen_fused_moe.py::
  test_moe_quantization_classes[NoShuffle_MajorK-DSv3-FP8_Block-*-1024-T]``
  (the parametrizations that run ``trtllm_fp8_block_scale_moe`` with this
  routing): T = 1 and 1024, offset 0, its logits/bias distribution, as smoke
  cases. Its hidden 1024 / intermediate {1024, 768, 384}, 256 local experts
  and tile_tokens_dim 32 (T = 1024) are other GEMM shapes, configs and
  padding than this definition fixes (7168, 2048, 32, 8); those kernels are
  not in this package. DSLite (72 experts, 1 group) is another definition.

Nothing in this module runs at benchmark time.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
MANIFEST = PACKAGE / "kernels.json"
PROVENANCE = PACKAGE / "provenance.json"

FLASHINFER_REPOSITORY = "https://github.com/flashinfer-ai/flashinfer"
FLASHINFER_RELEASE = "v0.2.10"
FLASHINFER_REVISION = "7c79b41b8efd512eb40ec9bd56c9e6cda328d12e"
FLASHINFER_DIR = "flashinfer-moe-v0210"
# Every pinned upstream file the kernel sources and the probe depend on
# (nvcc -M), plus the launcher/JIT sources whose flags and call sequence they
# reproduce.
FLASHINFER_FILES = {
    "LICENSE": ("d4b8eb93aecacd9bb4979d30248b437ef0812d2e85a8728e1fe12fb36eec09b4"),
    "csrc/nv_internal/tensorrt_llm/common/envUtils.h": (
        "f553bf140bb0773867e722502e420f8863be400d6b83f077e6c8115241b1f542"
    ),
    "csrc/trtllm_batched_gemm_runner.cu": (
        "a6b9c062fe4f47a383577623bb4302b8cdf923f5a3d6d43cccbe42517047af3f"
    ),
    "csrc/trtllm_fused_moe_dev_kernel.cu": (
        "dd92240fa5f14c0379a648644d474127aace361a7b0d1b661664e3b30c0d4515"
    ),
    "csrc/trtllm_fused_moe_kernel_launcher.cu": (
        "16408ccd36f12af7297aaf3b994ac84916464a463333a388a79178c969ae78ba"
    ),
    "csrc/trtllm_fused_moe_routing_deepseek.cu": (
        "26f6e19b4f0937c24ceef75b3b065a6c317708d3bb023d1d297dc8ca2da15442"
    ),
    "csrc/trtllm_fused_moe_runner.cu": (
        "c4fc17382313a573226d14b5800e80282bfc8f8da4451b72a572f581fae3a191"
    ),
    "flashinfer/fused_moe/core.py": (
        "0edce7ec31f0892ee7443210a5fd305bdf041dd9c6684536c3a078c6c8a5fa2b"
    ),
    "flashinfer/jit/core.py": (
        "520f7c182041cadf3d2a13e8d8ee8f4e3339302d7495799be93fd35143830f70"
    ),
    "include/flashinfer/trtllm/batched_gemm/KernelRunner.h": (
        "02b651117f61484cb677d6ce13092588e6878f0ad8e7a3e573f4b299d8e40a86"
    ),
    "include/flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/BatchedGemmEnums.h": (
        "690e3369229bdf00a1efce62cab73b28d7398a78ada5ebfc5566c7fb68a828d7"
    ),
    "include/flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/BatchedGemmInterface.h": (
        "b165082462fd53180039a5cb5c7db92e860d8881df30c33ba06c76738819f8fd"
    ),
    "include/flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/BatchedGemmOptions.h": (
        "e52c1137016143a6f5bdc6db2e485461c4a87f7513aa66cfc01417976d6c1010"
    ),
    "include/flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/Enums.h": (
        "654e6a78fa9e27f6e77744ca8f6905b0407539ff89874968e883aebed9930a25"
    ),
    "include/flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/GemmGatedActOptions.h": (
        "34106bc6192ffa0c867f6da88cbc9a6c2675f0019c6f0d0bbc16daf08bb8d477"
    ),
    "include/flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/GemmOptions.h": (
        "ff3a64aa6d56e6ffbcd2fa243307068666d78924f27275335d125d115db82719"
    ),
    "include/flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/KernelMetaInfo.h": (
        "0e1f55ae134ce768b4a07458cb95028eccc64d6f02803e1d94af6da109ec4174"
    ),
    "include/flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/KernelParams.h": (
        "c51058b14873ee0b2d0e88f5a1afff57039c0be60933dd06ee020184f0550285"
    ),
    "include/flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/KernelParamsDecl.h": (
        "473b27b6f8a3d8b9b27460e65b25d7be702823f49b5db0b942d622371a0923ec"
    ),
    "include/flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/KernelTraits.h": (
        "2815f72d4233be929432e168fd35c42ac94bff078a05c5e4483b7140a3119e4f"
    ),
    "include/flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/TmaDescriptor.h": (
        "76edff4384600deb331a6f70e566fccff6a7c748a5ddb9943c70c242752e00b4"
    ),
    "include/flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/trtllm/gen/CommonUtils.h": (
        "480bace11fd6032255f365cc7d6d7b5fd9c09432862405d53c057a1523c533ce"
    ),
    "include/flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/trtllm/gen/CudaKernelLauncher.h": (
        "cb385d3eade68e0ec4a1d6b2a4d9ef0162e3485f7e2dce8b7ab3ce80907494ab"
    ),
    "include/flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/trtllm/gen/DtypeDecl.h": (
        "f0032ef9b1f1aa45007f0c057b805d6700d96d95288b92a4aa0788a271c1edab"
    ),
    "include/flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/trtllm/gen/MmaDecl.h": (
        "77a2196cfcc3037ce4023a89f69e335ce212240e7f8cd24eca9fa8b3439f8d77"
    ),
    "include/flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/trtllm/gen/SfLayoutDecl.h": (
        "6756854a4a099ac39c4a55d3cfe2ffed05bdb282aa124fe36102a97108c81bc1"
    ),
    "include/flashinfer/trtllm/common/cudaUtils.h": (
        "430f2580bb4e047ce18f60e8d5dfb7544a7cbdbe09a5e5551fca1bffc989d2d7"
    ),
    "include/flashinfer/trtllm/fused_moe/DevKernel.h": (
        "2f8fc543c7c9d8b9604060c8fa100959cdceb6ab3b834727d393777afe7fa532"
    ),
    "include/flashinfer/trtllm/fused_moe/IntFastDiv.h": (
        "21608a4d3c69e166611d16996f955668c34cac341cb3cf3072311a8a28f1c004"
    ),
    "include/flashinfer/trtllm/fused_moe/RoutingKernel.cuh": (
        "33c182154f5ab7c60f81c7d253facff74276e0c3c49e8ba4061d6500bff9088d"
    ),
    "include/flashinfer/trtllm/fused_moe/RoutingKernel.h": (
        "01d9ae692eba1cdae92c98905f84c9dd7ddd9755a7261960efd851ce98468d10"
    ),
    "include/flashinfer/trtllm/fused_moe/RoutingKernelTopK.cuh": (
        "29736e3a67622c691d02a8900c2d2eda667919721afa844dddfa537918dd1452"
    ),
    "include/flashinfer/trtllm/fused_moe/runner.h": (
        "d666978f8518d16b09a103f122f24b90ed75c99b5600c4c461c384745254c3c4"
    ),
}

# v0.2.10 pins CUTLASS as a git submodule (3rdparty/cutlass at
# UPSTREAM_CUTLASS_SUBMODULE), absent from the source archive. The kernels only
# use CUTLASS numeric types, arrays and the SM90 cluster helpers; this package
# builds against the CUTLASS revision the other packages pin, whose files match
# the hashes the previous MoE package recorded for every CUTLASS header it used.
UPSTREAM_CUTLASS_SUBMODULE = "f115c3f85467d5d9619119d1dbeb9c03c3d73864"
CUTLASS_REPOSITORY = "https://github.com/NVIDIA/cutlass"
CUTLASS_REVISION = "b46b16d003484063bca4ed365e44095c4c6ed633"
CUTLASS_DIR = f"cutlass-{CUTLASS_REVISION}"
CUTLASS_INCLUDE_TREE_SHA256 = (
    "def1849ca143942f68179aeb664a6cb9bf46001082d7d4f5c63531ea03866e6d"
)

# Official trtllm-gen batched-GEMM cubins of this revision.
ARTIFACT_PATH = "c8e0abb4b0438880a2b0a9b68449e3cf1513aadf/batched_gemm-32110eb-a15c257"
ARTIFACT_BASE_URL = (
    "https://edge.urm.nvidia.com/artifactory/"
    "sw-kernelinferencelibrary-public-generic-local/" + ARTIFACT_PATH
)
ARTIFACT_DIR = "moe-artifacts"
META_INFO = (
    "include/flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/KernelMetaInfo.h"
)

USE_PDL = False
NUM_EXPERTS, TOP_K, N_GROUP, TOPK_GROUP = 256, 8, 8, 4
HIDDEN, INTERMEDIATE, LOCAL_EXPERTS, TILE_TOKENS = 7168, 2048, 32, 8
ROUTED_SCALING_FACTOR = 2.5
# (tokens, local_expert_offset) of every smoke and throughput case; they
# include both sides of every dispatch boundary (16/17, 32/33, 42/43, 1024/1025).
FIXTURE_CASES = (
    (1, 0),
    (1, 32),
    (4, 96),
    (7, 192),
    (8, 128),
    (14, 0),
    (15, 32),
    (16, 224),
    (17, 224),
    (24, 0),
    (32, 0),
    (32, 32),
    (33, 80),
    (42, 160),
    (43, 96),
    (52, 160),
    (53, 32),
    (54, 128),
    (55, 128),
    (56, 64),
    (57, 96),
    (58, 64),
    (59, 160),
    (62, 96),
    (64, 64),
    (80, 96),
    (128, 0),
    (300, 32),
    (512, 96),
    (901, 96),
    (1024, 0),
    (1025, 224),
    (3000, 128),
    (4096, 224),
    (8192, 160),
    (11948, 128),
    (14107, 32),
    (16384, 128),
    (16384, 224),
)

COMMON_FLAGS = (
    # flashinfer/jit/core.py (gen_jit_spec, non-verbose) and the fused_moe_sm100
    # module's extra_cuda_cflags (flashinfer/fused_moe/core.py).
    "-std=c++17",
    "-use_fast_math",
    "-DNDEBUG",
    "-DFLASHINFER_ENABLE_F16",
    "-DFLASHINFER_ENABLE_BF16",
    "-DFLASHINFER_ENABLE_FP8_E4M3",
    "-DFLASHINFER_ENABLE_FP8_E5M2",
    "-DTLLM_GEN_EXPORT_INTERFACE",
    "-DTLLM_ENABLE_CUDA",
    "-DENABLE_BF16",
    "-DENABLE_FP8",
    "-DENABLE_FP4",
    '-DTLLM_GEN_BMM_CUBIN_PATH=""',
    "-diag-suppress=20012",  # upstream's defaulted __host__ __device__ ctor
)
GENCODE = {
    "sm_86": "-gencode=arch=compute_86,code=sm_86",
    "sm_100a": "-gencode=arch=compute_100a,code=sm_100a",
}
_ROUTING = "_ZN3moe3dev7routing15routingDeepSeek"
_ROUTING_KP = "INS2_12KernelParamsIf13__nv_bfloat16Lb1ELb0EEEEEvT_"
_EMPTY_KERNEL = re.compile(r"^_ZN3cub\d*_.*detail11EmptyKernelIvEEvv$")


class KernelSpec(NamedTuple):
    """One workload: its kernel, where it comes from and which launches it owns.

    (A NamedTuple: compile_kernels.py loads this module by path.)
    """

    workload: str
    arches: tuple[str, ...]
    upstream: str
    probe_kernel: str  # launch name in the probe's recordings
    source: str | None = None  # kernels/<source>.cu (nvcc-built kernels)
    symbol: str | None = None  # expected mangled name (nvcc-built kernels)
    gemm_cubin: str | None = None  # official cubin file name (GEMMs)
    gemm_sha256: str | None = None


KERNELS = (
    KernelSpec(
        "moe_routing_main",
        ("sm_86", "sm_100a"),
        "routingDeepSeek::routingMainKernel<KernelParams<float, bf16, true, false>>",
        "routingMainKernel",
        source="moe_routing_main",
        symbol=_ROUTING + "17routingMainKernel" + _ROUTING_KP,
    ),
    KernelSpec(
        "moe_routing_cluster",
        ("sm_100a",),
        "routingDeepSeek::routingIndicesClusterKernel<KernelParams<float, bf16, true, "
        "false>>",
        "routingIndicesClusterKernel",
        source="moe_routing_cluster",
        symbol=_ROUTING + "27routingIndicesClusterKernel" + _ROUTING_KP,
    ),
    KernelSpec(
        "moe_routing_coop",
        ("sm_100a",),
        "routingDeepSeek::routingIndicesCoopKernel<KernelParams<float, bf16, true, "
        "false>>",
        "routingIndicesCoopKernel",
        source="moe_routing_coop",
        symbol=_ROUTING + "24routingIndicesCoopKernel" + _ROUTING_KP,
    ),
    KernelSpec(
        "moe_gemm1",
        ("sm_100a",),
        "PermuteGemm1: trtllm-gen batched GEMM (routed FC1, static scheduler)",
        "bmm_E4m3_E4m3E4m3_Fp32_t128x8x128u2_s8_et64x8_m64x8x32_cga1x1x1_16dp256b_TN_"
        "transOut_noShflA_dsFp8_schedS_bN_ldgsts_clmp_dynBatch_sm100a",
        gemm_cubin="Bmm_E4m3_E4m3E4m3_Fp32_t128x8x128u2_s8_et64x8_m64x8x32_cga1x1x1_"
        "16dp256b_TN_transOut_noShflA_dsFp8_schedS_bN_ldgsts_clmp_dynBatch_sm100a.cubin",
        gemm_sha256="bc2ffee4831f12acf4e54ac3ef3f6279c5517100ee04d16167587114ecabf409",
    ),
    KernelSpec(
        "moe_gemm1_persistent",
        ("sm_100a",),
        "PermuteGemm1: trtllm-gen batched GEMM (routed FC1, persistent scheduler)",
        "bmm_E4m3_E4m3E4m3_Fp32_t128x8x128u2_s8_et64x8_m64x8x32_cga1x1x1_16dp256b_TN_"
        "transOut_noShflA_dsFp8_schedP_bN_ldgsts_clmp_dynBatch_sm100a",
        gemm_cubin="Bmm_E4m3_E4m3E4m3_Fp32_t128x8x128u2_s8_et64x8_m64x8x32_cga1x1x1_"
        "16dp256b_TN_transOut_noShflA_dsFp8_schedP_bN_ldgsts_clmp_dynBatch_sm100a.cubin",
        gemm_sha256="52fe80baf1b55bfac15ca876b5c2b4111d1f475fa176219ba491543cbe908360",
    ),
    KernelSpec(
        "moe_activation",
        ("sm_86", "sm_100a"),
        "activation::activationDeepSeekKernel<KernelParams<cutlass::float_e4m3_t, false>>",
        "activationDeepSeekKernel",
        source="moe_activation",
        symbol="_ZN3moe3dev10activation24activationDeepSeekKernelINS1_12KernelParams"
        "IN7cutlass12float_e4m3_tELb0EEEEEvT_",
    ),
    KernelSpec(
        "moe_gemm2",
        ("sm_100a",),
        "Gemm2: trtllm-gen batched GEMM (FC2, static scheduler)",
        "bmm_Bfloat16_E4m3E4m3_Fp32_t128x8x128u2_s8_et64x8_m64x8x32_cga1x1x1_16dp256b_TN_"
        "transOut_noShflA_dsFp8_schedS_bN_clmp_dynBatch_sm100a",
        gemm_cubin="Bmm_Bfloat16_E4m3E4m3_Fp32_t128x8x128u2_s8_et64x8_m64x8x32_cga1x1x1_"
        "16dp256b_TN_transOut_noShflA_dsFp8_schedS_bN_clmp_dynBatch_sm100a.cubin",
        gemm_sha256="53d992332e02a59947bd40929f9c8efd3da79234a77d6fddabd868d58bdff5d5",
    ),
    KernelSpec(
        "moe_gemm2_persistent",
        ("sm_100a",),
        "Gemm2: trtllm-gen batched GEMM (FC2, persistent scheduler)",
        "bmm_Bfloat16_E4m3E4m3_Fp32_t128x8x128u2_s8_et64x8_m64x8x32_cga1x1x1_16dp256b_TN_"
        "transOut_noShflA_dsFp8_schedP_bN_clmp_dynBatch_sm100a",
        gemm_cubin="Bmm_Bfloat16_E4m3E4m3_Fp32_t128x8x128u2_s8_et64x8_m64x8x32_cga1x1x1_"
        "16dp256b_TN_transOut_noShflA_dsFp8_schedP_bN_clmp_dynBatch_sm100a.cubin",
        gemm_sha256="e40531b337de3dff738fd228d732675485ba0e32a3a0f4cfba34f106edce258a",
    ),
    KernelSpec(
        "moe_finalize",
        ("sm_86", "sm_100a"),
        "finalize::finalizeKernel<KernelParams<bf16, bf16, false>>",
        "finalizeKernel",
        source="moe_finalize",
        symbol="_ZN3moe3dev8finalize14finalizeKernelINS1_12KernelParamsIN7cutlass10"
        "bfloat16_tES5_Lb0EEEEEvT_",
    ),
    KernelSpec(
        "moe_finalize_vec",
        ("sm_86", "sm_100a"),
        "finalize::finalizeKernelVecLoad<KernelParams<bf16, bf16, false>>",
        "finalizeKernelVecLoad",
        source="moe_finalize_vec",
        symbol="_ZN3moe3dev8finalize21finalizeKernelVecLoadINS1_12KernelParamsIN7cutlass"
        "10bfloat16_tES5_Lb0EEEEEvT_",
    ),
)
# Probe launch name -> parameter struct of the layout mode.
STRUCT_OF = {
    "routingMainKernel": "routing",
    "routingIndicesClusterKernel": "routing",
    "routingIndicesCoopKernel": "routing",
    "activationDeepSeekKernel": "activation",
    "finalizeKernel": "finalize",
    "finalizeKernelVecLoad": "finalize",
}
DEAD_LAUNCHES = ("routingIndicesHistogramKernel", "routingIndicesOffsetsKernel")
GEMM_PARAMS_SIZE = 0x4380


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _cutlass_tree_sha256(cutlass: Path) -> str:
    include = cutlass / "include"
    return hashlib.sha256(
        b"".join(
            str(path.relative_to(cutlass)).encode()
            + hashlib.sha256(path.read_bytes()).digest()
            for path in sorted(include.rglob("*"))
            if path.is_file()
        )
    ).hexdigest()


def _relative(path: Path) -> str:
    """``path`` relative to the repository root when inside it, so recorded
    commands are portable (compilers run with ``cwd=ROOT``)."""
    path = Path(os.path.abspath(path))
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def _cubin_info(image: bytes) -> tuple[str, list[str], list[Any]]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from harness.workload import cubin_info

    return cubin_info(image)


def _ranges(indices: Sequence[int]) -> list[list[int]]:
    ranges: list[list[int]] = []
    for index in sorted(indices):
        if ranges and ranges[-1][1] == index:
            ranges[-1][1] += 1
        else:
            ranges.append([index, index + 1])
    return ranges


def _cubin_strip() -> Any:
    """The single-kernel ELF strip shared with other packages
    (``harness/cubin_strip.py``)."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from harness import cubin_strip

    return cubin_strip


# =============================================================================
# The compiler
# =============================================================================


def _verbatim_blocks(text: str) -> list[tuple[str, int, int, str]]:
    blocks = re.findall(
        r"// VERBATIM BEGIN (\S+):(\d+)-(\d+)\n(.*?)// VERBATIM END\n", text, re.S
    )
    return [(f, int(a), int(b), body) for f, a, b, body in blocks]


class ImplCompiler:
    """Builds ``cubins/<arch>/<workload>.cubin`` with sidecar and fixtures for
    every MoE workload of ``arch`` and merges them into ``kernels.json``."""

    supported_arches: tuple[str, ...] = ("sm_86", "sm_100a")

    def __init__(
        self,
        arch: str,
        *,
        optimization: str = "-O3",
        flags: Sequence[str] = (),
        nvcc: str | None = None,
        resources: Path | None = None,
    ):
        if arch not in self.supported_arches:
            raise ValueError(f"moe supports {self.supported_arches}, not {arch}")
        if optimization not in ("-O0", "-O1", "-O2", "-O3"):
            raise ValueError(f"invalid optimization {optimization!r}")
        self.arch = arch
        self.optimization = optimization
        self.flags = list(flags)
        self.nvcc = nvcc or os.environ.get("NVCC") or shutil.which("nvcc") or "nvcc"
        self.resources = Path(os.path.abspath(resources or ROOT / "resources"))
        self.flashinfer = self.resources / FLASHINFER_DIR
        self.cutlass = self.resources / CUTLASS_DIR
        self.artifacts = self.resources / ARTIFACT_DIR

    # -- pinned sources and artifacts ------------------------------------------

    def verify_sources(self) -> None:
        for name, digest in FLASHINFER_FILES.items():
            path = self.flashinfer / name
            if not path.is_file() or _sha256(path.read_bytes()) != digest:
                raise ValueError(
                    f"{path} is missing or differs from FlashInfer {FLASHINFER_REVISION}"
                )
        tree = _cutlass_tree_sha256(self.cutlass)
        if tree != CUTLASS_INCLUDE_TREE_SHA256:
            raise ValueError(f"unexpected CUTLASS include tree hash {tree}")
        for header in sorted((PACKAGE / "kernels").glob("*.cuh")):
            blocks = _verbatim_blocks(header.read_text())
            if not blocks:
                raise ValueError(f"{header.name} has no VERBATIM blocks")
            for name, first, last, body in blocks:
                if name not in FLASHINFER_FILES:
                    raise ValueError(f"{header.name}: {name} is not a pinned file")
                lines = (self.flashinfer / name).read_text().splitlines(keepends=True)
                if "".join(lines[first - 1 : last]) != body:
                    raise ValueError(f"{header.name}: {name}:{first}-{last} differs")
        meta = (self.flashinfer / META_INFO).read_text()
        for spec in KERNELS:
            if spec.gemm_cubin is None:
                continue
            found = re.search(
                r'"' + re.escape(spec.probe_kernel) + r'", \d+, "([0-9a-f]{64})"', meta
            )
            if found is None or found.group(1) != spec.gemm_sha256:
                raise ValueError(f"{spec.gemm_cubin}: hash not in pinned metadata")

    def gemm_image(self, spec: KernelSpec) -> bytes:
        """The official cubin from ``resources/moe-artifacts`` (downloaded from
        the pinned artifact path if absent), checked against its SHA256."""
        assert spec.gemm_cubin is not None
        path = self.artifacts / spec.gemm_cubin
        if not path.is_file():
            with urllib.request.urlopen(
                ARTIFACT_BASE_URL + "/" + spec.gemm_cubin, timeout=120
            ) as response:
                data = response.read()
            if _sha256(data) != spec.gemm_sha256:
                raise ValueError(f"{spec.gemm_cubin}: downloaded bytes do not match")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        data = path.read_bytes()
        if _sha256(data) != spec.gemm_sha256:
            raise ValueError(f"{path} does not match its pinned SHA256")
        return data

    # -- commands ------------------------------------------------------------

    def _includes(self) -> list[str]:
        return [
            "-I" + _relative(PACKAGE / "kernels/include"),
            "-I" + _relative(self.flashinfer / "include"),
            "-I" + _relative(self.cutlass / "include"),
        ]

    def kernel_command(self, spec: KernelSpec, output: Path) -> list[str]:
        return [
            self.nvcc,
            "-cubin",
            self.optimization,
            "-lineinfo",
            *COMMON_FLAGS,
            GENCODE[self.arch],
            *self._includes(),
            *self.flags,
            _relative(PACKAGE / f"kernels/{spec.source}.cu"),
            "-o",
            str(output),
        ]

    def probe_command(self, output: Path) -> list[str]:
        # Host executable. The shared cudart lets it interpose the runtime;
        # it never links libcuda (the driver entry points are its own).
        return [
            self.nvcc,
            self.optimization,
            *COMMON_FLAGS,
            GENCODE["sm_100a"],
            "--cudart=shared",
            "-I" + _relative(PACKAGE / "kernels/include"),
            "-I" + _relative(self.flashinfer),
            "-I" + _relative(self.flashinfer / "include"),
            "-I" + _relative(self.flashinfer / "csrc/nv_internal"),
            "-I" + _relative(self.flashinfer / "csrc/nv_internal/include"),
            "-I" + _relative(self.cutlass / "include"),
            _relative(PACKAGE / "kernels/moe_probe.cu"),
            "-o",
            str(output),
        ]

    @staticmethod
    def _portable(command: list[str]) -> list[str]:
        root = str(ROOT) + os.sep
        return [part.replace(root, "") for part in command]

    # -- probe -----------------------------------------------------------------

    def run_probe(self, probe: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        layout = json.loads(subprocess.check_output([str(probe), "layout"], text=True))
        runs = [
            json.loads(
                subprocess.check_output(
                    [str(probe), "run", str(t), str(o), str(ROUTED_SCALING_FACTOR)],
                    text=True,
                )
            )
            for t, o in FIXTURE_CASES
        ]
        for run in runs:
            if run["workspace_fc1"] or run["workspace_fc2"]:
                raise ValueError("upstream requests a GEMM workspace")
            if any(e["call"] not in ("cudaDeviceGetAttribute", "cuFuncSetAttribute")
                   for e in run["events"]):  # fmt: skip
                raise ValueError(f"unexpected host call: {run['events']}")
        return layout, runs

    @staticmethod
    def _launch_name(launch: dict[str, Any]) -> str:
        return launch["kernel"].split("<", 1)[0]

    def fixtures_of(
        self, spec: KernelSpec, runs: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        out = []
        for run in runs:
            launches = [
                launch for launch in run["launches"]
                if self._launch_name(launch) == spec.probe_kernel
            ]  # fmt: skip
            if not launches:
                continue
            if len(launches) != 1:
                raise ValueError(f"{spec.workload}: launched twice in one run")
            launch = dict(launches[0])
            launch["padding"] = _ranges(launch.pop("indeterminate_bytes"))
            out.append(
                {
                    "tokens": run["tokens"],
                    "local_expert_offset": run["local_expert_offset"],
                    "routed_scaling_factor_hex": run["routed_scaling_factor_hex"],
                    "sm_count": run["sm_count"],
                    "max_num_padded_tokens": run["max_num_padded_tokens"],
                    "max_num_ctas": run["max_num_ctas"],
                    "pointers": run["pointers"],
                    "launch": launch,
                }
            )
        return out

    # -- build ---------------------------------------------------------------------

    def compile(self) -> dict[str, str]:
        self.verify_sources()
        out_dir = PACKAGE / "cubins" / self.arch
        version = subprocess.check_output([self.nvcc, "--version"], text=True)
        specs = [s for s in KERNELS if self.arch in s.arches]
        strip = _cubin_strip()
        built: list[tuple[KernelSpec, bytes, dict[str, Any], dict[str, Any]]] = []
        with tempfile.TemporaryDirectory(prefix="moe-") as tmp:
            probe = Path(tmp) / "moe_probe"
            subprocess.run(self.probe_command(probe), check=True, cwd=ROOT)
            layout, runs = self.run_probe(probe)
            known = {s.probe_kernel for s in KERNELS}
            for run in runs:
                for launch in run["launches"]:
                    name = self._launch_name(launch)
                    if name in DEAD_LAUNCHES or name not in known:
                        raise ValueError(
                            f"unexpected upstream launch {launch['kernel']}"
                        )
            for spec in specs:
                if spec.source is not None:
                    staged = Path(tmp) / f"{spec.workload}.nvcc.cubin"
                    subprocess.run(
                        self.kernel_command(spec, staged), check=True, cwd=ROOT
                    )
                    original = staged.read_bytes()
                    keep = spec.symbol
                    others = [k for k in strip.cubin_kernels(original) if k != keep]
                    if any(not _EMPTY_KERNEL.match(k) for k in others):
                        raise ValueError(f"{spec.source}: unexpected kernels {others}")
                else:
                    original = self.gemm_image(spec)
                    keep = spec.probe_kernel
                    others = [k for k in strip.cubin_kernels(original) if k != keep]
                    if others != [keep + "GetSmemSize"]:
                        raise ValueError(
                            f"{spec.gemm_cubin}: unexpected kernels {others}"
                        )
                assert keep is not None
                image = strip.strip_cubin(original, keep)
                strip.check_strip(original, image, keep)
                arch, names, params = _cubin_info(image)
                if arch != self.arch or names != [keep]:
                    raise ValueError(f"{spec.workload}: cubin declares {arch} {names}")
                fixtures = self.fixtures_of(spec, runs)
                if not fixtures:
                    raise ValueError(f"{spec.workload}: never launched by upstream")
                sidecar = self._sidecar(
                    spec, image, keep, params, layout, fixtures, runs
                )
                sidecar["stripped_kernels"] = others
                sidecar["original_sha256"] = _sha256(original)
                built.append((spec, image, sidecar, {"fixtures": fixtures}))
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)
        mapping: dict[str, str] = {}
        records: dict[str, dict[str, Any]] = {}
        for spec, image, sidecar, recorded in built:
            target = out_dir / f"{spec.workload}.cubin"
            target.write_bytes(image)
            relative = target.relative_to(PACKAGE).as_posix()
            mapping[spec.workload] = relative
            self._write_json(out_dir / f"{spec.workload}.json", sidecar)
            self._write_json(
                out_dir / f"{spec.workload}.fixtures.json",
                {
                    "workload": spec.workload,
                    "arch": self.arch,
                    "tensor_map_encoder": "fake (moe_probe.cu fake_tensor_map)",
                    **recorded,
                },
            )
            record: dict[str, Any] = {
                "cubin": relative,
                "upstream": spec.upstream,
                "symbol": sidecar["symbol"],
                "sha256": sidecar["cubin_sha256"],
                "stripped_kernels": sidecar["stripped_kernels"],
                "unstripped_sha256": sidecar["original_sha256"],
            }
            if spec.source is not None:
                record["source"] = f"kernels/{spec.source}.cu"
                record["command"] = self._portable(self.kernel_command(spec, target))
            else:
                assert spec.gemm_cubin is not None
                record["official_cubin"] = spec.gemm_cubin
                record["official_sha256"] = spec.gemm_sha256
                record["url"] = ARTIFACT_BASE_URL + "/" + spec.gemm_cubin
            records[spec.workload] = record
        self._merge_manifest(mapping)
        self._write_provenance(records, version.strip().splitlines()[-1])
        return mapping

    def _sidecar(
        self,
        spec: KernelSpec,
        image: bytes,
        symbol: str,
        params: list[Any],
        layout: dict[str, Any],
        fixtures: list[dict[str, Any]],
        runs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        sizes = [(p.offset, p.size) for p in params]
        launch = fixtures[0]["launch"]
        sidecar: dict[str, Any] = {
            "workload": spec.workload,
            "arch": self.arch,
            "upstream": spec.upstream,
            "cubin": f"{spec.workload}.cubin",
            "cubin_sha256": _sha256(image),
            "symbol": symbol,
            "block": launch["block"],
            "shared_mem": launch["shared_mem"],
            "upstream_attrs": launch["attrs"],
            "use_pdl": USE_PDL,
            "served_tokens": sorted({f["tokens"] for f in fixtures}),
            "probed_tokens": sorted({t for t, _ in FIXTURE_CASES}),
        }
        for fixture in fixtures:
            other = fixture["launch"]
            if (other["block"], other["shared_mem"], other["attrs"]) != (
                launch["block"], launch["shared_mem"], launch["attrs"],
            ):  # fmt: skip
                raise ValueError(f"{spec.workload}: launch constants vary by case")
        if spec.gemm_cubin is None:
            struct_layout = layout["structs"][STRUCT_OF[spec.probe_kernel]]
            if sizes != [(0, struct_layout["size"])]:
                raise ValueError(f"{spec.workload}: kernel params {sizes}")
            sidecar["params"] = struct_layout
            sidecar["int_fast_div_size"] = layout["int_fast_div"]["size"]
        else:
            if layout["structs"]["gemm"]["size"] != GEMM_PARAMS_SIZE or sizes != [
                (0, GEMM_PARAMS_SIZE)
            ]:
                raise ValueError(f"{spec.workload}: kernel params {sizes}")
            options = next(
                run["gemm_configs"][spec.probe_kernel]
                for run in runs
                if spec.probe_kernel in run["gemm_configs"]
            )
            if options["sha256"] != spec.gemm_sha256:
                raise ValueError(f"{spec.workload}: probe selected another cubin")
            if launch["shared_mem"] != options["shared_mem"] or launch["block"] != [
                options["threads"], 1, 1,
            ]:  # fmt: skip
                raise ValueError(f"{spec.workload}: launch disagrees with its config")
            sidecar["params"] = layout["structs"]["gemm"]
            sidecar["gemm_options"] = options
        return sidecar

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.write_text(json.dumps(value, indent=2) + "\n")

    def _merge_manifest(self, mapping: dict[str, str]) -> None:
        manifest = json.loads(MANIFEST.read_text()) if MANIFEST.is_file() else {}
        manifest[self.arch] = dict(sorted(mapping.items()))
        self._write_json(MANIFEST, dict(sorted(manifest.items())))

    def _write_provenance(
        self, records: dict[str, dict[str, Any]], nvcc_version: str
    ) -> None:
        provenance = json.loads(PROVENANCE.read_text()) if PROVENANCE.is_file() else {}
        builds = provenance.get("builds", {}) if isinstance(provenance, dict) else {}
        builds[self.arch] = {
            "nvcc": nvcc_version,
            "probe": self._portable(self.probe_command(Path("moe_probe"))),
            "kernels": dict(sorted(records.items())),
        }
        provenance = {
            "repository": FLASHINFER_REPOSITORY,
            "release": FLASHINFER_RELEASE,
            "revision": FLASHINFER_REVISION,
            "source": f"resources/{FLASHINFER_DIR}",
            "files": FLASHINFER_FILES,
            "license": "LICENSE (FlashInfer, Apache-2.0); CUTLASS_LICENSE "
            "(BSD-3-Clause); official trtllm-gen cubins distributed by NVIDIA "
            "through FlashInfer's artifact repository",
            "cutlass_repository": CUTLASS_REPOSITORY,
            "cutlass_revision": CUTLASS_REVISION,
            "cutlass_include_tree_sha256": CUTLASS_INCLUDE_TREE_SHA256,
            "cutlass_choice": (
                "FlashInfer v0.2.10 pins CUTLASS as submodule "
                f"{UPSTREAM_CUTLASS_SUBMODULE}, absent from the source archive; the "
                f"kernels build against CUTLASS {CUTLASS_REVISION} (the revision the "
                "other packages pin), whose headers match the hashes the previous "
                "MoE package recorded. The kernels use only CUTLASS numeric types, "
                "arrays/converters and the SM90 cluster helpers."
            ),
            "artifacts": {
                "base_url": ARTIFACT_BASE_URL,
                "path": ARTIFACT_PATH,
                "metadata": META_INFO,
                "cubins": {
                    s.gemm_cubin: s.gemm_sha256 for s in KERNELS if s.gemm_cubin
                },
                "local_copy": f"resources/{ARTIFACT_DIR}/ (gitignored)",
            },
            "scope": "trtllm_fp8_block_scale_moe (DeepSeek-V3 routing, 256 experts, "
            "top-8, 8 groups / 4 kept, 32 local experts, hidden 7168, intermediate "
            "2048, tile 8, unshuffled MajorK weights), one kernel per cubin and per "
            "workload; each workload owns exactly the token counts upstream "
            "dispatches to its kernel",
            "dispatch": {
                "moe_routing_main": "all tokens",
                "moe_routing_cluster": "tokens <= 1024",
                "moe_routing_coop": "1024 < tokens <= 262144 (cooperative launch)",
                "moe_gemm1": "32 * ceil(tokens / 8) <= 148 (tokens <= 32)",
                "moe_gemm1_persistent": "tokens >= 33",
                "moe_gemm2": "56 * ceil(tokens / 8) <= 148 (tokens <= 16)",
                "moe_gemm2_persistent": "tokens >= 17",
                "moe_activation": "all tokens",
                "moe_finalize": "28 * min(tokens, 8192) < 1184 (tokens <= 42)",
                "moe_finalize_vec": "tokens >= 43",
            },
            "dead_kernels": {
                "routingIndicesHistogramKernel, routingIndicesOffsetsKernel": "tokens "
                "> 262144 only",
                "<gemm>GetSmemSize": "helper kernel in every official GEMM cubin; the "
                "host takes config.mSharedMemSize and never launches it; stripped",
                "cub::detail::EmptyKernel<void>": "emitted into every nvcc TU by "
                "cub::PtxVersionUncached; never launched; stripped",
                "Bmm_*_t128x8x128_s8_* (non-u2, the previous package's two cubins)": "not "
                "selected by upstream's config ordering for any token count (the "
                "u2 = unrolled-MMA configs sort first); recorded by the probe",
                "activationKernel, permuteKernel, convertSf*Kernel, "
                "finalizeDeepSeekKernel": "non-DeepSeek-FP8 branches of the same "
                "launchers",
                "UsePdl=true instantiations": "see adaptations",
                "routingLlama4 / routingRenormalize": "other routing methods",
            },
            "elf_strip": (
                "harness/cubin_strip.py strip_cubin: removes the dead kernels' "
                "sections (text, Mercury capsule and everything info-linked to them), "
                "symbols, relocations, global .nv.info/.nv.merc.nv.info/callgraph "
                "entries, .debug_frame FDEs/CIEs and DWARF line sequences; renumbers "
                "sections, symbols, relocation and .nv.info symbol references, "
                "sh_link/sh_info and segments; copies every other byte. The first "
                "u32 of each .nv.capmerc.text.<k> Mercury capsule holds the index of "
                "its SASS .text.<k> section (0x10/0x11 for sections 16/17 in the FC1 "
                "cubin, 0x12/0x13 for 18/19 in FC2) and is rewritten to the new index. "
                "compile() checks identity rebuild, one kernel, and unchanged .text, "
                ".nv.constant0, .nv.merc.nv.info and capsule bytes of the kept kernel. "
                "Proven to load and run on sm_86 (tests/test_cubin_strip.py); the "
                "four stripped sm_100a GEMM cubins loaded and validated on a B200 "
                "(VALIDATION.md, October 3, 2026)."
            ),
            "cases": (
                "Smoke and throughput cases of harness/workloads/moe.py vary "
                "tokens, local_expert_offset and the routing distribution (the "
                "definition fixes the other dims); every case's (tokens, offset) "
                "is a probe fixture, so its kernel choice is upstream's."
            ),
            "upstream_coverage": {
                "resources/moe.json": "constant axes = this package's constants; "
                "stage references chained against its run "
                "(tests/test_official_references.py)",
                "resources/benchmark_workloads/moe.jsonl": "all 19 rows (tokens, "
                "recorded local_expert_offset) are throughput cases",
                "tests/test_trtllm_gen_fused_moe.py::test_moe_quantization_classes"
                "[NoShuffle_MajorK-DSv3-FP8_Block-*-1024-{1,1024}]": "smoke cases "
                "tokens 1 and 1024, offset 0, upstream's logits/bias distribution; "
                "its hidden/intermediate sizes, 256 local experts and tile 32 "
                "(1024 tokens) are fixed by the definition to 7168/2048/32/8",
                "enforced_by": "tests/test_moe.py::UpstreamCoverage",
            },
            "adaptations": [
                "Routing/activation/finalize are built UsePdl=false and launched "
                "without programmatic stream serialization, as the previous package "
                "did; upstream uses UsePdl=true kernels with PDL (identical parameter "
                "layout; recorded in each sidecar's upstream_attrs).",
                "GEMM launches set the 1x1x1 cluster attribute but not the redundant "
                "CLUSTER_SCHEDULING_POLICY_PREFERENCE=DEFAULT and PDL=0 attributes "
                "upstream's launchKernel adds.",
                "routingIndicesCoopKernel accumulates into an expert-count histogram "
                "that upstream's routingMainKernel zeroes; the workload owns that "
                "scratch: run() zeroes it before the launch, prepare() hands each "
                "timed launch the next of 1024 pre-zeroed histograms.",
                "Kernel templates copied verbatim (checked) without the host "
                "launchers, so each TU instantiates one kernel; a PyTorch-free "
                "stand-in for <c10/util/Exception.h> provides TORCH_CHECK/TORCH_WARN.",
            ],
            "builds": dict(sorted(builds.items())),
        }
        self._write_json(PROVENANCE, provenance)
