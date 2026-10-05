"""Per-kernel PyTorch references composed into the official definitions.

Every workload is one kernel; multi-kernel upstream pipelines are split into
per-kernel workloads whose references cover only their own kernel. These tests
restore the end-to-end check against the downloaded official executable
definitions (``resources/<name>.json``, key ``"reference"``): for every smoke
case they build the official-format inputs, run the official ``run(...)``, and
chain the stage references (Python only, never native kernels) in pipeline
order, once per architecture path with a distinct decomposition:

==============  =======  ==================================================
definition      arch     composed stages
==============  =======  ==================================================
rmsnorm         both     rmsnorm (single kernel)
dsa_attention   sm_86    pack_sparse -> decode -> fix_empty
dsa_attention   sm_100a  pack -> mla -> unpack
dsa_indexer     sm_100a  metadata -> logits (+ top-k recovered from logits)
gdn_decode      sm_100a  gdn_gates -> gdn_chunk (B one-token sequences)
gdn_prefill     sm_100a  gdn_gates -> gdn_chunk
gqa_decode      sm_86    batch_decode (single kernel)
gqa_decode      sm_100a  plan -> gather -> fmha -> finish
moe             sm_100a  routing_main -> routing_{cluster,coop} -> gemm1 ->
                         activation -> gemm2 -> finalize{,_vec}
==============  =======  ==================================================

The chains run on CUDA (TF32 disabled) and are skipped without a GPU.
"""

from __future__ import annotations

import json
import sys
import unittest
from functools import cache
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness.workload import CaseSpec, Workload  # noqa: E402
from harness.workloads import dsa_attention as da  # noqa: E402
from harness.workloads import dsa_indexer as di  # noqa: E402
from harness.workloads import gdn as gd  # noqa: E402
from harness.workloads import gqa_decode as gq  # noqa: E402
from harness.workloads import moe  # noqa: E402
from harness.workloads import rmsnorm as rn  # noqa: E402

DEVICE = "cuda"

# BF16 outputs computed in FP32 by both sides with a different operation order:
# results differ by about one BF16 rounding step (2^-8 relative). The value is
# the attention stages' own tolerance (decode, MLA, FMHA, batch decode).
BF16_TOL = 1e-2


@cache
def official(name: str) -> dict[str, Any]:
    """Namespace of the trusted, pinned official definition's reference code."""
    path = ROOT / "resources" / f"{name}.json"
    namespace: dict[str, Any] = {}
    exec(  # trusted pinned definition
        compile(json.loads(path.read_text())["reference"], str(path), "exec"),
        namespace,
    )
    return namespace


def smoke(workload: Workload) -> list[CaseSpec]:
    return [c for c in workload.get_cases() if c.suite == "smoke"]


class _Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("the official chains run on CUDA")
        cls._tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False

    @classmethod
    def tearDownClass(cls) -> None:
        torch.backends.cuda.matmul.allow_tf32 = cls._tf32


class OfficialReferences(_Base):
    def test_rmsnorm(self):
        run = official("rmsnorm")["run"]
        device = DEVICE
        workload = rn.RMSNorm(None, device=device)
        for case in smoke(workload):
            with self.subTest(case=case.name):
                inputs = workload.get_inputs(case)
                expected = (run(*inputs),)
                workload.validate(expected, workload.get_reference(inputs))

    # -- dsa_attention -------------------------------------------------------

    def check_dsa_attention(self, device: str, case: CaseSpec, actual: tuple):
        """Final (output, lse) against the official run: BF16 tolerance; empty
        rows must be exactly (0, -inf) as officially defined (no NaN)."""
        inputs = da.DSAAttentionPack(None, device=device).sparse_inputs(case)
        expected = official("dsa_attention")["run"](*inputs)
        Workload.assert_close(expected, actual, rtol=BF16_TOL, atol=BF16_TOL)
        empty = ~(inputs[4] >= 0).any(1)
        self.assertTrue(bool((actual[0][empty] == 0).all()))
        self.assertTrue(bool((actual[1][empty] == -torch.inf).all()))

    def test_dsa_attention_sm86(self):
        device = DEVICE
        pack = da.DSAAttentionPackSparse(None, device=device)
        decode = da.DSAAttentionDecode(None, device=device)
        fix = da.DSAAttentionFixEmpty(None, device=device)
        for case in smoke(pack):
            inputs = pack.sparse_inputs(case)
            with self.subTest(case=case.name):
                q, qp, ckv, kpe, indices, scale = inputs
                indptr, last, requests, tiles, chunk, packed = pack.get_reference(
                    (indices,)
                )
                if not packed.numel():
                    # All rows empty: decode never reads the indices, but
                    # takes a valid pointer (as its own get_inputs does).
                    packed = torch.zeros(1, dtype=torch.int32, device=device)
                out, lse = decode.get_reference(
                    (q, qp, ckv, kpe, indptr, packed, last, requests)
                    + (tiles, chunk, scale)
                )
                final = fix.get_reference((indptr, out, lse))
                self.check_dsa_attention(device, case, final)

    def test_dsa_attention_sm100a(self):
        device = DEVICE
        pack = da.DSAAttentionPack(None, device=device)
        mla = da.DSAAttentionMLA(None, device=device)
        unpack = da.DSAAttentionUnpack(None, device=device)
        for case in smoke(pack):
            inputs = pack.sparse_inputs(case)
            with self.subTest(case=case.name):
                q, qp, ckv, kpe, indices, scale = inputs
                padded_q, padded_qp, table, lengths = pack.get_reference(
                    (q, qp, indices)
                )
                out, lse = mla.get_reference(
                    (padded_q, padded_qp, ckv, kpe, table, lengths, scale)
                )
                final = unpack.get_reference((out, lse, indices))
                self.check_dsa_attention(device, case, final)

    # -- dsa_indexer -----------------------------------------------------------

    def test_dsa_indexer(self):
        """metadata -> logits; the official reference also applies top-k.

        Logits are checked on the official definition's notion of a score,
        computed independently (per row) with its own ``dequant_fp8_kv_cache``,
        on columns ``< seq_len`` (later columns of the last 256-token split are
        written against padding keys, unwritten ones are NaN; neither is
        defined officially), with the logits stage's validate. The physical
        top-k set recovered from the composed logits must then equal the
        official result exactly.
        """
        namespace = official("dsa_indexer")
        device = DEVICE
        metadata = di.DSAIndexerMetadata(None, device=device)
        logits_stage = di.DSAIndexerLogits(None, device=device)
        for case in smoke(metadata):
            inputs = metadata.indexer_inputs(case)
            with self.subTest(case=case.name):
                q, cache_, weights, lengths, table = inputs
                (schedule,) = metadata.get_reference((lengths,))
                (logits,) = logits_stage.get_reference(inputs + (schedule,))

                keys = namespace["dequant_fp8_kv_cache"](cache_)
                expected = torch.full_like(logits, torch.nan)
                for row, length in enumerate(lengths.tolist()):
                    pages = table[row, : -(-length // di.PAGE_SIZE)].long()
                    logical = keys[pages].reshape(-1, di.HEAD_DIM)[:length]
                    scores = (q[row].float() @ logical.T).relu()
                    expected[row, :length] = weights[row] @ scores
                logits_stage.validate((expected,), (logits,))

                recovered = torch.full(
                    (q.shape[0], 2048), -1, dtype=torch.int32, device=device
                )
                for row, length in enumerate(lengths.tolist()):
                    count = min(length, 2048)
                    row_logits = logits[row, :length]
                    self.assertFalse(bool(torch.isnan(row_logits).any()))
                    idx = row_logits.topk(count).indices
                    recovered[row, :count] = (
                        table[row, idx // di.PAGE_SIZE] * di.PAGE_SIZE
                        + (idx % di.PAGE_SIZE).int()
                    )
                (topk,) = namespace["run"](*inputs)
                torch.testing.assert_close(
                    recovered.sort(-1).values, topk.sort(-1).values, rtol=0, atol=0
                )

    # -- gdn (gdn_decode and gdn_prefill) ---------------------------------------

    def check_gdn(self, decode: bool) -> None:
        """gdn_gates -> gdn_chunk against the official run, for every gdn_gates
        smoke case of the definition (decode: ``batch`` cases, viewed as
        ``[B, 1, H, 128]``; prefill: ragged cases). Both sides evaluate the
        same token-serial FP32 formula (gates in FP64 then rounded to FP32
        here, FP32 in the official code): the FP32 state agrees to 2.2e-6
        (|S| <= 3) and the BF16 output to one rounding step on <= 0.1% of the
        elements (measured), so state rtol = atol = 1e-5 and output rtol
        8e-3 (one BF16 step), atol 1e-4 RMS, relative L2 1e-3. Empty prefill
        sequences are skipped: the official code returns a zero state for
        them, the kernel (and gdn_chunk's reference) the initial state."""
        run = official("gdn_decode" if decode else "gdn_prefill")["run"]
        device = DEVICE
        gates = gd.GDNGates(None, device=device)
        chunk = gd.GDNChunk(None, device=device)
        for case in smoke(gates):
            lengths = gd.case_lengths(case.params)
            if ("batch" in case.params) != decode or 0 in lengths:
                continue
            with self.subTest(case=case.name):
                q, k, v, state, alog, a, bias, b, cu, scale = gates.definition_inputs(
                    case
                )
                gate, beta, offsets = gates.get_reference(
                    (alog, a, bias, b, None if decode else cu)
                )
                output, new_state = chunk.get_reference(
                    (q, k, v, state, gate, beta, offsets, scale)
                )
                if decode:
                    n = q.shape[0]
                    expected = run(
                        q.view(n, 1, 4, 128),
                        k.view(n, 1, 4, 128),
                        v.view(n, 1, 8, 128),
                        state,
                        alog,
                        a.view(n, 1, 8),
                        bias,
                        b.view(n, 1, 8),
                        float(scale),
                    )
                    expected = (expected[0].view(n, 8, 128), expected[1])
                else:
                    expected = run(q, k, v, state, alog, a, bias, b, cu, float(scale))
                gd.scaled_close(
                    "output",
                    expected[0],
                    output,
                    rtol=8e-3,
                    atol_rms=1e-4,
                    rel_l2=1e-3,
                )
                torch.testing.assert_close(new_state, expected[1], rtol=1e-5, atol=1e-5)

    def test_gdn_decode(self):
        self.check_gdn(decode=True)

    def test_gdn_prefill(self):
        self.check_gdn(decode=False)

    # -- gqa_decode ----------------------------------------------------------------

    def test_gqa_decode_sm86(self):
        """batch_decode alone; requests without KV are undefined in that kernel
        (NaN in its reference, exactly the empty requests, excluded by its
        validate)."""
        run = official("gqa_decode")["run"]
        device = DEVICE
        workload = gq.GQADecodeBatchDecode(None, device=device)
        for case in smoke(workload):
            inputs = workload.official_inputs(case)
            with self.subTest(case=case.name):
                actual = workload.get_reference(inputs)
                empty = inputs[3][1:] == inputs[3][:-1]
                self.assertTrue(torch.equal(torch.isnan(actual[1]).any(-1), empty))
                workload.validate(actual, run(*inputs))

    def test_gqa_decode_sm100a(self):
        run = official("gqa_decode")["run"]
        device = DEVICE
        plan = gq.GQADecodePlan(None, device=device)
        gather = gq.GQADecodeGather(None, device=device)
        fmha = gq.GQADecodeFMHA(None, device=device)
        finish = gq.GQADecodeFinish(None, device=device)
        for case in smoke(plan):
            inputs = plan.official_inputs(case)
            with self.subTest(case=case.name):
                q, k, v, indptr, indices, scale = inputs
                sms = plan.scalar(plan.plan_sms(), torch.int32)
                schedule = plan.get_reference((indptr, sms))
                packed_k, packed_v = gather.get_reference(
                    (k, v, indptr, indices, schedule[1])
                )
                out, lse = fmha.get_reference((q, packed_k, packed_v, *schedule, scale))
                actual = finish.get_reference((out, indptr, lse))
                Workload.assert_close(
                    run(*inputs), actual, rtol=BF16_TOL, atol=BF16_TOL
                )

    # -- moe ---------------------------------------------------------------------------

    def test_moe(self):
        """routing -> FC1 -> activation -> FC2 -> finalize vs the official run.

        The official reference computes in FP32 throughout with FP32 routing
        weights. The pipeline (upstream and its references) stores the routing
        weights in BF16, requantizes FC1's output and the SwiGLU output to E4M3
        with a DeepSeek scale per 128 columns (3 mantissa bits: up to 2^-4
        relative rounding error per element), and stores FC2's output and the
        result in BF16. The two E4M3 requantizations dominate: on the smoke
        cases the per-token relative L2 error is ~0.045 (measured; also for 40
        tokens), and it is required to stay below 0.08. A wrong expert, weight,
        permutation or scale gives an O(1) error. Tokens with no local expert
        must be exactly zero on both sides.

        The ``ties`` smoke case is left out: the official ``torch.topk`` leaves
        the order of tied experts undefined (upstream and the stage references
        prefer the lower index; ``tests/test_moe.py`` checks that).
        """
        run = official("moe")["run"]

        def pick(tokens: int, *classes: type[moe._MoEWorkload]):
            (chosen,) = [c for c in classes if c.serves(tokens)]
            return chosen

        device = DEVICE
        main = moe.MoERoutingMain(None, device=device)
        cases = [c for c in smoke(main) if c.params["routing"] != "ties"]
        for case in cases:
            with self.subTest(case=case.name):
                tokens, offset = case.params["tokens"], case.params["offset"]
                logits, bias, offset_t, scale_t = main.get_inputs(case)
                hidden, hidden_scale = moe.case_activations(main.device, case)
                w1, s1 = moe.official_weights(main.device, 1)
                w2, s2 = moe.official_weights(main.device, 2)

                packed, weights = main.get_reference((logits, bias, offset_t, scale_t))[
                    :2
                ]
                indices_cls = pick(tokens, moe.MoERoutingCluster, moe.MoERoutingCoop)
                routed = indices_cls(None, device=device).get_reference(
                    (logits, bias, packed, offset_t, scale_t)
                )
                if indices_cls.writes_weights:
                    weights, routed = routed[0], routed[1:]
                expanded, to_token, total, batch, limit, ctas = routed
                gemm1 = pick(tokens, moe.MoEGEMM1, moe.MoEGEMM1Persistent)
                g1, g1_scale = gemm1(None, device=device).get_reference(
                    (hidden, hidden_scale, w1, s1, to_token, total, batch)
                    + (limit, ctas, weights, expanded)
                )
                act, act_scale = moe.MoEActivation(None, device=device).get_reference(
                    (g1, g1_scale, expanded, total)
                )
                gemm2 = pick(tokens, moe.MoEGEMM2, moe.MoEGEMM2Persistent)
                (g2,) = gemm2(None, device=device).get_reference(
                    (act, act_scale, w2, s2, total, batch, limit, ctas, expanded)
                    + (main.scalar(tokens, torch.int32),)
                )
                finalize = pick(tokens, moe.MoEFinalize, moe.MoEFinalizeVec)
                (actual,) = finalize(None, device=device).get_reference(
                    (g2, weights, expanded, total)
                )
                del g1, g1_scale, act, act_scale, g2

                expected = run(
                    logits,
                    bias,
                    hidden,
                    hidden_scale,
                    w1,
                    s1,
                    w2,
                    s2,
                    offset,
                    moe.ROUTED_SCALING_FACTOR,
                )
                self.assertEqual(
                    (actual.shape, actual.dtype), (expected.shape, expected.dtype)
                )
                a, e = actual.float(), expected.float()
                self.assertFalse(bool(torch.isnan(a).any()))
                norm = e.norm(dim=1)
                zero = norm == 0
                self.assertTrue(bool((a[zero] == 0).all()))
                if (~zero).any():
                    relative = (a - e)[~zero].norm(dim=1) / norm[~zero]
                    worst = float(relative.max())
                    self.assertLess(worst, 0.08, f"per-token relative L2 {worst}")
                del expected, actual


if __name__ == "__main__":
    unittest.main()
