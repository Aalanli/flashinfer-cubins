"""NVFP4 helpers shared by the NVFP4 workloads: E2M1 decoding, the input
distributions, the one blocked scale relayout (and each package's view of it)
and the chunked reference GEMM."""

import unittest
from unittest import mock

import torch

from harness.workloads import nvfp4_dual_gemm, nvfp4_gemm, nvfp4_group_gemm
from harness.workloads import quantization as q


class Device:
    device = torch.device("cpu")


def kernel_pack(scales: torch.Tensor) -> torch.Tensor:
    """Element-by-element transcription of impls/nvfp4_dual_gemm/kernels/
    pack_scales.cu (the previous packages' relayout kernel)."""
    rows, cols, batches = scales.shape
    flat = scales.permute(2, 0, 1).contiguous().view(torch.uint8).flatten().tolist()
    padded_rows, padded_cols = (rows + 127) // 128 * 128, (cols + 3) // 4 * 4
    size = padded_rows * padded_cols
    out = []
    for i in range(batches * size):
        b, offset = divmod(i, size)
        tile, inside = divmod(offset, 512)
        row = (
            (tile // (padded_cols // 4)) * 128
            + inside // 16
            + ((inside % 16) // 4) * 32
        )
        col = (tile % (padded_cols // 4)) * 4 + inside % 4
        out.append(
            flat[(b * rows + row) * cols + col] if row < rows and col < cols else 0
        )
    return torch.tensor(out, dtype=torch.uint8)


def task_to_blocked(input_matrix: torch.Tensor) -> torch.Tensor:
    """``to_blocked`` of resources/nvfp4_*/reference.py, verbatim logic (with
    the padding the group task's version adds)."""
    rows, cols = input_matrix.shape
    n_row_blocks, n_col_blocks = -(-rows // 128), -(-cols // 4)
    padded = torch.nn.functional.pad(
        input_matrix, (0, n_col_blocks * 4 - cols, 0, n_row_blocks * 128 - rows)
    )
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.flatten()


def random_scales(rows, cols, batches, seed):
    g = torch.Generator().manual_seed(seed)
    data = torch.randint(1, 127, (batches, rows, cols), generator=g, dtype=torch.uint8)
    return data.view(torch.float8_e4m3fn).permute(1, 2, 0)


SHAPES = ((128, 8, 1), (200, 6, 2), (1, 2, 1), (260, 32, 3), (129, 16, 3))


class FP4(unittest.TestCase):
    def test_fp4_nibbles_and_scale(self):
        # Explicit sign, low/high nibble order, block-scale boundary and L axis.
        packed = torch.tensor(
            [0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE] * 2, dtype=torch.uint8
        ).reshape(1, 16, 1)
        scales = torch.tensor([2.0, 0.5]).reshape(1, 2, 1)
        decoded = q.dequant_fp4(packed, scales).flatten()
        values = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]
        )
        torch.testing.assert_close(decoded, torch.cat((2 * values, 0.5 * values)))

    def test_distributions(self):
        """The official generators' value sets (module docstring)."""
        g = torch.Generator().manual_seed(0)
        data, _ = q.make_fp4(Device, g, 64, 256, 2, values="restricted")
        nibbles = torch.cat((data & 15, data >> 4)).unique()
        self.assertEqual(sorted(nibbles.tolist()), [0, 1, 2, 3, 8, 9, 10, 11])
        data, _ = q.make_fp4(Device, g, 64, 256, 2, values="full")
        self.assertEqual(torch.cat((data & 15, data >> 4)).unique().numel(), 16)
        expected = {
            "int0_3": lambda s: set(s.unique().tolist()) == {0, 1, 2, 3},
            "int1_2": lambda s: set(s.unique().tolist()) == {1, 2},
            # zeros and subnormals (below 2**-6) occur, nothing reaches 1.0+
            "unit": lambda s: (s == 0).any()
            and ((s > 0) & (s < 2**-6)).any()
            and s.max() <= 1,
            "uniform": lambda s: s.min() >= 0.125 and s.max() <= 0.875,
        }
        for scales, check in expected.items():
            _, s = q.make_fp4(Device, g, 512, 1024, 1, scales=scales)
            self.assertTrue(check(s.float()), scales)
        with self.assertRaises(ValueError):
            q.make_fp4(Device, g, 4, 32, 1, scales="normal")

    def test_reference_is_chunk_independent_and_exact(self):
        """fp4_matmul equals an FP64 product (exact for integer scales) for any
        chunking, with alpha and l > 1, and never mutates its inputs."""
        g = torch.Generator().manual_seed(1)
        a, sa = q.make_fp4(Device, g, 37, 96, 2, scales="int0_3")
        b, sb = q.make_fp4(Device, g, 300, 96, 2, scales="int0_3")
        before = [t.clone() for t in (a, b, sa, sb)]
        A = q.dequant_fp4(a, sa).double().permute(2, 0, 1)
        B = q.dequant_fp4(b, sb).double().permute(2, 0, 1)
        exact = (0.375 * torch.bmm(A, B.transpose(1, 2))).permute(1, 2, 0)
        for chunk in (q.CHUNK, 1000, 1):
            with mock.patch.object(q, "CHUNK", chunk):
                out = q.fp4_matmul(a, b, sa, sb, alpha=0.375)
                half = q.fp4_matmul(a, b, sa, sb, alpha=0.375, out_dtype=torch.half)
            self.assertTrue(torch.equal(out.double(), exact))
            self.assertTrue(torch.equal(half, exact.half()))
            self.assertTrue(out.permute(2, 0, 1).is_contiguous())
        for old, new in zip(before, (a, b, sa, sb)):
            self.assertTrue(torch.equal(old.view(torch.uint8), new.view(torch.uint8)))


class BlockedScales(unittest.TestCase):
    """``quantization.to_blocked_scales`` is the only relayout; it equals the
    previous packages' pack_scales kernel and the NVIDIA tasks' to_blocked,
    places each scale at CUTLASS's tile_atom_to_shape_SF offset, round-trips,
    and each package's view addresses the same bytes."""

    def test_matches_kernel_task_and_cutlass(self):
        for rows, cols, batches in SHAPES:
            with self.subTest(rows=rows, cols=cols, batches=batches):
                scales = random_scales(rows, cols, batches, rows + cols)
                blocked = q.to_blocked_scales(scales)
                self.assertTrue(blocked.is_contiguous())
                memory = blocked.view(torch.uint8).flatten()
                self.assertTrue(torch.equal(memory, kernel_pack(scales)))
                per_batch = memory.view(batches, -1)
                for b in range(batches):
                    expected = task_to_blocked(scales[:, :, b].view(torch.uint8))
                    self.assertTrue(torch.equal(per_batch[b], expected))
                # (((32,4),mnb),((16,4),kb),(1,l)):(((16,4),512kb),((0,1),512),(_,size))
                kb, size = blocked.shape[2], blocked[0].numel()
                for r, c, b in ((0, 0, 0), (rows - 1, cols - 1, batches - 1)):
                    offset = (
                        (r % 32) * 16
                        + ((r // 32) % 4) * 4
                        + c % 4
                        + (c // 4) * 512
                        + (r // 128) * 512 * kb
                        + b * size
                    )
                    self.assertEqual(memory[offset], scales.view(torch.uint8)[r, c, b])
                back = q.from_blocked_scales(blocked, rows, cols)
                self.assertTrue(
                    torch.equal(back.view(torch.uint8), scales.view(torch.uint8))
                )

    def test_package_views(self):
        for rows, cols, batches in SHAPES:
            k = 16 * cols
            scales = random_scales(rows, cols, batches, 7 * rows)
            memory = q.to_blocked_scales(scales).view(torch.uint8).flatten()
            with self.subTest(view="nvfp4_gemm (task sfa_permuted)", rows=rows):
                task = nvfp4_gemm.to_task_scales(scales)
                self.assertEqual(
                    tuple(task.shape), nvfp4_gemm.blocked_scale_shape(rows, k, batches)
                )
                # reference.py create_scale_factor_tensors index map.
                ii, jj = torch.meshgrid(
                    torch.arange(rows), torch.arange(cols), indexing="ij"
                )
                for b in range(batches):
                    picked = task[
                        ii % 32, (ii % 128) // 32, ii // 128, jj % 4, jj // 4, b
                    ]
                    self.assertTrue(
                        torch.equal(
                            picked.view(torch.uint8), scales[:, :, b].view(torch.uint8)
                        )
                    )
                back = nvfp4_gemm.from_task_scales(task, rows, cols)
                self.assertTrue(
                    torch.equal(back.view(torch.uint8), scales.view(torch.uint8))
                )
            with self.subTest(view="nvfp4_dual_gemm (flat)", rows=rows):
                packed = nvfp4_dual_gemm.pack_scales_reference(scales)
                self.assertEqual(packed.shape[0], batches)
                self.assertTrue(torch.equal(packed.view(torch.uint8).flatten(), memory))
                back = nvfp4_dual_gemm.unpack_scales_reference(packed, rows, cols)
                self.assertTrue(
                    torch.equal(back.view(torch.uint8), scales.view(torch.uint8))
                )
            if batches == 1:
                with self.subTest(view="nvfp4_group_gemm (one group)", rows=rows):
                    group = nvfp4_group_gemm.to_blocked(scales[:, :, 0])
                    self.assertEqual(
                        tuple(group.shape), nvfp4_group_gemm.blocked_shape(rows, k)
                    )
                    self.assertTrue(
                        torch.equal(group.view(torch.uint8).flatten(), memory)
                    )
                    back = nvfp4_group_gemm.from_blocked(group, rows, cols)
                    self.assertTrue(
                        torch.equal(
                            back.view(torch.uint8), scales[:, :, 0].view(torch.uint8)
                        )
                    )


if __name__ == "__main__":
    unittest.main()
