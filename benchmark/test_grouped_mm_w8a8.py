# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standard grouped_mm_w8a8 benchmark with explicit baseline provenance.

Real-shape source (external MetaX server dataset):
/data/fzy/real_shapes_20261019/grouped_gemm_nt_i8i8bf16_nopad
The capture bundle contains shapes.json and metadata/*.jsonl. Prepared benchmark
JSON stays outside the repository and is supplied with --shape_file PATH.
It preserves per-scenario call counts, captured first-observation expert counts,
cumulative offsets and tensor layouts. Operand values and scales are generated
deterministically, not captured model values. INT8 uses ContinuousGroupedGEMM.
Prepared external INT8/FP8 diagnostic JSON files are also accepted; their
classification must identify synthetic routing.

Both routes receive A, As, B, Bs, preallocated BF16 C, m_indices and m_rows.
The mctlass adapter builds segment offsets and identity weight indices on the
GPU every invocation, selects the native block size, computes the native INT8
tile prefix, and calls the unmodified ContinuousGroupedGEMM Torch interface.
FP8 diagnostic cases retain their validated Torch tile-prefix adapter.
No dynamic routing metadata is cached or copied to the CPU. Initial input/output
allocation, JIT and object construction are outside event timing. The candidate's
own expert-prefix construction is inside timing. Accuracy tests belong to
tests/test_grouped_mm_w8a8.py. Historical reference=deep_gemm remains explicit.
"""

import json
from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl

import flag_gems

from . import base, consts


@triton.jit
def _mctlass_routing(Rows, Segments, Weights, G: tl.constexpr, BLOCK: tl.constexpr):
    # Only adapt metadata; all GEMM work remains in the native mctlass library.
    groups = tl.arange(0, BLOCK)
    counts = tl.load(Rows + groups, groups < G, other=0)
    tl.store(Segments, 0)
    tl.store(Segments + groups + 1, tl.cumsum(counts, 0), groups < G)
    tl.store(Weights + groups, groups, groups < G)


class GroupedMMW8A8Benchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        path = Path(shape_file_path or base.Config.shape_file)
        default = Path(__file__).with_name("core_shapes.yaml")
        if path.resolve() == default.resolve():
            pytest.skip(
                "External grouped_mm_w8a8 dataset required: pass --shape_file "
                "PATH to a prepared benchmark JSON. See this module docstring "
                "for the real-shape source."
            )
        self.dataset = json.loads(path.read_text())
        self.shapes = self.dataset["shapes"]
        self.use_deepgemm = self.dataset.get("reference") == "deep_gemm"
        if self.use_deepgemm or self.dataset.get("input_dtype") == "int8":
            requested = base.Config.user_desired_dtypes
            if requested and requested != [torch.int8]:
                raise ValueError("This captured export requires INT8 inputs")
            self.to_bench_dtypes = [torch.int8]
            self.torch_op = (
                self.deepgemm_reference if self.use_deepgemm else self.mctlass_reference
            )
            self.gems_op = flag_gems.grouped_mm_w8a8
        self.shape_desc = "G, M, N, K; " + self.dataset["classification"]
        for row in self.shapes:
            lengths = row["lengths"]
            ends, total = [], 0
            for length in lengths:
                if not isinstance(length, int) or length < 0:
                    raise ValueError("Expert lengths must be nonnegative integers")
                total += length
                ends.append(total)
            if (
                len(lengths) != row["G"]
                or total != row["M"]
                or ends != row["offs"]
                or row["source_count"] <= 0
                or row["K"] % 128
            ):
                raise ValueError(f"Invalid grouped-MM export row {row['index']}")

    def get_case_iter(self, dtype):
        for ordinal, row in enumerate(self.shapes):
            yield self._case_from_plan(
                dtype,
                ordinal,
                consts.BenchmarkCasePlan(
                    shape={key: row[key] for key in ("G", "M", "N", "K")},
                    params={
                        key: row[key]
                        for key in (
                            "index",
                            "source_index",
                            "stage",
                            "source_shape",
                            "source_count",
                            "source_sources",
                            "offs",
                            "routing_seed",
                        )
                    }
                    | {
                        "classification": self.dataset["classification"],
                        "source_sha256": self.dataset["source_sha256"],
                        "source_files_sha256": self.dataset.get("source_files_sha256"),
                        "scale_layout": ("FP32 As[M,1], Bs[G,N,1]"),
                        "scenario": row.get("scenario"),
                        "captured_tensor_metadata": row.get("tensor_metadata"),
                        "routing_origin": self.dataset.get("routing_origin"),
                        "baseline": (
                            "MetaX DeepGEMM contiguous, backend=mctlassEx, full wrapper"
                            if self.use_deepgemm
                            else (
                                "ContinuousGroupedGEMM Torch API; routing + "
                                "block selection + tile prefix included; "
                                "FP8 uses Torch tile-prefix adapter"
                            )
                        ),
                    },
                    builder_args=(row,),
                ),
            )

    def build_inputs(self, case):
        row = case.builder_args[0].builder_args[0]
        g, m, n, k = (row[key] for key in ("G", "M", "N", "K"))
        dtype = case.dtype
        generator = torch.Generator().manual_seed(
            20261014 + row["index"] + (10000 if dtype == torch.int8 else 0)
        )
        if dtype == torch.int8:
            a = torch.randint(-128, 128, (m, k), dtype=dtype, generator=generator)
            b = torch.randint(-128, 128, (g, n, k), dtype=dtype, generator=generator)
            sa = torch.tensor([2.0 ** (i % 4 - 8) for i in range(m)])
            sb = torch.tensor([2.0 ** (i % 5 - 8) for i in range(g * n)])
        else:
            a = torch.empty((m, k), dtype=dtype)
            for start in range(0, m, 4096):
                end = min(start + 4096, m)
                a[start:end] = (
                    torch.randn((end - start, k), generator=generator) * 0.25
                ).to(dtype)
            b = torch.empty((g, n, k), dtype=dtype)
            for group in range(g):
                b[group] = (torch.randn((n, k), generator=generator) * 0.25).to(dtype)
            sa = torch.tensor([2.0 ** (i % 4 - 1) for i in range(m)])
            sb = torch.tensor([2.0 ** (i % 3 - 1) for i in range(g * n)])
        a, b = a.cuda(), b.cuda()
        sa, sb = sa.float().cuda(), sb.float().reshape(g, n).cuda()
        if self.use_deepgemm:
            module = pytest.importorskip("deep_gemm")
            self.deepgemm = module.m_grouped_gemm_int8_int8_bf16_nt_contiguous
            lengths = torch.tensor(row["lengths"], dtype=torch.int32, device="cuda")
            indices = torch.tensor(
                [
                    group
                    for group, length in enumerate(row["lengths"])
                    for _ in range(length)
                ],
                dtype=torch.int32,
                device="cuda",
            )
            output = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
            return a, sa[:, None], b, sb[:, :, None], output, indices, lengths
        self.native_config = {"n_group": g}
        if dtype != torch.int8:
            self.native_config.update(
                is_blockwise=True,
                use_fp8=True,
                is_scale_a_1d=True,
                is_scale_b_1d=True,
                scale_a_layout="k-major",
                scale_b_layout="k-major",
                M=m,
                m_indices=torch.tensor(
                    [
                        group
                        for group, length in enumerate(row["lengths"])
                        for _ in range(length)
                    ],
                    dtype=torch.int32,
                    device="cuda",
                ),
            )
        native_module = pytest.importorskip("mctlassEx.ContinuousGroupedGEMM")
        self.native = native_module.ContinuousGroupedGEMM()
        scratch = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
        lengths = torch.tensor(row["lengths"], dtype=torch.int32, device="cuda")
        indices = torch.repeat_interleave(
            torch.arange(g, dtype=torch.int32, device="cuda"), lengths, output_size=m
        )
        return a, sa[:, None], b, sb[:, :, None], scratch, indices, lengths

    def deepgemm_reference(self, a, sa, b, sb, output, indices, lengths):
        # Preserve the installed public wrapper and explicitly select native INT8.
        # This is a contiguous baseline, not the unavailable MetaX nopad API.
        self.deepgemm(
            (a, sa[:, 0]), (b, sb[:, :, 0]), output, indices, backend="mctlassEx"
        )
        return output

    @staticmethod
    def _fp8_scales(sa, sb, k):
        sa, sb = sa[:, None], sb[:, :, None]
        if k > 128:
            sa, sb = sa.repeat(1, k // 128), sb.repeat(1, 1, k // 128)
        return sa, sb

    def mctlass_reference(self, a, sa, b, sb, output, indices, lengths):
        # Both routes receive identical NT operands and caller-owned BF16 output.
        m, k = a.shape
        g, n, _ = b.shape
        sa, sb = sa[:, 0], sb[:, :, 0]
        segments = torch.empty((g + 1,), dtype=torch.int32, device=a.device)
        weights = torch.empty((g,), dtype=torch.int32, device=a.device)
        _mctlass_routing[(1,)](
            lengths, segments, weights, g, triton.next_power_of_2(g), num_warps=4
        )
        if a.dtype != torch.int8:
            sa, sb = self._fp8_scales(sa, sb, k)
            self.native_config["m_indices"] = indices
        block_m = self.native.get_blocksize_m(
            a, b, output, sa, sb, **self.native_config
        )
        if not isinstance(block_m, int) or block_m <= 0:
            raise RuntimeError(f"Invalid mctlass block size: {block_m}")
        if a.dtype == torch.int8:
            tiles = torch.empty((g + 1,), dtype=torch.int32, device=a.device)
            self.native.compute_m_num_tiles_indptr(
                a, b, output, segments, tiles, block_m, **self.native_config
            )
        else:
            counts = torch.div(lengths + block_m - 1, block_m, rounding_mode="floor")
            tiles = torch.cat(
                (
                    torch.zeros((1,), dtype=torch.int32, device=a.device),
                    torch.cumsum(counts, dim=0, dtype=torch.int32),
                )
            )
        self.native(
            [m, n, k],
            a,
            b,
            output,
            sa,
            sb,
            None,
            segments,
            weights,
            tiles,
            **self.native_config,
        )
        return output


@pytest.mark.grouped_mm_w8a8
@pytest.mark.skipif(
    flag_gems.vendor_name != "metax", reason="Requires MetaX mctlass Torch bindings"
)
def test_grouped_mm_w8a8():
    bench = GroupedMMW8A8Benchmark(
        op_name="grouped_mm_w8a8",
        torch_op=None,
        gems_op=flag_gems.grouped_mm_w8a8,
        dtypes=[torch.int8, torch.float8_e4m3fn],
    )
    bench.torch_op = bench.mctlass_reference
    bench.run()
