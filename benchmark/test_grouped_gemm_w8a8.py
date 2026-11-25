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

"""Standard grouped_gemm_w8a8 benchmark with explicit baseline provenance.

Real-shape source (external MetaX server dataset):
/data/fzy/real_shapes_20261019/grouped_gemm_nt_i8i8bf16_nopad
The capture bundle contains shapes.json and metadata/*.jsonl. Prepared benchmark
JSON stays outside the repository and is supplied with --shape_file PATH.
It preserves per-scenario call counts, captured first-observation expert counts,
cumulative offsets and tensor layouts. Operand values and scales are generated
deterministically, not captured model values. INT8 uses ContinuousGroupedGEMM.
Prepared external INT8/FP8 diagnostic JSON files are also accepted; their
classification must identify synthetic routing.

Without --shape_file, use the existing core_shapes.yaml geometries with
synthetic balanced expert routing and unit weights. These default cases are
not captured model calls and do not establish invocation-weighted performance.

Both routes receive A, As, B, Bs, preallocated BF16 C, m_indices and m_rows.
The mctlass adapter builds segment offsets and identity weight indices on the
GPU every invocation, selects the native block size, computes the native INT8
tile prefix, and calls the unmodified ContinuousGroupedGEMM Torch interface.
FP8 diagnostic cases retain their validated Torch tile-prefix adapter.
No dynamic routing metadata is cached or copied to the CPU. Initial input/output
allocation, JIT and object construction are outside event timing. The candidate's
own expert-prefix construction is inside timing. Accuracy tests belong to
tests/test_grouped_gemm_w8a8.py. Historical reference=deep_gemm remains explicit.

The block-scaled FP8 route reuses the core geometries and accepts captured JSON
with cases=[{shape:[M,N,K,E], count:...}]. Counts describe invocation frequency;
FP8 values, block scales and near-uniform expert routing are regenerated.
Its DeepGEMM baseline times only prepacked native FP8 GEMM: routing, padding,
scale-layout preparation and unpadding are outside timing. The compact candidate
includes its own task preparation. This route was validated with DeepGEMM 2.8.0
from vLLM 0.32.0; INT8 captures supply shapes and counts only.
"""

import json
from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl

import flag_gems
from tests.test_grouped_gemm_w8a8 import (
    deepgemm_grouped_gemm,
    make_grouped_gemm_case,
    make_grouped_gemm_fp8_case,
)

from . import base, consts
from .base import Benchmark


@triton.jit
def _mctlass_routing(Rows, Segments, Weights, G: tl.constexpr, BLOCK: tl.constexpr):
    # Only adapt metadata; all GEMM work remains in the native mctlass library.
    groups = tl.arange(0, BLOCK)
    counts = tl.load(Rows + groups, groups < G, other=0)
    tl.store(Segments, 0)
    tl.store(Segments + groups + 1, tl.cumsum(counts, 0), groups < G)
    tl.store(Weights + groups, groups, groups < G)


class GroupedGEMMW8A8Benchmark(base.Benchmark):
    @staticmethod
    def _default_dataset(shapes):
        rows = []
        for index, (m, n, k, g) in enumerate(shapes, 1):
            lengths = [m // g + (expert < m % g) for expert in range(g)]
            offsets, total = [], 0
            for length in lengths:
                total += length
                offsets.append(total)
            rows.append(
                {
                    "index": index,
                    "source_index": index,
                    "G": g,
                    "M": m,
                    "N": n,
                    "K": k,
                    "lengths": lengths,
                    "offs": offsets,
                    "stage": "gate_up" if n < k else "down",
                    "source_shape": [m, n, k, g],
                    "source_count": 1,
                    "source_sources": ["benchmark/core_shapes.yaml"],
                    "routing_seed": None,
                    "scenario": "synthetic_core",
                }
            )
        return {
            "classification": "Synthetic balanced routing; unit weights, not model call counts",
            "routing_origin": "synthetic_balanced",
            "source_sha256": None,
            "shapes": rows,
        }

    def set_shapes(self, shape_file_path=None):
        path = Path(shape_file_path or base.Config.shape_file)
        default = Path(__file__).with_name("core_shapes.yaml")
        if path.resolve() == default.resolve():
            super().set_shapes(str(path))
            self.dataset = self._default_dataset(self.shapes)
        else:
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
            self.gems_op = flag_gems.grouped_gemm_w8a8
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


class GroupedGemmInt8Benchmark(Benchmark):
    DEFAULT_SHAPES = [(45, 512, 6144, 257), (45, 6144, 256, 257)]
    DEFAULT_SHAPE_DESC = "M,N,K,E"

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            m, n, k, e = shape[:4]
            if len(shape) == 5:
                lengths = shape[4]
            else:
                lengths = [m // e + (expert < m % e) for expert in range(e)]
            if len(lengths) != e or sum(lengths) != m:
                raise ValueError("expert row lengths must match M and E")
            yield make_grouped_gemm_case(lengths, n, k, self.device)


@pytest.mark.grouped_gemm_w8a8
@pytest.mark.skipif(
    not (
        flag_gems.vendor_name == "nvidia"
        and "grouped_gemm_w8a8_fp8" in vars(flag_gems)
        or flag_gems.vendor_name in ("metax", "thead")
        and "grouped_gemm_w8a8" in vars(flag_gems)
    ),
    reason="requires a registered grouped GEMM implementation",
)
def test_grouped_gemm_w8a8():
    if flag_gems.vendor_name == "nvidia":
        pytest.importorskip("deep_gemm")
        bench = GroupedFp8Benchmark(
            op_name="grouped_gemm_w8a8",
            torch_op=None,
            gems_op=flag_gems.grouped_gemm_w8a8_fp8,
            dtypes=[torch.float8_e4m3fn],
        )
        bench.torch_op = bench.native_reference
    elif flag_gems.vendor_name == "metax":
        bench = GroupedGEMMW8A8Benchmark(
            op_name="grouped_gemm_w8a8",
            torch_op=None,
            gems_op=flag_gems.grouped_gemm_w8a8,
            dtypes=[torch.int8, torch.float8_e4m3fn],
        )
        bench.torch_op = bench.mctlass_reference
    else:
        pytest.importorskip("deep_gemm")
        bench = GroupedGemmInt8Benchmark(
            op_name="grouped_gemm_w8a8",
            torch_op=deepgemm_grouped_gemm,
            gems_op=flag_gems.grouped_gemm_w8a8,
            dtypes=[torch.bfloat16],
        )
    bench.run()


@triton.jit
def routing_offsets(
    Counts, Starts, Padded, E: tl.constexpr, ALIGN: tl.constexpr, BE: tl.constexpr
):
    e = tl.arange(0, BE)
    count = tl.load(Counts + e, e < E, 0)
    padded = tl.cdiv(count, ALIGN) * ALIGN
    end = tl.cumsum(count)
    padded_end = tl.cumsum(padded)
    tl.store(Starts + e, end - count, e < E)
    tl.store(Padded + e, padded_end - padded, e < E)
    tl.store(Padded + E, tl.sum(padded))


@triton.jit
def clear_layout(Layout, CAP: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Layout + row, -1, row < CAP)


@triton.jit
def pack_inputs(
    A,
    SA,
    Groups,
    Starts,
    Padded,
    AP,
    SAP,
    Layout,
    M: tl.constexpr,
    K: tl.constexpr,
    CAP: tl.constexpr,
    SAM: tl.constexpr,
    SAK: tl.constexpr,
    BM: tl.constexpr,
    BS: tl.constexpr,
):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    valid = row < M
    expert = tl.load(Groups + row, valid, 0)
    first = tl.load(Starts + expert)
    padded = tl.load(Padded + expert)
    pr = padded + row - first
    kk = tl.program_id(1) * 128 + tl.arange(0, 128)
    value = tl.load(
        A + row.to(tl.int64)[:, None] * K + kk[None, :], valid[:, None], 0.0
    )
    # Invalid padded rows never reach the compact output; no zero-fill is needed.
    tl.store(AP + pr.to(tl.int64)[:, None] * K + kk[None, :], value, valid[:, None])
    if tl.program_id(1) == 0:
        sk = tl.arange(0, BS)
        mask = valid[:, None] & (sk[None, :] < K // 128)
        scale = tl.load(
            SA + row.to(tl.int64)[:, None] * SAM + sk[None, :] * SAK, mask, 0
        )
        tl.store(SAP + pr[:, None] + sk[None, :] * CAP, scale, mask)
        tl.store(Layout + pr, expert, valid)


@triton.jit
def unpack_output(
    CP, C, Groups, Starts, Padded, M: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr
):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = x // N, x % N
    expert = tl.load(Groups + row, row < M, 0)
    start = tl.load(Starts + expert)
    padded = tl.load(Padded + expert)
    value = tl.load(CP + (padded + row - start).to(tl.int64) * N + col, row < M, 0)
    tl.store(C + x.to(tl.int64), value, row < M)


class DeepGemmBaseline:
    """Reuse allocated workspaces, but refresh all routing and tensor data per call.

    __call__ times the complete compact-to-compact operation. native_only is an
    explicitly separate diagnostic with already padded operands. Neither path
    dequantizes to BF16, rounds scales to powers of two, or reads counts on CPU.
    """

    def __init__(self, args, *, padded_capacity=None):
        import deep_gemm

        self.dg = deep_gemm
        self.args = args
        self.sb = (
            args[3]
            if args[3].is_contiguous()
            else torch.empty(args[3].shape, device=args[3].device, dtype=args[3].dtype)
        )
        a, sa, b, sb, out, groups, counts = args
        self.m, self.k = a.shape
        self.e, self.n, _ = b.shape
        self.align = deep_gemm.get_mk_alignment_for_contiguous_layout()
        assert self.align % 128 == 0
        self.cap = (
            min(self.m, triton.cdiv(self.m, self.align) + self.e - 1) * self.align
        )
        if padded_capacity is not None:
            self.cap = padded_capacity
        self.starts = torch.empty((self.e,), device=a.device, dtype=torch.int32)
        self.padded = torch.empty((self.e + 1,), device=a.device, dtype=torch.int32)
        self.ap = torch.empty((self.cap, self.k), device=a.device, dtype=a.dtype)
        self.sap = torch.empty_strided(
            (self.cap, self.k // 128), (1, self.cap), device=a.device, dtype=sa.dtype
        )
        self.cp = torch.empty((self.cap, self.n), device=a.device, dtype=out.dtype)
        self.layout = torch.empty((self.cap,), device=a.device, dtype=torch.int32)

    def prepare(self):
        a, sa, b, sb, out, groups, counts = self.args
        if not sb.is_contiguous():
            self.sb.copy_(sb)
        routing_offsets[(1,)](
            counts,
            self.starts,
            self.padded,
            self.e,
            self.align,
            triton.next_power_of_2(self.e),
        )
        clear_layout[(triton.cdiv(self.cap, 1024),)](self.layout, self.cap, 1024)
        pack_inputs[(triton.cdiv(self.m, 32), self.k // 128)](
            a,
            sa,
            groups,
            self.starts,
            self.padded,
            self.ap,
            self.sap,
            self.layout,
            self.m,
            self.k,
            self.cap,
            *sa.stride(),
            32,
            triton.next_power_of_2(self.k // 128),
            num_warps=4,
        )

    def native_only(self):
        a, sa, b, sb, out, groups, counts = self.args
        self.dg.m_grouped_fp8_gemm_nt_contiguous(
            (self.ap, self.sap),
            (b, self.sb),
            self.cp,
            self.layout,
            recipe=(1, 128, 128),
            disable_ue8m0_cast=True,
        )

    def __call__(self):
        if self.m == 0:
            return
        self.prepare()
        self.native_only()
        self.unpack()

    def unpack(self):
        unpack_output[(triton.cdiv(self.m * self.n, 1024),)](
            self.cp,
            self.args[4],
            self.args[5],
            self.starts,
            self.padded,
            self.m,
            self.n,
            1024,
        )


class PrepackedDeepGemmBaseline(DeepGemmBaseline):
    """Native-call benchmark over a frozen, prepacked input snapshot.

    Construction performs GPU routing, padding and scale layout adaptation.
    Calls time only native FP8 GEMM and leave output in its padded layout.
    Reconstruct this object when input values or routing change; use the compact
    baseline when measuring adaptation or validating compact output.
    """

    def __init__(self, args):
        import deep_gemm

        # Prepacking is outside the timer: size buffers to the actual padded
        # row count, without the compact adapter's worst-case routing reserve.
        align = deep_gemm.get_mk_alignment_for_contiguous_layout()
        lengths = args[6].cpu().tolist()
        capacity = sum(triton.cdiv(length, align) * align for length in lengths)
        super().__init__(args, padded_capacity=capacity)
        if self.m:
            self.prepare()

    def __call__(self):
        if self.m:
            self.native_only()
        return self.cp


class GroupedFp8Benchmark(base.Benchmark):
    DEFAULT_SHAPES = []
    DEFAULT_SHAPE_DESC = "M,N,K,E; prepacked native FP8, adaptation excluded"

    def set_shapes(self, shape_file_path=None):
        path = Path(shape_file_path or base.Config.shape_file)
        if path.suffix == ".json":
            self.rows = json.loads(path.read_text())["cases"]
            for row in self.rows:
                m, n, k, e = row["shape"]
                if (
                    m <= 0
                    or n <= 0
                    or n % 64
                    or k <= 0
                    or k % 128
                    or e <= 0
                    or row["count"] <= 0
                ):
                    raise ValueError("Invalid grouped FP8 workload shape or count")
            self.shapes = [tuple(row["shape"]) for row in self.rows]
            self.shape_desc = self.DEFAULT_SHAPE_DESC
        else:
            super().set_shapes(shape_file_path)
            self.rows = [dict(shape=list(shape), count=1) for shape in self.shapes]
        self.native = None

    def get_case_iter(self, dtype):
        for index, row in enumerate(self.rows):
            yield self._case_from_plan(
                dtype,
                index,
                consts.BenchmarkCasePlan(
                    shape=row["shape"],
                    params=dict(
                        source_count=row["count"],
                        input_dtype="float8_e4m3fn",
                        routing="regenerated near-uniform counts",
                        scale_layout="FP32 A[M,K/128], B[E,ceil(N/128),K/128]",
                        baseline="Prepacked DeepGEMM native FP8; packing/unpacking excluded",
                    ),
                ),
            )

    def build_inputs(self, case):
        m, n, k, e = case.shape
        args = make_grouped_gemm_fp8_case(
            [m // e + (i < m % e) for i in range(e)], n, k
        )[0]
        self.native = PrepackedDeepGemmBaseline(args)
        return args

    def get_input_iter(self, dtype):
        for case in self.get_case_iter(dtype):
            yield self.build_inputs(case)

    def native_reference(self, *args):
        if self.native is None or any(
            x is not y for x, y in zip(args, self.native.args)
        ):
            self.native = PrepackedDeepGemmBaseline(args)
        return self.native()

    def get_tflops(self, op, *args, **kwargs):
        return 2 * args[0].shape[0] * args[0].shape[1] * args[2].shape[1]
