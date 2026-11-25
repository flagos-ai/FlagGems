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

"""DeepGEMM FP8 baseline, including GPU adaptation of compact expert rows."""

import torch
import triton
import triton.language as tl


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
