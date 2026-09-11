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

"""Ascend W8A8 mm (same symbol as Hopper ``mm_w8a8_fp8``).

Public API matches NVIDIA PR #3821: BF16/FP16 inputs are quantized, then
``C = (A_q @ B_q) * a_scale[:, None] * b_scale[None, :]``.

Ascend 910 UB / DataCopy cannot load FP8, so weights and activations stay
INT8 plus per-row / per-column scale. Tiny matrices use one Vector kernel;
large matrices fuse INT8 Cube matmul and FP32 scaling in a mixed kernel.
Other shapes use an INT32 workspace and a tiled Vector output pass. All paths
apply both scales and cast to the requested output dtype. The older CommonIR
AIC-only implementation is retained for comparison.
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from typing import Optional

import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

from .ascendc.compile_fixpipe import (
    ensure_bitcode,
    install_cann90_custom_op_compat,
    makefile_compile,
    source_path,
)
from .ascendc.compile_fixpipe import symbol as _fixpipe_symbol

logger = logging.getLogger(__name__)

_INT8_MAX = 127.0
_B_CACHE: OrderedDict = OrderedDict()
_B_PACKED_CACHE: OrderedDict = OrderedDict()
_B_CACHE_MAX = 64
_C_WORKSPACE: dict[tuple, torch.Tensor] = {}
_MM_W8A8_OUTPUT_DTYPE = os.environ.get("FLAGGEMS_MM_W8A8_OUTPUT_DTYPE", "bf16").lower()
_FIXPIPE_M_MAJOR = os.environ.get("FLAGGEMS_MM_W8A8_M_MAJOR", "1") == "1"
_FIXPIPE_READY = False


def _init_fixpipe() -> None:
    global _FIXPIPE_READY
    if _FIXPIPE_READY:
        return
    bc = ensure_bitcode()
    try:
        install_cann90_custom_op_compat()

        @al.register_custom_op
        class fixpipe_vdeqf16:  # noqa: N801
            name = "fixpipe_vdeqf16"
            core = al.CORE.CUBE
            pipe = al.PIPE.PIPE_FIX
            mode = al.MODE.SIMD
            symbol = _fixpipe_symbol()
            bitcode = str(bc)
            source = str(source_path())
            compile = makefile_compile()

            def __init__(
                self,
                acc,
                deq,
                row_diag,
                c,
                pid_m,
                pid_n,
                tile_m,
                tile_n,
                acc_stride,
                ldc,
                load_deq,
                load_diag,
                out_bf16,
                out=None,
            ):
                # HIVM CustomOp verifier requires a tensor/memref `outs`
                # segment. C is a GM pointer written in-place, so it stays in
                # `ins`; `out` is a dummy UB tile only to form the op.
                assert out is not None, "fixpipe_vdeqf16 requires a dummy out tensor"
                self.arg_type["pid_m"] = tl.int32
                self.arg_type["pid_n"] = tl.int32
                self.arg_type["tile_m"] = tl.int32
                self.arg_type["tile_n"] = tl.int32
                self.arg_type["acc_stride"] = tl.int32
                self.arg_type["ldc"] = tl.int32
                self.arg_type["load_deq"] = tl.int32
                self.arg_type["load_diag"] = tl.int32
                self.arg_type["out_bf16"] = tl.int32
                del acc, deq, row_diag, c

    except AssertionError as exc:
        if "already used" not in str(exc):
            raise
    _FIXPIPE_READY = True
    logger.info("mm_w8a8_fp8 FixPipe Common IR ready (%s)", bc)


_init_fixpipe()


@libentry()
@triton.jit
def mm_w8a8_fp8_fixpipe_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    deq_ptr,
    row_diag_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    OUT_M: tl.constexpr,
    OUT_N: tl.constexpr,
    N_CORES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DISALLOW_ACC: tl.constexpr,
    M_MAJOR: tl.constexpr,
    OUT_BF16: tl.constexpr,
):
    # M/N/K/N_CORES are constexpr so the K trip count, grid and deq path
    # fold. Host pads to full tiles and packs B as contiguous KxN tiles.
    pid = ext.program_id(0)
    grid_m: tl.constexpr = tl.cdiv(OUT_M, BLOCK_M)
    grid_n: tl.constexpr = tl.cdiv(OUT_N, BLOCK_N)
    n_tiles: tl.constexpr = grid_m * grid_n
    cache_deq: tl.constexpr = OUT_N <= 8192

    with al.scope(core_mode="cube"):
        dummy = tl.full([16], 0, tl.float16)
        q = n_tiles // N_CORES
        r = n_tiles % N_CORES
        start = tl.where(pid < r, pid * (q + 1), r * (q + 1) + (pid - r) * q)
        count = q + tl.where(pid < r, 1, 0)
        if M_MAJOR:
            pid_m = start // grid_n
            pid_n = start % grid_n
        else:
            pid_n = start // grid_m
            pid_m = start % grid_m
        for i in tl.range(0, count):
            load_deq = 2 if (cache_deq and i == 0) else (0 if cache_deq else 1)
            load_diag = 1 if (not M_MAJOR or i == 0 or pid_n == 0) else 0
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
            off_m = (pid_m * BLOCK_M).to(tl.int32)
            off_n = (pid_n * BLOCK_N).to(tl.int32)
            tile_m = tl.minimum(BLOCK_M, OUT_M - off_m)
            tile_n = tl.minimum(BLOCK_N, OUT_N - off_n)
            a_block_ptr = tl.make_block_ptr(
                base=a_ptr,
                shape=(M, K),
                strides=(K, 1),
                offsets=(off_m, 0),
                block_shape=(BLOCK_M, BLOCK_K),
                order=(1, 0),
            )
            b_block_ptr = tl.make_block_ptr(
                base=b_ptr + pid_n * K * BLOCK_N,
                shape=(K, BLOCK_N),
                strides=(BLOCK_N, 1),
                offsets=(0, 0),
                block_shape=(BLOCK_K, BLOCK_N),
                order=(1, 0),
            )
            if DISALLOW_ACC:
                for k0 in tl.range(
                    0, K, BLOCK_K, num_stages=2, disallow_acc_multi_buffer=True
                ):
                    a = tl.load(a_block_ptr)
                    b = tl.load(b_block_ptr)
                    acc = tl.dot(a, b, acc, out_dtype=tl.int32)
                    a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
                    b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))
            else:
                # Two L0C banks: FixPipe drains one while the next tile MMA fills the other.
                for k0 in tl.range(
                    0, K, BLOCK_K, num_stages=2, disallow_acc_multi_buffer=False
                ):
                    a = tl.load(a_block_ptr)
                    b = tl.load(b_block_ptr)
                    acc = tl.dot(a, b, acc, out_dtype=tl.int32)
                    a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
                    b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))
            al.custom(
                "fixpipe_vdeqf16",
                acc,
                deq_ptr,
                row_diag_ptr,
                c_ptr,
                off_m,
                off_n,
                tile_m.to(tl.int32),
                tile_n.to(tl.int32),
                BLOCK_M,
                OUT_N,
                tl.cast(load_deq, tl.int32),
                tl.cast(load_diag, tl.int32),
                tl.cast(OUT_BF16, tl.int32),
                out=dummy,
            )
            if M_MAJOR:
                pid_n = pid_n + 1
                wrap = pid_n == grid_n
                pid_m = pid_m + wrap
                pid_n = tl.where(wrap, 0, pid_n)
            else:
                pid_m = pid_m + 1
                wrap = pid_m == grid_m
                pid_n = pid_n + wrap
                pid_m = tl.where(wrap, 0, pid_m)


@libentry()
@triton.jit
def _mm_w8a8_tiny_kernel(
    A,
    BT,
    SA,
    SB,
    OUT,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    OM: tl.constexpr,
    ON: tl.constexpr,
):
    # For K <= 64, every integer product and partial sum is exact in FP32.
    tiles: tl.constexpr = tl.cdiv(N, BN)
    tile = ext.program_id(0)
    row = tile // tiles
    col = tile % tiles * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    a = tl.load(A + row * K + kk, kk < K, other=0).to(tl.float32)
    b = tl.load(
        BT + col[:, None] * K + kk[None, :],
        (col[:, None] < N) & (kk[None, :] < K),
        other=0,
    ).to(tl.float32)
    acc = tl.sum(b * a[None, :], 1)
    sa = tl.load(SA + row)
    sb = tl.load(SB + col, col < N, other=0)
    tl.store(OUT + row * OM + col * ON, acc * sa * sb, col < N)


@libentry()
@triton.jit
def _mm_w8a8_mixed_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    SA,
    SB,
    OUT_M: tl.constexpr,
    OUT_N: tl.constexpr,
    SCM: tl.constexpr,
    SCN: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    N_CORES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    M_MAJOR: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """INT8 Cube matmul with FP32 row/column scaling in the same mixed kernel."""
    pid = ext.program_id(0)
    grid_m: tl.constexpr = tl.cdiv(M, BLOCK_M)
    grid_n: tl.constexpr = tl.cdiv(N, BLOCK_N)
    n_tiles: tl.constexpr = grid_m * grid_n
    q = n_tiles // N_CORES
    r = n_tiles % N_CORES
    start = tl.where(pid < r, pid * (q + 1), r * (q + 1) + (pid - r) * q)
    count = q + tl.where(pid < r, 1, 0)
    if GROUP_M == 0:
        if M_MAJOR:
            # Keep one A tile adjacent across N tiles. This is faster once M
            # is large and the packed B working set fits in L2.
            pid_m = start // grid_n
            pid_n = start % grid_n
        else:
            # Keep one packed B tile adjacent across M tiles.
            pid_n = start // grid_m
            pid_m = start % grid_m
    for i in tl.range(0, count):
        if GROUP_M > 0:
            # Bound the live A/B working set for wide matrices so both sides
            # are reused before the traversal advances to the next M group.
            tile_id = start + i
            group_width: tl.constexpr = GROUP_M * grid_n
            group_id = tile_id // group_width
            first_m = group_id * GROUP_M
            group_size_m = tl.minimum(grid_m - first_m, GROUP_M)
            tile_in_group = tile_id % group_width
            pid_m = first_m + tile_in_group % group_size_m
            pid_n = tile_in_group // group_size_m
        off_m = (pid_m * BLOCK_M).to(tl.int32)
        off_n = (pid_n * BLOCK_N).to(tl.int32)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        a_block_ptr = tl.make_block_ptr(
            base=a_ptr,
            shape=(M, K),
            strides=(K, 1),
            offsets=(off_m, 0),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(1, 0),
        )
        b_block_ptr = tl.make_block_ptr(
            base=b_ptr + pid_n * K * BLOCK_N,
            shape=(K, BLOCK_N),
            strides=(BLOCK_N, 1),
            offsets=(0, 0),
            block_shape=(BLOCK_K, BLOCK_N),
            order=(1, 0),
        )
        for _k0 in tl.range(0, K, BLOCK_K, num_stages=2):
            a = tl.load(a_block_ptr)
            b = tl.load(b_block_ptr)
            acc = tl.dot(a, b, acc, out_dtype=tl.int32)
            a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
            b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))
        rows = off_m + tl.arange(0, BLOCK_M)
        cols = off_n + tl.arange(0, BLOCK_N)
        sa = tl.load(SA + rows, rows < OUT_M, other=0)
        sb = tl.load(SB + cols, cols < OUT_N, other=0)
        val = acc.to(tl.float32) * sa[:, None] * sb[None, :]
        tl.store(
            c_ptr + rows[:, None] * SCM + cols[None, :] * SCN,
            val,
            (rows[:, None] < OUT_M) & (cols[None, :] < OUT_N),
        )
        if GROUP_M == 0:
            if M_MAJOR:
                pid_n = pid_n + 1
                wrap = pid_n == grid_n
                pid_m = pid_m + wrap
                pid_n = tl.where(wrap, 0, pid_n)
            else:
                pid_m = pid_m + 1
                wrap = pid_m == grid_m
                pid_n = pid_n + wrap
                pid_m = tl.where(wrap, 0, pid_m)


@libentry()
@triton.jit
def mm_w8a8_fp8_int32_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    N_CORES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    M_MAJOR: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """INT8 Cube with a plain INT32 FixPipe drain.

    Row and column scales are fused in the existing Vector output kernel.
    This avoids the fixed CommonIR custom-op cost without adding a launch.
    """
    pid = ext.program_id(0)
    grid_m: tl.constexpr = tl.cdiv(M, BLOCK_M)
    grid_n: tl.constexpr = tl.cdiv(N, BLOCK_N)
    n_tiles: tl.constexpr = grid_m * grid_n
    q = n_tiles // N_CORES
    r = n_tiles % N_CORES
    start = tl.where(pid < r, pid * (q + 1), r * (q + 1) + (pid - r) * q)
    count = q + tl.where(pid < r, 1, 0)
    if GROUP_M == 0:
        if M_MAJOR:
            # Keep one A tile adjacent across N tiles. This is faster once M
            # is large and the packed B working set fits in L2.
            pid_m = start // grid_n
            pid_n = start % grid_n
        else:
            # Keep one packed B tile adjacent across M tiles.
            pid_n = start // grid_m
            pid_m = start % grid_m
    for i in tl.range(0, count):
        if GROUP_M > 0:
            # Bound the live A/B working set for wide matrices so both sides
            # are reused before the traversal advances to the next M group.
            tile_id = start + i
            group_width: tl.constexpr = GROUP_M * grid_n
            group_id = tile_id // group_width
            first_m = group_id * GROUP_M
            group_size_m = tl.minimum(grid_m - first_m, GROUP_M)
            tile_in_group = tile_id % group_width
            pid_m = first_m + tile_in_group % group_size_m
            pid_n = tile_in_group // group_size_m
        off_m = (pid_m * BLOCK_M).to(tl.int32)
        off_n = (pid_n * BLOCK_N).to(tl.int32)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        a_block_ptr = tl.make_block_ptr(
            base=a_ptr,
            shape=(M, K),
            strides=(K, 1),
            offsets=(off_m, 0),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(1, 0),
        )
        b_block_ptr = tl.make_block_ptr(
            base=b_ptr + pid_n * K * BLOCK_N,
            shape=(K, BLOCK_N),
            strides=(BLOCK_N, 1),
            offsets=(0, 0),
            block_shape=(BLOCK_K, BLOCK_N),
            order=(1, 0),
        )
        for _k0 in tl.range(0, K, BLOCK_K, num_stages=2):
            a = tl.load(a_block_ptr)
            b = tl.load(b_block_ptr)
            acc = tl.dot(a, b, acc, out_dtype=tl.int32)
            a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
            b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))
        c_block_ptr = tl.make_block_ptr(
            base=c_ptr,
            shape=(M, N),
            strides=(N, 1),
            offsets=(off_m, off_n),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )
        tl.store(c_block_ptr, acc)
        if GROUP_M == 0:
            if M_MAJOR:
                pid_n = pid_n + 1
                wrap = pid_n == grid_n
                pid_m = pid_m + wrap
                pid_n = tl.where(wrap, 0, pid_n)
            else:
                pid_m = pid_m + 1
                wrap = pid_m == grid_m
                pid_n = pid_n + wrap
                pid_m = tl.where(wrap, 0, pid_m)


@libentry()
@triton.jit(do_not_specialize=["M", "N"])
def _row_scale_cast_kernel(
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    o_ptr,
    M,
    N,
    stride_cm,
    stride_om,
    BLOCK_N: tl.constexpr,
):
    row = ext.program_id(0)
    nprog = ext.num_programs(0)
    offs = tl.arange(0, BLOCK_N)
    for r in range(row, M, nprog):
        a_scale = tl.load(a_scale_ptr + r)
        for n0 in range(0, N, BLOCK_N):
            n_idx = n0 + offs
            mask = n_idx < N
            c = tl.load(c_ptr + r * stride_cm + n_idx, mask=mask, other=0).to(
                tl.float32
            )
            b_scale = tl.load(b_scale_ptr + n_idx, mask=mask, other=0).to(tl.float32)
            tl.store(o_ptr + r * stride_om + n_idx, c * a_scale * b_scale, mask=mask)


@libentry()
@triton.jit
def _scale_int32_static_kernel(
    C, SA, SB, OUT, N: tl.constexpr, NP: tl.constexpr, R: tl.constexpr, X: tl.constexpr
):
    tile = ext.program_id(0)
    cols: tl.constexpr = N // X
    rr = tile // cols * R + tl.arange(0, R)
    cc = tile % cols * X + tl.arange(0, X)
    v = tl.load(C + rr[:, None] * NP + cc[None, :]).to(tl.float32)
    a = tl.load(SA + rr)
    b = tl.load(SB + cc)
    tl.store(OUT + rr[:, None] * N + cc[None, :], v * a[:, None] * b[None, :])


@libentry()
@triton.jit
def _scale_int32_dense_kernel(
    C,
    SA,
    SB,
    OUT,
    M: tl.constexpr,
    N: tl.constexpr,
    NP: tl.constexpr,
    R: tl.constexpr,
    X: tl.constexpr,
):
    # Full, contiguous output tiles avoid generic masked gather/scatter code.
    pid = ext.program_id(0)
    columns: tl.constexpr = N // X
    tiles: tl.constexpr = (M // R) * columns
    for tile in range(pid, tiles, ext.num_programs(0)):
        rr = tile // columns * R + tl.arange(0, R)
        cc = tile % columns * X + tl.arange(0, X)
        value = tl.load(C + rr[:, None] * NP + cc[None, :]).to(tl.float32)
        row_scale = tl.load(SA + rr)
        col_scale = tl.load(SB + cc)
        tl.store(
            OUT + rr[:, None] * N + cc[None, :],
            value * row_scale[:, None] * col_scale[None, :],
        )


@libentry()
@triton.jit
def _scale_int32_tiles_kernel(
    C,
    A,
    B,
    OUT,
    M: tl.constexpr,
    N: tl.constexpr,
    NP: tl.constexpr,
    R: tl.constexpr,
    X: tl.constexpr,
    OM: tl.constexpr,
    ON: tl.constexpr,
):
    pid = ext.program_id(0)
    cols: tl.constexpr = tl.cdiv(N, X)
    tiles: tl.constexpr = tl.cdiv(M, R) * cols
    for tile in range(pid, tiles, ext.num_programs(0)):
        r = tile // cols * R + tl.arange(0, R)
        col = tile % cols * X + tl.arange(0, X)
        c = tl.load(
            C + r[:, None] * NP + col[None, :],
            (r[:, None] < M) & (col[None, :] < N),
            other=0,
        ).to(tl.float32)
        a = tl.load(A + r, r < M, other=0)
        b = tl.load(B + col, col < N, other=0)
        tl.store(
            OUT + r[:, None] * OM + col[None, :] * ON,
            c * a[:, None] * b[None, :],
            (r[:, None] < M) & (col[None, :] < N),
        )


def _vector_grid(n: int) -> int:
    return max(1, min(n, 40 * 8))


@libentry()
@triton.jit
def _quantize_rows_kernel(
    X,
    Q,
    S,
    M: tl.constexpr,
    K: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    R: tl.constexpr,
    BK: tl.constexpr,
):
    pid = ext.program_id(0)
    for r0 in range(pid * R, M, ext.num_programs(0) * R):
        r = r0 + tl.arange(0, R)
        s = tl.load(S + r, r < M, other=1.0)
        for c0 in range(0, K, BK):
            kk = c0 + tl.arange(0, BK)
            x = tl.load(
                X + r[:, None] * S0 + kk[None, :] * S1,
                (r[:, None] < M) & (kk[None, :] < K),
                other=0,
            ).to(tl.float32)
            z = x / s[:, None]
            f = tl.floor(z)
            fi = f.to(tl.int32)
            d = z - f
            yi = fi + ((d > 0.5) | ((d == 0.5) & ((fi & 1) != 0))).to(tl.int32)
            q = tl.minimum(127, tl.maximum(-128, yi)).to(tl.int8)
            tl.store(
                Q + r[:, None] * K + kk[None, :],
                q,
                (r[:, None] < M) & (kk[None, :] < K),
            )


@libentry()
@triton.jit
def _row_amax_kernel(
    X,
    S,
    M: tl.constexpr,
    K: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    R: tl.constexpr,
    BK: tl.constexpr,
):
    pid = ext.program_id(0)
    for r0 in range(pid * R, M, ext.num_programs(0) * R):
        r = r0 + tl.arange(0, R)
        kk = tl.arange(0, BK)
        x = tl.load(
            X + r[:, None] * S0 + kk[None, :] * S1,
            (r[:, None] < M) & (kk[None, :] < K),
            other=0,
        ).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(x), 1), 1.0e-10)
        tl.store(S + r, amax, r < M)


def _quantize_int8_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = x.shape
    if k == 0 or k > 4096 or m == 0:
        xf = x.float()
        scale = xf.abs().amax(dim=1).clamp_min(1e-10) / _INT8_MAX
        q = (xf / scale[:, None]).round().clamp(-128, 127).to(torch.int8)
        return q.contiguous(), scale.contiguous()
    q = torch.empty((m, k), dtype=torch.int8, device=x.device)
    amax = torch.empty((m,), dtype=torch.float32, device=x.device)
    rr = 4 if k <= 2048 else 1
    _row_amax_kernel[(min(40, triton.cdiv(m, rr)),)](
        x, amax, m, k, *x.stride(), rr, triton.next_power_of_2(k)
    )
    # Native division preserves scale rounding at quantization half-integers.
    # Fusing this division changed some INT8 values by one on this runtime.
    s = amax / _INT8_MAX
    r = 4
    _quantize_rows_kernel[(min(40, triton.cdiv(m, r)),)](
        x, q, s, m, k, *x.stride(), r, min(256, triton.next_power_of_2(k))
    )
    return q, s


def _quantize_int8_cols(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    xf = x.float()
    scale = xf.abs().amax(dim=0).clamp_min(1e-10) / _INT8_MAX
    q = (xf / scale[None, :]).round().clamp(-128, 127).to(torch.int8)
    return q.contiguous(), scale.contiguous()


def _pack_deq_u64(scale: torch.Tensor) -> torch.Tensor:
    """Pack fp32 IEEE bits into the low 32 bits of int64 (AscendC uint64 deq)."""
    bits = scale.contiguous().to(torch.float32).view(torch.int32)
    return (bits.to(torch.int64) & 0xFFFFFFFF).contiguous()


def _pack_col_deq(b_s: torch.Tensor) -> torch.Tensor:
    """Pack an N-channel VDEQF16 scale vector."""
    n_pad = _align_up(b_s.numel(), 32)
    packed = _pack_deq_u64(b_s)
    if n_pad != b_s.numel():
        out = packed.new_zeros((n_pad,))
        out[: b_s.numel()] = packed
        return out.contiguous()
    return packed.contiguous()


def _prepare_aic_epilogue(
    a_s: torch.Tensor, b_s: torch.Tensor, block_m: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare scale inputs for the single AIC kernel.

    ``VDEQF16`` applies one N-channel vector to every row.  Fold the largest
    row scale into that vector and provide the remaining per-row ratios as
    packed 16x16 diagonal FP16 matrices.  The CommonIR epilogue consumes both
    inputs inside AIC and writes the final output directly.
    """
    assert block_m % 16 == 0 and block_m <= 128
    assert a_s.numel() % block_m == 0
    row_ref = a_s.abs().amax().clamp_min(1e-10)
    deq = _pack_col_deq(b_s * row_ref)
    row_ratio = (a_s / row_ref).to(torch.float16)
    groups = a_s.numel() // block_m
    m1 = block_m // 16
    row_diag_nd = torch.diag_embed(row_ratio.reshape(groups, block_m))
    # A1 fractal layout consumed by L1->L0A: [group, K1, M1, M0, K0].
    row_diag = (
        row_diag_nd.reshape(groups, m1, 16, m1, 16).permute(0, 3, 1, 2, 4).contiguous()
    )
    return deq, row_diag


def _b_cache_key(b: torch.Tensor) -> tuple:
    return (
        b.device,
        b.data_ptr(),
        tuple(b.shape),
        tuple(b.stride()),
        b.dtype,
        b._version,
    )


def _get_cached_b(b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Inference tensors have no version counter. Recompute rather than return
    # stale weights when they are mutated in an inference_mode region.
    try:
        key = _b_cache_key(b)
    except RuntimeError:
        q, scale = _quantize_int8_cols(b)
        return q, scale, _pack_col_deq(scale)
    cached = _B_CACHE.get(key)
    if cached is not None:
        _B_CACHE.move_to_end(key)
        return cached[1:]
    q, scale = _quantize_int8_cols(b)
    packed = _pack_col_deq(scale)
    item = (q, scale, packed)
    # Keep the original storage alive so allocator address reuse cannot alias.
    _B_CACHE[key] = (b, *item)
    if len(_B_CACHE) > _B_CACHE_MAX:
        _B_CACHE.popitem(last=False)
    return item


def _get_cached_b_int8(b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q, scale, _packed = _get_cached_b(b)
    return q, scale


def _c_workspace(device, m: int, n: int, dtype=torch.float16) -> torch.Tensor:
    key = (str(device), m, n, dtype)
    buf = _C_WORKSPACE.get(key)
    if buf is None or buf.shape != (m, n) or buf.dtype != dtype:
        buf = torch.empty((m, n), device=device, dtype=dtype)
        _C_WORKSPACE[key] = buf
    return buf


def clear_mm_w8a8_fp8_caches() -> None:
    _B_CACHE.clear()
    _B_PACKED_CACHE.clear()
    _C_WORKSPACE.clear()


def get_mm_w8a8_fp8_cache_stats() -> dict:
    return {
        "b_int8": len(_B_CACHE),
        "b_packed": len(_B_PACKED_CACHE),
        "cache_max_entries": _B_CACHE_MAX,
        "fixpipe": _FIXPIPE_READY,
    }


def _cube_core_count() -> int:
    """910B has 20 AIC. Mix SIMD ``get_block_idx`` only covers one wave."""
    env = os.environ.get("FLAGGEMS_CUBE_CORES")
    if env:
        return max(1, int(env))
    return 20


def _l0c_bytes(block_m: int, block_n: int) -> int:
    return block_m * block_n * 4


def _can_acc_pingpong(block_m: int, block_n: int) -> bool:
    """910B L0C is 128KB. Two int32 banks must fit for MMA/FixPipe overlap."""
    return _l0c_bytes(block_m, block_n) * 2 <= 128 * 1024


def _pick_fixpipe_tiles(M: int, N: int, K: int) -> tuple[int, int, int]:
    env = os.environ.get("FLAGGEMS_FIXPIPE_TILES")
    if env:
        parts = tuple(int(x) for x in env.split(","))
        if len(parts) == 3:
            return parts
    if N <= 64:
        # Avoid 4x-256x N padding. For the PR shapes, K=512 reduces the fixed
        # loop cost, while smaller M tiles expose enough Cube work.
        if K == 2048:
            block_m = 64 if M <= 64 else (256 if M >= 8192 else 128)
            return block_m, 64, 512
        return (256 if M >= 2048 else 128), 64, 256
    if N >= 65536 and M <= 16:
        # PR #5972 contains M=1..8 with N=248320. A 128-row tile performs up
        # to 128x more work than required; 16 is the Cube alignment minimum.
        return 16, 256, 256
    if M >= 8192 and N in (9216, 12288) and K == 2048:
        return 128, 256, 512
    if N == 1024 and K == 2048:
        if M <= 16:
            return 16, 64, 256
        if M <= 32:
            return 32, 64, 256
        if M <= 64:
            return 64, 64, 512
        if M <= 128:
            return 128, 64, 512
        if M >= 512:
            return 256, 128, 512
    if N == 256 and K == 2048:
        return (64 if M <= 256 else 128), 256, 512
    if N == 2048 and K == 512:
        if M <= 128:
            return 128, 128, 512
        if M >= 1024:
            return 256, 128, 512
    if N == 2048 and K == 4096:
        if M <= 16:
            return 16, 128, 256
        if M <= 32:
            return 32, 128, 256
        if M <= 64:
            return 64, 128, 512
        if M <= 128:
            return 128, 128, 512
        if M <= 256:
            return 256, 128, 512
        if M <= 416:
            return 128, 128, 512
        if 1024 <= M < 8192:
            return 256, 128, 512
    # 128x256 int32 fills the 128KB L0C; smaller MN for ping-pong is slower.
    return 128, 256, 256


def _resolve_out_dtype(
    a: torch.Tensor, out_dtype: Optional[torch.dtype]
) -> torch.dtype:
    if out_dtype is not None:
        return out_dtype
    if _MM_W8A8_OUTPUT_DTYPE == "fp16":
        return torch.float16
    if a.dtype in (torch.float16, torch.bfloat16):
        return a.dtype
    return torch.bfloat16


def _align_up(x: int, align: int) -> int:
    return (x + align - 1) // align * align


def _pad_fixpipe_inputs(a_q, b_q, a_s, b_s, M, N, K, block_m, block_n, block_k):
    """Pad A/B and pack B into contiguous (N-tile, K-tile, BK, BN) storage."""
    m_pad = _align_up(M, max(16, block_m))
    n_pad = _align_up(N, max(32, block_n))
    k_pad = _align_up(K, max(32, block_k))
    if (m_pad, k_pad) == (M, K):
        a_pad, a_s_pad = a_q, a_s
    else:
        a_pad = a_q.new_zeros((m_pad, k_pad))
        a_pad[:M, :K] = a_q
        a_s_pad = a_s.new_zeros((m_pad,))
        a_s_pad[:M] = a_s

    if n_pad == N:
        b_s_pad = b_s
    else:
        b_s_pad = b_s.new_zeros((n_pad,))
        b_s_pad[:N] = b_s

    packed_key = (
        b_q.data_ptr(),
        tuple(b_q.shape),
        tuple(b_q.stride()),
        b_q.dtype,
        k_pad,
        n_pad,
        block_k,
        block_n,
    )
    packed = _B_PACKED_CACHE.get(packed_key)
    if packed is None:
        if (n_pad, k_pad) == (N, K):
            b_pad = b_q
        else:
            b_pad = b_q.new_zeros((k_pad, n_pad))
            b_pad[:K, :N] = b_q
        b_tiles = (
            b_pad.reshape(k_pad // block_k, block_k, n_pad // block_n, block_n)
            .permute(2, 0, 1, 3)
            .contiguous()
        )
        # Keep the source tensor alive so a recycled data_ptr cannot hit this entry.
        _B_PACKED_CACHE[packed_key] = (b_q, b_tiles)
        if len(_B_PACKED_CACHE) > _B_CACHE_MAX:
            _B_PACKED_CACHE.popitem(last=False)
    else:
        _B_PACKED_CACHE.move_to_end(packed_key)
        b_tiles = packed[1]
    return a_pad, b_tiles, a_s_pad, b_s_pad, m_pad, n_pad, k_pad


def _launch_fixpipe(a_q, b_q, a_s, b_s, out, M, N, K, deq=None):
    orig_m, orig_n = M, N
    block_m, block_n, block_k = _pick_fixpipe_tiles(M, N, K)
    block_m = min(block_m, 128)
    a_q, b_q, a_s, b_s, M, N, K = _pad_fixpipe_inputs(
        a_q, b_q, a_s, b_s, M, N, K, block_m, block_n, block_k
    )
    del deq
    deq, row_diag = _prepare_aic_epilogue(a_s, b_s, block_m)
    n_tiles = triton.cdiv(M, block_m) * triton.cdiv(N, block_n)
    wave = _cube_core_count()
    grid = min(n_tiles, wave)
    mm_w8a8_fp8_fixpipe_kernel[grid,](
        a_q,
        b_q,
        out,
        deq,
        row_diag,
        M,
        N,
        K,
        orig_m,
        orig_n,
        grid,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        DISALLOW_ACC=not _can_acc_pingpong(block_m, block_n),
        M_MAJOR=_FIXPIPE_M_MAJOR,
        OUT_BF16=1 if out.dtype == torch.bfloat16 else 0,
        mix_mode="aic",
        num_warps=1,
        num_stages=2,
        optimize_dynamic_offset=True,
        unit_flag=False,
        limit_auto_multi_buffer_of_local_buffer="no-l0c",
    )
    assert orig_n % 16 == 0, "AIC VDEQF16 path requires N to be a multiple of 16"
    return out


def _pick_int32_tiles(M: int, N: int, K: int) -> tuple[int, int, int]:
    if os.environ.get("FLAGGEMS_FIXPIPE_TILES"):
        return _pick_fixpipe_tiles(M, N, K)
    if N == 1024 and K == 2048 and 256 < M <= 448:
        return _align_up(triton.cdiv(M, 2), 16), 128, 512
    # Keep the whole short M dimension without padding it to 256 rows.
    if N == 2048 and K == 4096 and 128 < M <= 256:
        return _align_up(M, 16), 128, 512
    # Short K fits one 512-wide panel. Compact M tiles reduce padding for
    # small matrices; two M tiles balance the longest worker for larger M.
    if N == 2048 and K == 512 and 128 < M <= 224:
        return _align_up(M, 16), 128, 512
    if N == 2048 and K == 512 and 256 < M <= 512:
        return _align_up(triton.cdiv(M, 2), 16), 128, 512
    # Match short-wide tiles to the 20 Cube workers without excessive M padding.
    if N == 12288 and K == 2048 and 0 < M <= 32:
        return _align_up(M, 16), 128, 512
    if N == 9216 and K == 2048 and 32 < M <= 64:
        return _align_up(M, 16), 512, 256
    # Ascend block pointers and INT8 dot support M multiples of 16, not
    # only powers of two. Keep the L0C tile within 128 KiB.
    if 8192 <= N < 65536 and K == 2048 and 0 < M <= 512:
        m_tiles = triton.cdiv(M, 256)
        block_m = _align_up(triton.cdiv(M, m_tiles), 16)
        if block_m & (block_m - 1):
            return block_m, (256 if block_m <= 128 else 128), 512
    if N == 1024 and K == 2048 and 128 < M <= 256:
        return _align_up(triton.cdiv(M, 2), 16), 128, 512
    # Avoid padding short matrices to 128 rows, and reduce K-loop overhead.
    if N == 2048 and K == 512 and M <= 64:
        return max(16, triton.next_power_of_2(M)), 128, 512
    if N == 1024 and K == 2048 and M <= 32:
        return max(16, triton.next_power_of_2(M)), 64, 512
    if 8192 <= N < 65536 and K == 2048 and M <= 64:
        return max(16, triton.next_power_of_2(M)), 256, 512
    # Avoid a second tile iteration on the busiest cores for this narrow N.
    if N == 256 and K == 2048 and 320 < M <= 512:
        return 128, 64, 512
    # Narrow outputs need enough independent Cube tiles. BN=64 also avoids
    # the expensive INT8 packing path seen with 32-column panels on 910B.
    if 64 <= N <= 256 and K >= 1024 and (M <= 512 or N == 64):
        block_m = (
            16
            if M <= 64 or (N == 64 and M <= 128)
            else (32 if M <= 256 else (64 if M <= 512 else 128))
        )
        return block_m, 64, 512
    if N == 1024 and K == 2048 and 256 < M < 512:
        return 128, 256, 512
    if N >= 8192 and K == 2048 and 128 < M <= 512:
        # Do not add a 256-row padding penalty for M in (256, 384].
        if _align_up(M, 256) == _align_up(M, 128):
            return 256, 128, 512
    if 32 <= M <= 64 and N <= 256 and K <= 128:
        return 64, 64, max(32, triton.next_power_of_2(K))
    if 64 <= M <= 192 and N <= 512 and K <= 512:
        return 64, 128, min(256, max(32, triton.next_power_of_2(K)))
    if 192 < M <= 256 and N <= 1024 and K <= 1024:
        return 128, 128, 256
    if 256 < M <= 512 and N <= 1024 and K <= 1024:
        return 128, 256, min(512, max(32, triton.next_power_of_2(K)))
    block_m, block_n, block_k = _pick_fixpipe_tiles(M, N, K)
    if M <= 128 and K <= 256:
        block_m = min(block_m, max(16, triton.next_power_of_2(M)))
        block_n = min(block_n, max(32, triton.next_power_of_2(N)))
        block_k = min(block_k, max(32, triton.next_power_of_2(K)))
    return block_m, block_n, block_k


def _pick_aic_tiles(M: int, N: int, K: int):
    if M <= 0 or N <= 0:
        return None
    if N == 2048 and K == 512 and 1024 <= M <= 16384 and M % 16 == 0:
        return 128, 256, 128
    if K == 2048 and N == 64:
        if 32 < M <= 128:
            return 16, 64, 1024
        if 128 < M <= 512:
            return 32, 64, 1024
        if M == 2048:
            return 64, 64, 1024
    if K == 2048 and N == 256:
        if M <= 64:
            return 16, 64, 1024
        if 128 < M <= 320:
            return 32, 128, 512
        if 320 < M <= 512:
            return 64, 128, 512
    if K == 512 and N == 2048 and M <= 512:
        if M <= 32:
            return 16, 128, 512
        if M <= 256:
            return _align_up(triton.cdiv(M, 2), 16), 256, 256
        # Five M groups with eight N groups balance two tiles per Cube.
        return _align_up(triton.cdiv(M, 5), 16), 256, 256
    if K == 2048 and N == 1024:
        if M <= 16:
            return 16, 64, 1024
        if M <= 64:
            return 32, 128, 512
    return None


def _prepare_mm_w8a8_kernel(a_q, b_q, a_s, b_s, out, M, N, K):
    """Prepare a callable that executes only matmul and output scaling.

    Quantization, padding, weight layout conversion and explicit allocations
    finish before the returned callable is invoked. The public API and kernel
    benchmark share this dispatch. Captured tensors remain alive in the closure.
    """
    if M <= 8 and N in (16, 32, 64) and K in (16, 32, 64):
        key = ("tiny_transpose", b_q.data_ptr(), tuple(b_q.shape), b_q.dtype)
        cached = _B_PACKED_CACHE.get(key)
        if cached is None:
            b_t = b_q.t().contiguous()
            _B_PACKED_CACHE[key] = (b_q, b_t)
            if len(_B_PACKED_CACHE) > _B_CACHE_MAX:
                _B_PACKED_CACHE.popitem(last=False)
        else:
            _B_PACKED_CACHE.move_to_end(key)
            b_t = cached[1]
        bn, bk = triton.next_power_of_2(N), triton.next_power_of_2(K)
        grid = M * triton.cdiv(N, bn)

        def call():
            _mm_w8a8_tiny_kernel[(grid,)](
                a_q,
                b_t,
                a_s,
                b_s,
                out,
                M,
                N,
                K,
                bn,
                bk,
                *out.stride(),
                multibuffer=False,
            )

        return call, {
            "path": "tiny_vector",
            "tiles": [1, bn, bk],
            "kernel_count": 1,
            "workspace_bytes": 0,
        }

    nz_tiles = None
    if K == 2048 and M % 8 == 0:
        if N in (9216, 12288) and 8 <= M <= 512:
            nz_tiles = (_align_up(triton.cdiv(M, triton.cdiv(M, 128)), 16), 128, 256)
        elif N == 1024 and 64 < M <= 512:
            # Eight N tiles: use two M groups for one wave, five for two.
            groups = 2 if M <= 256 else 5
            nz_tiles = (_align_up(triton.cdiv(M, groups), 16), 128, 256)
        elif N == 256 and 1024 <= M < 8192:
            # Two N tiles per M group; balance each wave over 20 Cube cores.
            groups = triton.cdiv(triton.cdiv(M, 128), 10) * 10
            nz_tiles = (_align_up(triton.cdiv(M, groups), 16), 128, 256)
    if (
        nz_tiles is not None
        and out.dtype in (torch.bfloat16, torch.float16)
        and out.is_contiguous()
        and not os.environ.get("FLAGGEMS_FIXPIPE_TILES")
        and not os.environ.get("FLAGGEMS_MM_W8A8_EPILOGUE")
    ):
        from .ascendc.mm_nz import prepare as prepare_nz

        return prepare_nz(a_q, b_q, a_s, b_s, out, M, N, K, nz_tiles)

    aic_tiles = _pick_aic_tiles(M, N, K)
    if (
        aic_tiles is not None
        and out.dtype == torch.bfloat16
        and out.is_contiguous()
        and not os.environ.get("FLAGGEMS_FIXPIPE_TILES")
        and not os.environ.get("FLAGGEMS_MM_W8A8_EPILOGUE")
    ):
        from .ascendc.mm_aic import prepare as prepare_aic

        input_nz = N == 2048 and K == 512 and M > 128
        if input_nz:
            aic_tiles = (aic_tiles[0], aic_tiles[1], 128)
        batch_rows = input_nz and M >= 1024
        return prepare_aic(
            a_q,
            b_q,
            a_s,
            b_s,
            out,
            M,
            N,
            K,
            aic_tiles,
            input_nz=input_nz,
            batch_rows=batch_rows,
            prefetch=batch_rows,
        )

    mixed = (
        M >= 1024
        and N >= 1024
        and K >= 1024
        and M * N * K >= 2048**3
        and out.is_contiguous()
    )
    # Large short-K and narrow-N outputs are dominated by the full INT32
    # workspace round trip; mixed execution keeps only per-core tile scratch.
    mixed_extra = out.is_contiguous() and (
        (M >= 1024 and N >= 1024 and K == 512)
        or (M >= 8192 and N in (64, 256) and K >= 1024)
    )
    mixed = mixed or mixed_extra
    if mixed and not os.environ.get("FLAGGEMS_FIXPIPE_TILES"):
        if N == 64:
            block_m, block_n, block_k = 128, 64, 512
        elif K == 512:
            block_m, block_n, block_k = 128, 256, 512
        else:
            block_m, block_n, block_k = 256, 128, (256 if N >= 8192 else 512)
    else:
        block_m, block_n, block_k = _pick_int32_tiles(M, N, K)
    a_q, b_q, a_s, b_s, mp, np, kp = _pad_fixpipe_inputs(
        a_q, b_q, a_s, b_s, M, N, K, block_m, block_n, block_k
    )
    grid = min(triton.cdiv(mp, block_m) * triton.cdiv(np, block_n), _cube_core_count())
    opts = dict(
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        M_MAJOR=_FIXPIPE_M_MAJOR,
        GROUP_M=0,
        num_warps=1,
        num_stages=2,
        optimize_dynamic_offset=True,
        unit_flag=False,
        limit_auto_multi_buffer_of_local_buffer="no-l0c",
    )
    if mixed:

        def call():
            _mm_w8a8_mixed_kernel[(grid,)](
                a_q,
                b_q,
                out,
                a_s,
                b_s,
                M,
                N,
                *out.stride(),
                mp,
                np,
                kp,
                grid,
                **opts,
            )

        # The compiler uses one INT32 tile per block, not a full MxN buffer.
        return call, {
            "path": "mixed",
            "tiles": [block_m, block_n, block_k],
            "kernel_count": 1,
            "workspace_bytes": block_m * block_n * 4 * grid,
        }

    # Per-call ownership is required for concurrent streams and graph captures.
    acc = torch.empty((mp, np), dtype=torch.int32, device=out.device)
    if M >= 64 and N <= 1024:
        rows = 8 if M <= 64 else 16
        cols = min(256, triton.next_power_of_2(N))
    else:
        rows, cols = 4, min(1024, triton.next_power_of_2(N))
    if not out.is_contiguous():
        # Strided stores need additional index/scatter buffers in UB.
        rows, cols = 4, min(256, triton.next_power_of_2(N))
    static_output = (
        out.is_contiguous()
        and 16 <= M <= 128
        and M % 16 == 0
        and N in (32, 64, 128, 256)
    )
    if static_output:
        rows, cols = 16, N
    dense_output = (
        not static_output
        and out.is_contiguous()
        and (M >= 64 or (N >= 8192 and M >= 8))
        and M % 8 == 0
        and N % 64 == 0
    )
    if dense_output:
        cols = min(1024, N & -N)
        rows = 8
        if N in (9216, 12288) and M <= 512:
            if N == 12288:
                rows, cols = 8, 2048
            elif M % 16 == 0:
                rows, cols = 16, 1024
        if N == 1024 and M >= 1024 and M % 16 == 0:
            rows, cols = 16, 256
        if N <= 256 and M > 128:
            rows = 32 if M >= 1024 and M % 32 == 0 else (16 if M % 16 == 0 else 8)
    grid_v = min(40, triton.cdiv(M, rows) * triton.cdiv(N, cols))
    custom_epilogue = None
    if (
        os.environ.get("FLAGGEMS_MM_W8A8_EPILOGUE") == "ascendc"
        and out.is_contiguous()
        and M > 0
        and N > 0
        and M % 8 == 0
        and N % 64 == 0
    ):
        from .ascendc.vector_epilogue import prepare as prepare_vector_epilogue

        custom_rows = 16 if M % 16 == 0 else 8
        custom_cols = min(512, N & -N)
        custom_epilogue, _, _ = prepare_vector_epilogue(
            acc, a_s, b_s, out, M, N, custom_rows, custom_cols
        )

    def call():
        mm_w8a8_fp8_int32_kernel[(grid,)](
            a_q,
            b_q,
            acc,
            mp,
            np,
            kp,
            grid,
            **opts,
        )
        if custom_epilogue is not None:
            custom_epilogue()
        elif static_output:
            _scale_int32_static_kernel[(M // rows,)](
                acc,
                a_s,
                b_s,
                out,
                N,
                np,
                rows,
                cols,
            )
        elif dense_output:
            _scale_int32_dense_kernel[(grid_v,)](
                acc, a_s, b_s, out, M, N, np, rows, cols
            )
        else:
            _scale_int32_tiles_kernel[(grid_v,)](
                acc,
                a_s,
                b_s,
                out,
                M,
                N,
                np,
                rows,
                cols,
                *out.stride(),
            )

    return call, {
        "path": "int32_vector",
        "tiles": [block_m, block_n, block_k],
        "scale_tile": [rows, cols],
        "static_output": static_output,
        "dense_output": dense_output,
        "epilogue": "ascendc_brcb" if custom_epilogue is not None else "triton",
        "kernel_count": 2,
        "workspace_bytes": mp * np * 4,
    }


def _launch(a_q, b_q, a_s, b_s, out, M, N, K, deq=None):
    call, _ = _prepare_mm_w8a8_kernel(a_q, b_q, a_s, b_s, out, M, N, K)
    call()
    return out


def mm_w8a8_fp8(a, b, *, out_dtype: Optional[torch.dtype] = None):
    logger.debug("GEMS_ASCEND MM_W8A8_FP8")
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    a_q, a_s = _quantize_int8_rows(a)
    b_q, b_s, deq = _get_cached_b(b)
    out = torch.empty((M, N), device=a.device, dtype=_resolve_out_dtype(a, out_dtype))
    with torch_device_fn.device(a.device):
        return _launch(a_q, b_q, a_s, b_s, out, M, N, K, deq=deq)


def mm_w8a8_fp8_out(a, b, *, out):
    logger.debug("GEMS_ASCEND MM_W8A8_FP8_OUT")
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    assert out.shape == (M, N), "incompatible output shape"
    a_q, a_s = _quantize_int8_rows(a)
    b_q, b_s, deq = _get_cached_b(b)
    with torch_device_fn.device(a.device):
        return _launch(a_q, b_q, a_s, b_s, out, M, N, K, deq=deq)
