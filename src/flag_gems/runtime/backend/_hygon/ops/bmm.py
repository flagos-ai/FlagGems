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


"""Hygon BMM entry points, workload dispatch, and whole-pipeline FlagTune."""

import hashlib
import logging
import threading
from pathlib import Path

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.libentry import LibTuner

from . import bmm_regular as regular
from . import bmm_ring as ring

logger = logging.getLogger(__name__)


def select_kernel(dtype, batch, m, n, k):
    """Return (family, tile config) from dimensions, dtype and parallelism.

    Assembly kernels require complete reduction tiles and 16-byte row loads.
    M/N tile tails are handled by resource bounds and masked output stores.
    The generic family supports arbitrary strides and reduction tails.
    """
    if min(batch, m, n, k) <= 0:
        return "generic", None
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return "generic", None
    es = 4 if dtype == torch.float32 else 2
    # Offsets with their top bit set are reserved for masked resource accesses.
    # Keep tensor and intermediate batch offsets below signed 32-bit range.
    if max(batch * m * k, batch * k * n, batch * m * n) * es >= 2**31:
        return "generic", None
    tiles64 = batch * ((m + 63) // 64) * ((n + 63) // 64)
    tiles128 = batch * ((m + 127) // 128) * ((n + 127) // 128)
    tiles256 = batch * ((m + 255) // 256) * ((n + 255) // 256)
    # Parallelize long reductions when output tiling cannot fill the device.
    # This family uses masked tl.load and also supports arbitrary strides.
    if min(m, n) >= 32 and k >= 1024 and tiles64 < 128:
        desired = (128 + tiles64 - 1) // tiles64
        splits = min(16, 1 << (desired - 1).bit_length())
        return "splitk", (splits, 64, 64, 32)

    small = max(m, n) <= 512 and k <= 512
    if dtype == torch.float32:
        if ((m + 127) // 128) * 128 * n * es >= 2**31:
            return "generic", None
        if min(m, n) < 32 or k % 16 or n % 4:
            return "generic", None
        if small and k % 128 == 0:
            return "regular", (2, 64, 64, 128, 2, 2)
        tile = 64 if tiles128 < 64 else 128
        return "regular", (2, tile, tile, 16, 2, 2)

    padded_area = ((m + 255) // 256) * ((n + 255) // 256) * 256**2
    # A large ring tile amortizes DMA/synchronization when reduction/output
    # work is substantial. Reject tiles wasting more than half their lanes.
    if (
        not small
        and min(m, n) >= 256
        and k >= 256
        and k % 64 == 0
        and n % 8 == 0
        and tiles256 >= 24
        and 2 * m * n >= padded_area
        and ((m + 255) // 256) * 256 * n * es < 2**31
    ):
        return "ring", None
    if small and min(m, n) >= 32 and k % 128 == 0 and n % 8 == 0:
        if dtype == torch.float16:
            return "regular", (0, 64, 64, 128, 2, 2)
        return "regular", (1, 32, 64, 128, 2, 2)
    if min(m, n) >= 32:
        dt = 0 if dtype == torch.float16 else 1
        tile = 64 if min(m, n) < 64 or tiles128 < 64 else 128
        if ((m + tile - 1) // tile) * tile * n * es < 2**31:
            return "regular", (dt, tile, tile, 32, 2, 2)
    return "generic", None


def supports_isa(a):
    props = torch.cuda.get_device_properties(a.device)
    return getattr(props, "gcnArchName", "").split(":", 1)[0] == "gfx936"


def supports_vector_layout(a, b, out):
    """Check physical layout requirements independently of shape selection."""
    if not supports_isa(a):
        return False
    if not (a.is_contiguous() and b.is_contiguous() and out.is_contiguous()):
        return False
    # Contiguous views may start at an unaligned storage offset.
    return all(t.data_ptr() % 16 == 0 for t in (a, b, out))


def supports_native_layout(a, b, out):
    """Native masked loads need contiguous tensors, not ISA row/pointer alignment."""
    return (
        a.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and a.is_contiguous()
        and b.is_contiguous()
        and out.is_contiguous()
    )


@libentry()
@triton.jit
def _bmm_regular_kernel(
    A,
    B,
    O,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_ab: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bb: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
    DIVISIBLE_K: tl.constexpr,
    LOOP_STAGES: tl.constexpr = 1,
    A_CONTIGUOUS: tl.constexpr = False,
    B_CONTIGUOUS: tl.constexpr = False,
    B_COLUMN_MAJOR: tl.constexpr = False,
    # B==1 uses a two-dimensional launch and skips the batch pointer update.
    BATCHED: tl.constexpr = True,
    IS_FP64: tl.constexpr = False,
):
    if BATCHED:
        pid_b = ext.program_id(1).to(tl.int64)
        A += pid_b * stride_ab
        B += pid_b * stride_bb
        O += pid_b * stride_ob

    # Flatten tiles with N varying fastest, as in the original device entry.
    # This preserves operand reuse while all strides remain compile-time values.
    pid = ext.program_id(0)
    grid_n = tl.cdiv(N, TILE_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)

    if DIVISIBLE_M:
        offs_m_load = tl.max_contiguous(tl.multiple_of(offs_m % M, TILE_M), TILE_M)
    else:
        offs_m_load = offs_m
    if DIVISIBLE_N:
        offs_n_load = tl.max_contiguous(tl.multiple_of(offs_n % N, TILE_N), TILE_N)
    else:
        offs_n_load = offs_n

    if not DIVISIBLE_M:
        mask_m = offs_m < M
    if not DIVISIBLE_N:
        mask_n = offs_n < N

    if A_CONTIGUOUS:
        a_ptrs = A + offs_m_load[:, None].to(tl.int64) * stride_am + offs_k[None, :]
        a_step = TILE_K
    else:
        a_ptrs = (
            A
            + offs_m_load[:, None].to(tl.int64) * stride_am
            + offs_k[None, :].to(tl.int64) * stride_ak
        )
        a_step = TILE_K * stride_ak
    if B_COLUMN_MAJOR:
        # A transposed B has physical [N, K] order.  Load that order so the
        # K dimension is contiguous, then transpose the register tile back to
        # the logical [K, N] layout expected by tl.dot.
        b_ptrs = (
            B
            + offs_n_load[:, None].to(tl.int64) * stride_bn
            + offs_k[None, :].to(tl.int64) * stride_bk
        )
        b_step = TILE_K * stride_bk
    elif B_CONTIGUOUS:
        b_ptrs = (
            B
            + offs_k[:, None].to(tl.int64) * stride_bk
            + offs_n_load[None, :].to(tl.int64)
        )
        b_step = TILE_K * stride_bk
    else:
        b_ptrs = (
            B
            + offs_k[:, None].to(tl.int64) * stride_bk
            + offs_n_load[None, :].to(tl.int64) * stride_bn
        )
        b_step = TILE_K * stride_bk
    o_ptrs = (
        O
        + offs_m[:, None].to(tl.int64) * stride_om
        + offs_n[None, :].to(tl.int64) * stride_on
    )

    if IS_FP64:
        # Hygon's matrix-dot lowering does not support FP64 accumulation.
        # Use native Triton outer products for this correctness fallback.
        o64 = tl.zeros((TILE_M, TILE_N), dtype=tl.float64)
        for kk in range(K):
            av = tl.load(
                A + offs_m.to(tl.int64) * stride_am + kk.to(tl.int64) * stride_ak,
                offs_m < M,
                other=0.0,
            )
            bv = tl.load(
                B + kk.to(tl.int64) * stride_bk + offs_n.to(tl.int64) * stride_bn,
                offs_n < N,
                other=0.0,
            )
            o64 += av[:, None] * bv[None, :]
        tl.store(o_ptrs, o64, (offs_m < M)[:, None] & (offs_n < N)[None, :])
    else:
        o = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)

        if DIVISIBLE_K:
            loop_end = K
        else:
            loop_end = tl.cdiv(K, TILE_K) * TILE_K - TILE_K
        for _ in tl.range(
            0, loop_end, TILE_K, num_stages=LOOP_STAGES, loop_unroll_factor=1
        ):
            if DIVISIBLE_M:
                mask_a = None
            else:
                mask_a = mask_m[:, None]
            if B_COLUMN_MAJOR:
                if DIVISIBLE_N:
                    mask_b = None
                else:
                    mask_b = mask_n[:, None]
            elif DIVISIBLE_N:
                mask_b = None
            else:
                mask_b = mask_n[None, :]
            a = tl.load(a_ptrs, mask_a)
            b = tl.load(b_ptrs, mask_b)
            if B_COLUMN_MAJOR:
                b = tl.trans(b)
            o = tl.dot(a, b, o, input_precision="ieee")
            a_ptrs += a_step
            b_ptrs += b_step

        if not DIVISIBLE_K:
            rk = loop_end + offs_k
            mask_k = rk < K
            if DIVISIBLE_M:
                mask_a = mask_k[None, :]
            else:
                mask_a = mask_m[:, None] & mask_k[None, :]
            if B_COLUMN_MAJOR:
                if DIVISIBLE_N:
                    mask_b = mask_k[None, :]
                else:
                    mask_b = mask_n[:, None] & mask_k[None, :]
            elif DIVISIBLE_N:
                mask_b = mask_k[:, None]
            else:
                mask_b = mask_n[None, :] & mask_k[:, None]
            a = tl.load(a_ptrs, mask_a, other=0.0)
            b = tl.load(b_ptrs, mask_b, other=0.0)
            if B_COLUMN_MAJOR:
                b = tl.trans(b)
            o = tl.dot(a, b, o, input_precision="ieee")

        if DIVISIBLE_M and DIVISIBLE_N:
            mask_c = None
        elif DIVISIBLE_M and not DIVISIBLE_N:
            mask_c = mask_n[None, :]
        elif not DIVISIBLE_M and DIVISIBLE_N:
            mask_c = mask_m[:, None]
        else:
            mask_c = mask_m[:, None] & mask_n[None, :]
        tl.store(o_ptrs, o, mask_c)


def _launch_generic(A, B, out, batch, M, N, K):
    tile_m = 16 if M <= 16 else 32 if M <= 32 else 64
    tile_n = 16 if N <= 16 else 32 if N <= 32 else 64
    tile_k = 32
    _bmm_regular_kernel[(triton.cdiv(M, tile_m) * triton.cdiv(N, tile_n), batch)](
        A,
        B,
        out,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        A.stride(2),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        TILE_M=tile_m,
        TILE_N=tile_n,
        TILE_K=tile_k,
        DIVISIBLE_M=M % tile_m == 0,
        DIVISIBLE_N=N % tile_n == 0,
        DIVISIBLE_K=K % tile_k == 0,
        LOOP_STAGES=1,
        A_CONTIGUOUS=A.stride(2) == 1 and A.stride(1) == K,
        B_CONTIGUOUS=B.stride(2) == 1 and B.stride(1) == N,
        B_COLUMN_MAJOR=B.stride(1) == 1 and B.stride(2) == K,
        BATCHED=batch != 1,
        IS_FP64=A.dtype == torch.float64,
        num_warps=4,
        num_stages=1,
        allow_flush_denorm=A.dtype != torch.float64,
        enable_fp_fusion=True,
    )


@libentry()
@triton.jit
def _pack_b(
    B,
    P,
    K: tl.constexpr,
    N: tl.constexpr,
    BB: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    nk = tl.cdiv(K, BLOCK)
    tile = tl.program_id(0)
    batch = tl.program_id(1).to(tl.int64)
    # Read physical [N,K] order for column-major B. The output is [K,N].
    rn = (tile // nk * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    rk = (tile % nk * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    x = tl.load(
        B + batch * BB + rn[:, None] * BN + rk[None, :] * BK,
        (rn[:, None] < N) & (rk[None, :] < K),
        other=0,
    )
    tl.store(
        P + batch * K * N + rk[None, :] * N + rn[:, None],
        x,
        (rn[:, None] < N) & (rk[None, :] < K),
    )


def pack_b(b, block=64):
    batch, k, n = b.shape
    p = torch.empty((batch, k, n), dtype=b.dtype, device=b.device)
    _pack_b[(triton.cdiv(k, block) * triton.cdiv(n, block), batch)](
        b, p, k, n, *b.stride(), block, num_warps=4, num_stages=1
    )
    return p


def supports_packing(a, b, out):
    return (
        supports_isa(a)
        and a.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and a.is_contiguous()
        and out.is_contiguous()
        and not b.is_contiguous()
        and a.data_ptr() % 16 == 0
        and out.data_ptr() % 16 == 0
    )


def launch_packed_if_supported(a, b, out):
    if not supports_packing(a, b, out):
        return False
    batch, m, k = a.shape
    n = b.shape[2]
    if m < 256:
        return False
    family, config = select_kernel(a.dtype, batch, m, n, k)
    if family not in ("ring", "regular"):
        return False
    p = pack_b(b)
    if family == "ring":
        ring.launch_ring(a, p, out)
    else:
        regular.launch_regular(a, p, out, config)
    return True


@libentry()
@triton.jit
def _partials(
    A,
    B,
    P,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    AB: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    BB: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    BATCH: tl.constexpr,
    SPLITS: tl.constexpr,
    TM: tl.constexpr,
    TN: tl.constexpr,
    TK: tl.constexpr,
):
    tile = tl.program_id(0)
    batch = tl.program_id(1)
    split = tl.program_id(2)
    gn = tl.cdiv(N, TN)
    rm = tile // gn * TM + tl.arange(0, TM)
    rn = tile % gn * TN + tl.arange(0, TN)
    rk = tl.arange(0, TK)
    segment: tl.constexpr = triton.cdiv(K, SPLITS * TK) * TK
    start = split * segment
    acc = tl.zeros((TM, TN), tl.float32)
    for offset in tl.range(0, segment, TK, num_stages=2):
        kk = start + offset + rk
        a = tl.load(
            A
            + batch.to(tl.int64) * AB
            + rm[:, None].to(tl.int64) * AM
            + kk[None, :].to(tl.int64) * AK,
            (rm[:, None] < M) & (kk[None, :] < K),
            other=0,
        )
        b = tl.load(
            B
            + batch.to(tl.int64) * BB
            + kk[:, None].to(tl.int64) * BK
            + rn[None, :].to(tl.int64) * BN,
            (kk[:, None] < K) & (rn[None, :] < N),
            other=0,
        )
        acc = tl.dot(a, b, acc, input_precision="ieee")
    offsets = (
        (split * BATCH + batch).to(tl.int64) * M * N
        + rm[:, None].to(tl.int64) * N
        + rn[None, :]
    )
    tl.store(P + offsets, acc, (rm[:, None] < M) & (rn[None, :] < N))


@libentry()
@triton.jit
def _reduce(
    P,
    O,
    M: tl.constexpr,
    N: tl.constexpr,
    BATCH: tl.constexpr,
    OB: tl.constexpr,
    OM: tl.constexpr,
    ON: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    size: tl.constexpr = BATCH * M * N
    split = tl.arange(0, SPLITS)
    values = tl.load(
        P + split[:, None].to(tl.int64) * size + idx[None, :],
        idx[None, :] < size,
        other=0,
    )
    result = tl.sum(values, axis=0)
    batch = idx // (M * N)
    row = idx // N % M
    col = idx % N
    tl.store(
        O + batch.to(tl.int64) * OB + row.to(tl.int64) * OM + col.to(tl.int64) * ON,
        result,
        idx < size,
    )


def launch_splitk(a, b, out, config):
    batch, m, k = a.shape
    n = b.shape[2]
    splits, tm, tn, tk = config
    partials = torch.empty((splits, batch, m, n), dtype=torch.float32, device=a.device)
    _partials[(triton.cdiv(m, tm) * triton.cdiv(n, tn), batch, splits)](
        a,
        b,
        partials,
        m,
        n,
        k,
        *a.stride(),
        *b.stride(),
        batch,
        splits,
        tm,
        tn,
        tk,
        num_warps=4,
        num_stages=2,
        allow_flush_denorm=True,
    )
    _reduce[(triton.cdiv(batch * m * n, 256),)](
        partials,
        out,
        m,
        n,
        batch,
        *out.stride(),
        splits,
        256,
        num_warps=4,
        num_stages=1,
    )


def launch_splitk_if_supported(a, b, out):
    batch, m, k = a.shape
    n = b.shape[2]
    family, config = select_kernel(a.dtype, batch, m, n, k)
    if family != "splitk":
        return False
    launch_splitk(a, b, out, config)
    return True


def _supports_regular_layout(a, b, out, config):
    if config in regular.NATIVE_REGULAR_KEYS:
        return supports_native_layout(a, b, out)
    return supports_vector_layout(a, b, out)


def _launch_tle_regular(a, b, out):
    batch, m, k = a.shape
    n = b.shape[2]
    family, config = select_kernel(a.dtype, batch, m, n, k)
    if family != "regular" or not _supports_regular_layout(a, b, out, config):
        return False
    regular.launch_regular(a, b, out, config)
    return True


def _launch_tle_ring(a, b, out):
    batch, m, k = a.shape
    n = b.shape[2]
    family, config = select_kernel(a.dtype, batch, m, n, k)
    if family != "ring" or not supports_vector_layout(a, b, out):
        return False
    ring.launch_ring(a, b, out)
    return True


_KEYS = [
    "BATCH",
    "M",
    "N",
    "K",
    "AB",
    "AM",
    "AK",
    "BB",
    "BK",
    "BN",
    "OB",
    "OM",
    "ON",
    "VECTOR",
    "PACKED",
    "DEVICE",
]
_ARGS = ["A", "B", "O"] + _KEYS
_TUNER = None
_LOCK = threading.RLock()


def _bmm_pipeline(
    A,
    B,
    O,
    BATCH,
    M,
    N,
    K,
    AB,
    AM,
    AK,
    BB,
    BK,
    BN,
    OB,
    OM,
    ON,
    VECTOR,
    PACKED,
    DEVICE,
    FAMILY,
    TM,
    TN,
    TK,
    GROUP,
    SPLITS,
    PACK=64,
    num_warps=4,
    num_stages=1,
    **unused,
):
    if FAMILY in (4, 5):
        B = pack_b(B, PACK)
        FAMILY = 1 if FAMILY == 4 else 2
    if FAMILY == 1:
        dt = {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}[A.dtype]
        kernel = regular.REGULAR_KERNELS[(dt, TM, TN, TK, 2, 2)]
        kernel[(triton.cdiv(M, TM) * triton.cdiv(N, TN), BATCH)](
            A,
            B,
            O,
            M,
            N,
            K,
            GROUP,
            num_warps=4,
            num_stages=1,
            allow_flush_denorm=True,
            enable_fp_fusion=True,
        )
    elif FAMILY == 2:
        ring.launch_ring(A, B, O, group=GROUP)
    elif FAMILY == 3:
        # Both launches and workspace allocation are in the measured callable.
        launch_splitk(A, B, O, (SPLITS, TM, TN, TK))
    else:
        _bmm_regular_kernel[(triton.cdiv(M, TM) * triton.cdiv(N, TN), BATCH)](
            A,
            B,
            O,
            M,
            N,
            K,
            AB,
            AM,
            AK,
            BB,
            BK,
            BN,
            OB,
            OM,
            ON,
            TILE_M=TM,
            TILE_N=TN,
            TILE_K=TK,
            DIVISIBLE_M=M % TM == 0,
            DIVISIBLE_N=N % TN == 0,
            DIVISIBLE_K=K % TK == 0,
            LOOP_STAGES=num_stages,
            A_CONTIGUOUS=AK == 1 and AM == K,
            B_CONTIGUOUS=BN == 1 and BK == N,
            B_COLUMN_MAJOR=BK == 1 and BN == K,
            BATCHED=BATCH != 1,
            IS_FP64=A.dtype == torch.float64,
            num_warps=num_warps,
            num_stages=num_stages,
            allow_flush_denorm=A.dtype != torch.float64,
            enable_fp_fusion=True,
        )
    return O


class _Pipeline:
    fn = staticmethod(_bmm_pipeline)
    arg_names = _ARGS

    def run(self, *args, **kwargs):
        return _bmm_pipeline(*args, **kwargs)


def _prune(configs, named_args, **kwargs):
    a = {**named_args, **kwargs}
    batch, m, n, k = (a[x] for x in ("BATCH", "M", "N", "K"))
    dtype = a["A"].dtype
    es = a["A"].element_size()
    bounded = max(batch * m * k, batch * k * n, batch * m * n) * es < 2**31
    vector = a["VECTOR"] and bounded
    native = supports_native_layout(a["A"], a["B"], a["O"])
    packed = supports_packing(a["A"], a["B"], a["O"]) and bounded
    dt = {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}.get(dtype)
    default_m = 16 if m <= 16 else 32 if m <= 32 else 64
    default_n = 16 if n <= 16 else 32 if n <= 32 else 64
    choices = []
    for c in configs:
        p = c.kwargs
        family, tm, tn, tk = (p[x] for x in ("FAMILY", "TM", "TN", "TK"))
        if family == 0:
            # Bounded nearby candidates; includes the previous generic default.
            if tm not in (default_m, min(128, 2 * default_m)):
                continue
            if tn not in (default_n, min(128, 2 * default_n)):
                continue
            if p["GROUP"] != 1 or (tm * tn >= 16384 and c.num_stages > 1):
                continue
            if dtype == torch.float64 and (tm > 64 or tn > 64):
                continue
        elif family in (1, 4):
            key = (dt, tm, tn, tk, 2, 2)
            if key not in regular.REGULAR_KERNELS:
                continue
            if key in regular.NATIVE_REGULAR_KEYS:
                if not (native if family == 1 else packed):
                    continue
            elif (
                not (vector if family == 1 else packed)
                or min(m, n) < 32
                or n % (16 // es)
                or k % tk
                or ((m + tm - 1) // tm) * tm * n * es >= 2**31
            ):
                continue
        elif family in (2, 5):
            if dt not in (0, 1) or min(m, n) < 256 or k < 256:
                continue
            if (
                not (vector if family == 2 else packed)
                or n % 8
                or k % 64
                or ((m + 255) // 256) * 256 * n * es >= 2**31
            ):
                continue
            padded = triton.cdiv(m, 256) * triton.cdiv(n, 256) * 256**2
            if 2 * m * n < padded:
                continue
        elif family == 3:
            tiles = batch * triton.cdiv(m, tm) * triton.cdiv(n, tn)
            if (
                dt is None
                or k < 1024
                or tiles >= 128
                or k < p["SPLITS"] * tk
                or batch * m * n * p["SPLITS"] * 4 >= 2**31
            ):
                continue
        else:
            continue
        choices.append(c)
    if not choices:
        raise RuntimeError("BMM FlagTune search has no legal candidate")
    return choices


class _PipelineTuner(LibTuner.get("default")):
    def _flagtune_configs_for_mode(self, op_name, mode):
        if mode is not runtime.TuningMode.EXPANDED:
            return self._flagtune_default_configs, self._flagtune_default_strategy
        # Use the official vendor config loader for the whole-pipeline domain.
        # The standard expanded BMM schema describes a single Triton kernel.
        return runtime.get_tuned_config(op_name), "default"

    @property
    def cache_key(self):
        # LibTuner normally derives this from one JIT function. This pipeline
        # spans several kernels, so invalidate caches on ANY implementation edit.
        root = Path(__file__).parent
        files = (
            "bmm.py",
            "bmm_ring.py",
            "bmm_regular.py",
            "../tune_configs.yaml",
        )
        return hashlib.sha256(
            b"".join((root / f).read_bytes() for f in files)
        ).hexdigest()


def _get_tuner():
    global _TUNER
    if _TUNER is None:
        _TUNER = _PipelineTuner(
            _Pipeline(),
            _ARGS,
            [
                triton.Config(
                    {
                        "FAMILY": 0,
                        "TM": 64,
                        "TN": 64,
                        "TK": 32,
                        "GROUP": 1,
                        "SPLITS": 1,
                        "PACK": 64,
                    },
                    num_warps=4,
                    num_stages=1,
                )
            ],
            _KEYS,
            None,
            None,
            strategy="default",
            prune_configs_by={"early_config_prune": _prune},
            warmup=2,
            rep=80,
            use_cuda_graph=True,
            flagtune_op_name="bmm",
        )
    return _TUNER


def _launch_autotuned(a, b, out):
    if (
        runtime.resolve_tuning_mode("bmm", supports_cost_model=False)
        is not runtime.TuningMode.EXPANDED
    ):
        return False
    batch, m, k = a.shape
    n = b.shape[2]
    if k == 0:
        return False
    props = torch.cuda.get_device_properties(a.device)
    device = (
        f"{props.name}:{getattr(props, 'gcnArchName', 'unknown')}:"
        f"{props.major}.{props.minor}:{a.device.index}:"
        f"{torch.version.hip}:{triton.__version__}"
    )
    args = (
        a,
        b,
        out,
        batch,
        m,
        n,
        k,
        *a.stride(),
        *b.stride(),
        *out.stride(),
        supports_vector_layout(a, b, out),
        supports_packing(a, b, out),
        device,
    )
    # The tuner mutates nargs/best_config while selecting and launching.
    # Serialize those operations, but leave all cache access to LibTuner.
    with _LOCK:
        tuner = _get_tuner()
        tuner.apply_flagtune()
        tuner.run(*args)
    return True


def _dispatch_bmm(A, B, out):
    batch, M, K = A.shape
    N = B.shape[2]
    if batch == 0 or M == 0 or N == 0:
        return out
    with torch_device_fn.device(A.device):
        if _launch_autotuned(A, B, out):
            return out
        if _launch_tle_ring(A, B, out):
            return out
        if _launch_tle_regular(A, B, out):
            return out
        if launch_splitk_if_supported(A, B, out):
            return out
        if launch_packed_if_supported(A, B, out):
            return out
        _launch_generic(A, B, out, batch, M, N, K)
    return out


def _validate_inputs(A, B):
    if A.ndim != 3 or B.ndim != 3:
        raise RuntimeError("bmm expects two 3-dimensional tensors")
    if A.shape[0] != B.shape[0] or A.shape[2] != B.shape[1]:
        raise RuntimeError("bmm input batch or reduction dimensions do not match")
    if A.dtype != B.dtype or A.device != B.device:
        raise RuntimeError("bmm inputs must have the same dtype and device")
    if A.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise RuntimeError(f"bmm does not support {A.dtype}")


def bmm(A, B):
    logger.debug("GEMS BMM")
    _validate_inputs(A, B)
    batch, M, K = A.shape
    N = B.shape[2]
    out = torch.empty((batch, M, N), dtype=A.dtype, device=A.device)
    return _dispatch_bmm(A, B, out)


def bmm_out(A, B, out):
    logger.debug("GEMS BMM_OUT")
    _validate_inputs(A, B)
    if out.dtype != A.dtype or out.device != A.device:
        raise RuntimeError("bmm output must have the input dtype and device")
    expected = (A.shape[0], A.shape[1], B.shape[2])
    if tuple(out.shape) != expected:
        out.resize_(expected)
    return _dispatch_bmm(A, B, out)
