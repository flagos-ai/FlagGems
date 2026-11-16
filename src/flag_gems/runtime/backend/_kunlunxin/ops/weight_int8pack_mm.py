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

"""Kunlunxin XPU3 override for ``aten::_weight_int8pack_mm``.

``C = (A @ dequant(B).T) * scales`` with ``A`` fp16/bf16, ``B`` int8 ``(N, K)``
and per-row ``scales`` ``(N,)``.

Why a backend override
----------------------
The generic implementation runs a fixed 8-config ``@libtuner`` sweep and, more
importantly, feeds ``tl.dot`` from **masked** 2-D loads whose K-mask is always
true for the supported shapes.  On XPU3 a ``tl.dot`` kernel with any masked
2-D load (even a constant-true mask) is silently miscompiled (KERNEL_OPT rows
17 / 72): the functional baseline returned NaN for every shape and the LLM-scale
benchmark shapes faulted the kernel (-714 / status 719).

Design
------
Mirror the proven ``mm_kernel`` scheme (see ``ops/mm.py``): in-bounds-by-
construction addressing, **no masks inside the dot loop**, and an unmasked store
into a padded output buffer that the host slices.  ``K`` and ``N`` are always
handled without a mask by choosing tile sizes that divide them (or by
zero-padding ``K`` through a non-dot copy kernel when they do not).  Rows are
``%-wrapped`` (``rm % M``) so the last partial M-tile still reads inside ``A``;
``M == 1`` therefore reads row 0 for all lanes (a broadcast, no special path).

A broadcast ``tl.dot`` built via ``tl.broadcast_to(a[None, :], (16, BK))`` was
measured to return NaN on XPU3; the ``% M`` row wrap is the working form.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_GROUP_M = 8
_MM_MIN_LANES = 16  # tl.dot needs >= 16 rows; also the XPU arange floor.


@triton.jit
def _w8_grouped_pids(
    pid, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_M: tl.constexpr
):
    """Grouped program-id -> (pid_m, pid_n) with L2 reuse swizzle."""
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size
    return pid_m, pid_n


@libentry()
@triton.jit
def _w8_mm_kernel(
    A,
    B,
    Scales,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    stride_scales,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """C[rm, rn] = (sum_k A[rm, k] * B[rn, k]) * scales[rn].

    ``C`` is the padded buffer ``(cdiv(M,BLOCK_M)*BLOCK_M, ...)``; the store is
    unmasked and every access is in bounds because rows/cols are wrapped with
    ``% M`` / ``% N``.  ``M == 1`` makes ``rm % M == 0``: every dot row reads
    row 0 of ``A`` (a broadcast) and the host keeps only row 0.
    """
    pid = tl.program_id(0)
    pid_m, pid_n = _w8_grouped_pids(pid, M, N, BLOCK_M, BLOCK_N, GROUP_M)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = rm % M
    rbn = rn % N

    rk = tl.arange(0, BLOCK_K)
    a_ptrs = A + ram[:, None] * stride_am + rk[None, :] * stride_ak
    b_ptrs = B + rbn[:, None] * stride_bn + rk[None, :] * stride_bk

    compute_dtype = C.dtype.element_ty
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        b = b.to(compute_dtype)
        acc += tl.dot(a, tl.trans(b), out_dtype=tl.float32, allow_tf32=False)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    s = tl.load(Scales + rbn * stride_scales).to(tl.float32)
    acc = acc * s[None, :]

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(compute_dtype))


@triton.jit
def _w8_copy_kernel(
    Src,
    Dst,
    R,
    Ccols,
    stride_sr,
    stride_sc,
    stride_dr,
    stride_dc,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Plain (non-dot) masked 2-D copy: pad/copy buffers.

    Not a dot kernel, so ordinary bound masks are safe here (same pattern as
    ``mm.py::_syrk_guard_kernel``).
    """
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    rr = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    rc = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (rr < R)[:, None] & (rc < Ccols)[None, :]
    x = tl.load(
        Src + rr[:, None] * stride_sr + rc[None, :] * stride_sc, mask=mask, other=0
    )
    tl.store(Dst + rr[:, None] * stride_dr + rc[None, :] * stride_dc, x, mask=mask)


def _pad_cols(src, rows, cols, pad_to, device):
    """Zero-pad ``src`` (rows, cols) to (rows, pad_to) along the last axis."""
    out = torch.zeros((rows, pad_to), device=device, dtype=src.dtype)
    br, bc = 64, 64
    _w8_copy_kernel[(triton.cdiv(rows, br), triton.cdiv(cols, bc))](
        src,
        out,
        rows,
        cols,
        src.stride(0),
        src.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_R=br,
        BLOCK_C=bc,
    )
    return out


def _pick_block_m(M):
    if M <= _MM_MIN_LANES:
        return _MM_MIN_LANES
    if M <= 32:
        return 32
    if M <= 64:
        return 64
    return 128


def _pick_block_n(N):
    # Measured (XPU3, fopen16/bf16, LLM shapes): BLOCK_K=128 is the key win and
    # BLOCK_N=256 helps only when N is large enough to keep the program count
    # high (N >= 8192); at N == 4096 the wider tile spills and regresses.
    if N >= 8192 and N % 256 == 0:
        return 256
    if N % 128 == 0:
        return 128
    if N % 64 == 0:
        return 64
    return 64


def _pick_block_k(K):
    if K % 128 == 0:
        return 128
    if K % 64 == 0:
        return 64
    if K % 32 == 0:
        return 32
    return 16


def weight_int8pack_mm(A, B, scales):
    logger.debug("GEMS_KUNLUNXIN WEIGHT_INT8PACK_MM")

    assert (
        A.shape[1] == B.shape[1]
    ), f"incompatible K dimensions: A.shape[1]={A.shape[1]}, B.shape[1]={B.shape[1]}"
    assert (
        B.shape[0] == scales.shape[0]
    ), f"incompatible N dimensions: B.shape[0]={B.shape[0]}, scales.shape[0]={scales.shape[0]}"
    assert B.dtype == torch.int8, f"B must be int8, got {B.dtype}"

    M, K = A.shape
    N = B.shape[0]
    if M == 0 or N == 0 or K == 0:
        return torch.empty((M, N), device=A.device, dtype=A.dtype)

    if scales.dim() == 2 and 1 in scales.shape:
        scales = scales.reshape(-1)

    device = A.device
    BLOCK_M = _pick_block_m(M)
    BLOCK_N = _pick_block_n(N)
    BLOCK_K = _pick_block_k(K)

    # K must be a multiple of BLOCK_K for the unmasked loop.  K is divisible by
    # 128 for every LLM weight shape; the copy path is a safety net for ragged K.
    if K % BLOCK_K != 0:
        kp = triton.cdiv(K, BLOCK_K) * BLOCK_K
        A = _pad_cols(A, M, K, kp, device)
        B = _pad_cols(B, N, K, kp, device)
        K = kp
    stride_ak = A.stride(1)
    stride_bk = B.stride(1)

    # Padded output: the store is always unmasked, so C must cover every tile.
    Mp = triton.cdiv(M, BLOCK_M) * BLOCK_M
    Np = triton.cdiv(N, BLOCK_N) * BLOCK_N
    C = torch.empty((Mp, Np), device=device, dtype=A.dtype)

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    kernel = _w8_mm_kernel
    with torch_device_fn.device(device):
        kernel[grid](
            A,
            B,
            scales,
            C,
            M,
            N,
            K,
            A.stride(0),
            stride_ak,
            B.stride(0),
            stride_bk,
            C.stride(0),
            C.stride(1),
            scales.stride(0),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            GROUP_M=_GROUP_M,
        )

    if Mp == M and Np == N:
        return C
    # Row-only padding of a row-major buffer is a contiguous view.
    if Np == N:
        return C[:M, :]
    out = torch.empty((M, N), device=device, dtype=A.dtype)
    _w8_copy_kernel[(triton.cdiv(M, 64), triton.cdiv(N, 64))](
        C,
        out,
        M,
        N,
        C.stride(0),
        C.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_R=64,
        BLOCK_C=64,
    )
    return out
