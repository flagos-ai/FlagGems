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

import logging

import torch
import triton
import triton.language as tl

# from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


# GEMV tile sizes for the dot-based kernel: (BLOCK_M rows, BLOCK_K K-slice,
# BLOCK_N dot-width).  The matvec is a (BLOCK_M, BLOCK_K) x (BLOCK_K,
# BLOCK_N) `tl.dot` whose right operand is the K-vector broadcast into
# BLOCK_N identical columns; the backend's 2D axis-1 reductions are
# slow/restricted for wide tiles, but the hardware dot lowers the K-reduction.
# BLOCK_N=16 is the smallest the dot accepts, bounding the redundant MAC and
# store traffic (16x) instead of the 128x (BLOCK_N=128) of the mm-delegated
# path this replaces.  Measured (XPU 3, 2026-10-14): (64,64) beats
# (32,128)/(64,128) on 2K shapes; (128,64) wins on the 16K shapes.
_GV_TILE_SMALL = (64, 64, 16)  # (BLOCK_M, BLOCK_K, BLOCK_N) for N < 4096
_GV_TILE_LARGE = (128, 64, 16)
_GV_LARGE_N = 4096
_GV_WARPS = 8


def heur_block_n(args):
    return triton.next_power_of_2(triton.cdiv(args["N"], 12))


def heur_block_m(args):
    import builtins

    return builtins.min(triton.next_power_of_2(args["M"]), 4096)


@libentry()
# @triton.autotune(
#     configs=runtime.get_tuned_config("mv"),
#     key=["M", "N"],
# )
@triton.heuristics(
    {
        "BLOCK_N": heur_block_n,
        "BLOCK_M": heur_block_m,
    }
)
@triton.jit
def mv_kernel(
    A,
    B,
    C,
    N: tl.constexpr,
    M: tl.constexpr,
    stride_an: tl.constexpr,
    stride_am: tl.constexpr,
    stride_bm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    buffer_size_limit: tl.constexpr,  # NOTE: `constexpr` so it can be used as a shape value.
):
    pid = ext.program_id(0)
    offset_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)[:, None]
    offset_m = tl.arange(0, BLOCK_M)[None, :]
    n_mask = offset_n < N
    A_ptrs = A + offset_n * stride_an + offset_m * stride_am
    B_ptrs = B + offset_m * stride_bm
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for m in range(0, M, BLOCK_M):
        m_mask = m + offset_m < M
        a = tl.load(A_ptrs, mask=n_mask & m_mask, other=0.0).to(tl.float32)
        b = tl.load(B_ptrs, mask=m_mask, other=0.0).to(tl.float32)
        acc += a * b
        A_ptrs += BLOCK_M * stride_am
        B_ptrs += BLOCK_M * stride_bm

    acc = tl.sum(acc, axis=1)
    C_ptrs = C + offset_n * stride_cn
    tl.store(C_ptrs, acc[:, None], mask=n_mask)


@libentry()
@triton.jit
def mv_gemv_kernel(
    A,
    B,
    C,
    N,
    M,
    stride_an,
    stride_am,
    stride_bm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """GEMV: one program computes BLOCK_M output rows, looping over K.

    The K-reduction is a (BLOCK_M, BLOCK_K) x (BLOCK_K, BLOCK_N) hardware dot
    whose right operand is the K-vector broadcast into BLOCK_N identical
    columns; all BLOCK_N components are bit-identical so column 0 is the
    result (recovered as sum(acc, axis=1) / BLOCK_N, exact for power-of-two
    BLOCK_N; a one-hot tl.where select is unusable -- the i1-select pattern
    faults the XPU backend).  allow_tf32=False keeps the f32 accumulation at
    full precision (tf32 gives rel err ~8e-4, which fails the f32 1.3e-6 test
    tolerance).

    Every access is fully affine and unmasked: ragged rows are handled by the
    `%`-wrap (same device-tested pattern as mm_kernel: lanes past the last
    row re-read valid rows and the store mask discards them), never by a
    per-lane min/max clamp -- a clamped index makes the loads non-affine
    (~1000x slower) and the clamped-dot combination faults the device
    (KL_XID_KERNEL_EXCEPTION / status 700).  Ragged K only occurs for odd M
    and is handled by host-padding B to a multiple of BLOCK_K (zeros
    annihilate the wrapped A re-reads).
    """
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rc = rows % N
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, M, BLOCK_K):
        a = tl.load(A + rc[:, None] * stride_an + (k0 + rk)[None, :] * stride_am)
        b = tl.load(B + (k0 + rk) * stride_bm)
        b2 = tl.broadcast_to(b[None, :].T, (BLOCK_K, BLOCK_N))
        acc += tl.dot(
            a.to(tl.float32),
            b2.to(tl.float32),
            out_dtype=tl.float32,
            allow_tf32=False,
        )
    s = tl.sum(acc, axis=1) * (1.0 / BLOCK_N)
    tl.store(C + rows * stride_cn, s, mask=rows < N)


def _mv_rows(inp, vec, out, N, M):
    """Launch mv_gemv_kernel.  Only called when some tile in {64, 32, 16}
    divides M, so the K-loop tiles exactly and every load is in-bounds; the
    row side needs no re-layout (kernel-side %-wrap)."""
    if N >= _GV_LARGE_N:
        BLOCK_M, BLOCK_K, BLOCK_N = _GV_TILE_LARGE
    else:
        BLOCK_M, BLOCK_K, BLOCK_N = _GV_TILE_SMALL
    # Pick the largest tile that exactly tiles the K dimension.
    for bk in (BLOCK_K, 32, 16):
        if M % bk == 0:
            BLOCK_K = bk
            break
    with torch_device_fn.device(inp.device):
        mv_gemv_kernel[(triton.cdiv(N, BLOCK_M),)](
            inp,
            vec,
            out,
            N,
            M,
            inp.stride(0),
            inp.stride(1),
            vec.stride(0),
            out.stride(0),
            BLOCK_M=BLOCK_M,
            BLOCK_K=BLOCK_K,
            BLOCK_N=BLOCK_N,
            num_warps=_GV_WARPS,
        )
    return out


def mv(inp, vec):
    logger.debug("GEMS_KUNLUNXIN MV")
    assert inp.shape[1] == vec.shape[0], "incompatible dimensions"
    N, M = inp.shape
    # TODO: fix autotune config has no item
    if M == 5333 and N == 497:
        return mv_cluster(inp, vec)

    out = torch.empty((N,), device=inp.device, dtype=inp.dtype)
    with torch_device_fn.device(inp.device):
        if M == 1 or N < 16:
            # The dot kernel needs BLOCK_M >= 16; tiny-N shapes go through the
            # proven 2D-tile+sum kernel (single program, whole-K tile).
            grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]),)
            mv_kernel[grid](
                inp,
                vec,
                out,
                N,
                M,
                inp.stride(0),
                inp.stride(1),
                vec.stride(0),
                out.stride(0),
                buffer_size_limit=256,
            )
        else:
            # The dot kernel requires BLOCK_K to exactly tile M.  When no
            # tile in {64, 32, 16} divides M (odd M), keep the proven
            # masked 2-D tile kernel instead: a host-padded M causes the
            # runtime-trip-count K-loop to miscompile on this backend
            # (correct in-bounds padded inputs still yield garbage, verified
            # 2026-10-14), and the K-loop cannot read past the row without
            # masking.
            if M % 16 == 0:
                return _mv_rows(inp, vec, out, N, M)
            grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]),)
            mv_kernel[grid](
                inp,
                vec,
                out,
                N,
                M,
                inp.stride(0),
                inp.stride(1),
                vec.stride(0),
                out.stride(0),
                buffer_size_limit=256,
            )
    return out


def mv_cluster(inp, vec):
    logger.debug("GEMS_KUNLUNXIN MV")
    assert inp.shape[1] == vec.shape[0], "incompatible dimensions"
    N, M = inp.shape
    out = torch.empty((N,), device=inp.device, dtype=inp.dtype)
    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]),)
    with torch_device_fn.device(inp.device):
        mv_kernel[grid](
            inp,
            vec,
            out,
            N,
            M,
            inp.stride(0),
            inp.stride(1),
            vec.stride(0),
            out.stride(0),
            buffer_size_limit=256,
        )
    return out
