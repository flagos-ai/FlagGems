import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_DET_BLOCK_MAX = 64


@triton.jit
def _reduce_mul(a, b):
    return a * b


@libentry()
@triton.jit
def _det_register_kernel(
    A,
    out,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)
    rows = tl.arange(0, BLOCK_N)
    cols = tl.arange(0, BLOCK_N)

    base = pid * N * N
    offsets = base + rows[:, None] * N + cols[None, :]
    load_mask = (rows[:, None] < N) & (cols[None, :] < N)
    work = tl.load(A + offsets, mask=load_mask, other=0.0)

    swap_count = tl.zeros((), dtype=tl.int32)

    for k in range(N):
        col_k = tl.reshape(
            tl.gather(work, tl.full((BLOCK_N, 1), k, tl.int32), axis=1), (BLOCK_N,)
        )
        abs_col = tl.abs(col_k)
        abs_col = tl.where((rows < k) | (rows >= N), -1.0, abs_col)
        pivot_val = tl.max(abs_col, axis=0)
        pivot_row = tl.min(tl.where(abs_col == pivot_val, rows, BLOCK_N), axis=0)

        if pivot_row != k:
            row_k = tl.reshape(
                tl.gather(work, tl.full((1, BLOCK_N), k, tl.int32), axis=0), (BLOCK_N,)
            )
            row_p = tl.reshape(
                tl.gather(work, tl.full((1, BLOCK_N), pivot_row, tl.int32), axis=0),
                (BLOCK_N,),
            )
            work = tl.where(rows[:, None] == k, row_p[None, :], work)
            work = tl.where(rows[:, None] == pivot_row, row_k[None, :], work)
            swap_count += 1

        col_k = tl.reshape(
            tl.gather(work, tl.full((BLOCK_N, 1), k, tl.int32), axis=1), (BLOCK_N,)
        )
        pivot = tl.sum(col_k * (rows == k).to(tl.float32), axis=0)
        safe_pivot = pivot + (pivot == 0.0).to(tl.float32)
        l_col = col_k / safe_pivot * (rows > k).to(tl.float32)
        u_row = tl.reshape(
            tl.gather(work, tl.full((1, BLOCK_N), k, tl.int32), axis=0), (BLOCK_N,)
        )

        mask2d = ((rows[:, None] > k) & (cols[None, :] > k)).to(tl.float32)
        work = work - l_col[:, None] * u_row[None, :] * mask2d

    diag = tl.reshape(tl.gather(work, rows[:, None], axis=1), (BLOCK_N,))
    diag = diag * (cols < N).to(tl.float32) + (cols >= N).to(tl.float32)
    det = tl.reduce(diag, 0, combine_fn=_reduce_mul)
    det = det * (1.0 - 2.0 * (swap_count % 2).to(tl.float32))
    tl.store(out + pid, det)


@libentry()
@triton.jit
def _det_blocked_kernel(
    A,
    out,
    N,
    BLOCK: tl.constexpr,
):
    pid = tle.program_id(0)
    base = pid * N * N
    swap_count = tl.zeros((), dtype=tl.int32)

    for k in range(N):
        best_val = tl.full((), -1.0, dtype=A.dtype.element_ty)
        best_row = tl.full((), k, dtype=tl.int32)
        for i0 in range(k, N, BLOCK):
            rows = i0 + tl.arange(0, BLOCK)
            col = tl.load(A + base + rows * N + k, mask=rows < N, other=0.0)
            abs_col = tl.where((rows >= k) & (rows < N), tl.abs(col), -1.0)
            tile_max = tl.max(abs_col, axis=0)
            tile_row = tl.min(tl.where(abs_col == tile_max, rows, N), axis=0)
            is_better = tile_max > best_val
            best_row = tl.where(is_better, tile_row, best_row)
            best_val = tl.where(is_better, tile_max, best_val)

        for j0 in range(k, N, BLOCK):
            cols = j0 + tl.arange(0, BLOCK)
            cmask = cols < N
            row_k = tl.load(A + base + k * N + cols, mask=cmask, other=0.0)
            row_p = tl.load(A + base + best_row * N + cols, mask=cmask, other=0.0)
            tl.store(A + base + k * N + cols, row_p, mask=cmask)
            tl.store(A + base + best_row * N + cols, row_k, mask=cmask)
        swap_count = tl.where(best_row != k, swap_count + 1, swap_count)

        tl.debug_barrier()

        pivot = tl.load(A + base + k * N + k)
        safe_pivot = tl.where(pivot == 0.0, 1.0, pivot)

        for i0 in range(k + 1, N, BLOCK):
            rows = i0 + tl.arange(0, BLOCK)
            rmask = rows < N
            col = tl.load(A + base + rows * N + k, mask=rmask, other=0.0)
            tl.store(A + base + rows * N + k, col / safe_pivot, mask=rmask)

        tl.debug_barrier()

        for i0 in range(k + 1, N, BLOCK):
            rows = i0 + tl.arange(0, BLOCK)
            rmask = rows < N
            l_col = tl.load(A + base + rows * N + k, mask=rmask, other=0.0)
            for j0 in range(k + 1, N, BLOCK):
                cols = j0 + tl.arange(0, BLOCK)
                cmask = cols < N
                u_row = tl.load(A + base + k * N + cols, mask=cmask, other=0.0)
                tmask = rmask[:, None] & cmask[None, :]
                tile = tl.load(
                    A + base + rows[:, None] * N + cols[None, :], mask=tmask, other=0.0
                )
                tile = tile - l_col[:, None] * u_row[None, :]
                tl.store(A + base + rows[:, None] * N + cols[None, :], tile, mask=tmask)

        tl.debug_barrier()

    det = tl.full((), 1.0, dtype=A.dtype.element_ty)
    for i0 in range(0, N, BLOCK):
        d = i0 + tl.arange(0, BLOCK)
        diag = tl.load(A + base + d * N + d, mask=d < N, other=1.0)
        det = det * tl.reduce(diag, 0, combine_fn=_reduce_mul)
    det = tl.where(swap_count % 2 == 0, det, -det)
    tl.store(out + pid, det)


def linalg_det(A, *, out=None):
    logger.debug("GEMS LINALG_DET")
    if out is not None:
        if out.dtype != A.dtype:
            raise RuntimeError(
                f"linalg_det: dtype of out ({out.dtype}) does not match "
                f"dtype of input ({A.dtype})"
            )
        if out.device != A.device:
            raise RuntimeError(
                f"linalg_det: device of out ({out.device}) does not match "
                f"device of input ({A.device})"
            )
        if out.shape != A.shape[:-2]:
            raise RuntimeError(
                f"linalg_det: shape of out {tuple(out.shape)} does not match "
                f"expected shape {tuple(A.shape[:-2])}"
            )
        out.copy_(_linalg_det_impl(A))
        return out
    return _linalg_det_impl(A)


def _linalg_det_impl(A):
    if A.dtype != torch.float32:
        raise NotImplementedError(
            f"FlagGems linalg_det on Ascend currently supports float32 only, "
            f"got {A.dtype}"
        )

    if A.dim() < 2:
        raise ValueError(
            f"linalg_det: input tensor must be at least 2D, got {A.dim()}D"
        )

    m, n = A.shape[-2], A.shape[-1]
    if m != n:
        raise ValueError(
            f"linalg_det: input tensor must be a square matrix, got {m}x{n}"
        )

    batch_shape = A.shape[:-2]
    if n == 0:
        return torch.ones(batch_shape, dtype=A.dtype, device=A.device)

    batch_count = math.prod(batch_shape)
    if batch_count == 0:
        return torch.empty(batch_shape, dtype=A.dtype, device=A.device)

    A_work = A.clone(memory_format=torch.contiguous_format).reshape(batch_count, n, n)
    out = torch.empty(batch_count, dtype=A.dtype, device=A.device)

    grid = (batch_count,)
    with torch_device_fn.device(A.device):
        if n <= _DET_BLOCK_MAX:
            _det_register_kernel[grid](
                A_work, out, n, BLOCK_N=max(16, triton.next_power_of_2(n))
            )
        else:
            _det_blocked_kernel[grid](A_work, out, n, BLOCK=64)

    return out.reshape(batch_shape)
