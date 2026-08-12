import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device as runtime_device
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
    N,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)
    rows = tl.arange(0, BLOCK_N)
    cols = tl.arange(0, BLOCK_N)

    offsets = pid * N * N + rows[:, None] * N + cols[None, :]
    load_mask = (rows[:, None] < N) & (cols[None, :] < N)
    work = tl.load(A + offsets, mask=load_mask, other=0.0)

    swap_count = tl.zeros((), dtype=tl.int32)

    for k in range(N):
        col_k = tl.sum(tl.where(cols[None, :] == k, work, 0.0), axis=1)
        abs_col = tl.abs(col_k)
        abs_col = tl.where((rows < k) | (rows >= N), -1.0, abs_col)
        pivot_val = tl.max(abs_col, axis=0)
        pivot_row = tl.min(tl.where(abs_col == pivot_val, rows, BLOCK_N), axis=0)

        row_k = tl.sum(tl.where(rows[:, None] == k, work, 0.0), axis=0)
        row_p = tl.sum(tl.where(rows[:, None] == pivot_row, work, 0.0), axis=0)
        work = tl.where(rows[:, None] == k, row_p[None, :], work)
        work = tl.where(rows[:, None] == pivot_row, row_k[None, :], work)
        swap_count = tl.where(pivot_row != k, swap_count + 1, swap_count)

        col_k = tl.sum(tl.where(cols[None, :] == k, work, 0.0), axis=1)
        pivot = tl.sum(tl.where(rows == k, col_k, 0.0), axis=0)

        safe_pivot = tl.where(pivot == 0.0, 1.0, pivot)
        multipliers = tl.where(rows > k, col_k / safe_pivot, 0.0)
        u_row = row_p
        update_mask = (rows[:, None] > k) & (cols[None, :] > k)
        work = tl.where(update_mask, work - multipliers[:, None] * u_row[None, :], work)

    diag = tl.sum(tl.where(rows[:, None] == cols[None, :], work, 0.0), axis=0)
    diag = tl.where(cols < N, diag, 1.0)
    det = tl.reduce(diag, 0, combine_fn=_reduce_mul)
    det = tl.where(swap_count % 2 == 0, det, -det)
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


@libentry()
@triton.jit
def _det_panel_kernel(
    A,
    out,
    N,
    PANEL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tle.program_id(0)
    base = pid * N * N
    swap_count = tl.zeros((), dtype=tl.int32)

    for k0 in range(0, N, PANEL):
        kend = tl.minimum(k0 + PANEL, N)

        for k in range(k0, kend):
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

            for j0 in range(k0, N, BLOCK):
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

            pcols = k + 1 + tl.arange(0, PANEL)
            pcmask = pcols < kend
            u_row = tl.load(A + base + k * N + pcols, mask=pcmask, other=0.0)
            for i0 in range(k + 1, N, BLOCK):
                rows = i0 + tl.arange(0, BLOCK)
                rmask = rows < N
                l_col = tl.load(A + base + rows * N + k, mask=rmask, other=0.0)
                tmask = rmask[:, None] & pcmask[None, :]
                tile = tl.load(
                    A + base + rows[:, None] * N + pcols[None, :],
                    mask=tmask,
                    other=0.0,
                )
                tile = tile - l_col[:, None] * u_row[None, :]
                tl.store(
                    A + base + rows[:, None] * N + pcols[None, :], tile, mask=tmask
                )

            tl.debug_barrier()

        for c in range(k0, kend):
            prows = c + 1 + tl.arange(0, PANEL)
            prmask = prows < kend
            l_strip = tl.load(A + base + prows * N + c, mask=prmask, other=0.0)
            for j0 in range(kend, N, BLOCK):
                cols = j0 + tl.arange(0, BLOCK)
                cmask = cols < N
                u_strip = tl.load(A + base + c * N + cols, mask=cmask, other=0.0)
                tmask = prmask[:, None] & cmask[None, :]
                tile = tl.load(
                    A + base + prows[:, None] * N + cols[None, :],
                    mask=tmask,
                    other=0.0,
                )
                tile = tile - l_strip[:, None] * u_strip[None, :]
                tl.store(
                    A + base + prows[:, None] * N + cols[None, :], tile, mask=tmask
                )

            tl.debug_barrier()

        pcols = k0 + tl.arange(0, PANEL)
        pmask = pcols < kend
        for i0 in range(kend, N, BLOCK):
            rows = i0 + tl.arange(0, BLOCK)
            rmask = rows < N
            l_tile = tl.load(
                A + base + rows[:, None] * N + pcols[None, :],
                mask=rmask[:, None] & pmask[None, :],
                other=0.0,
            )
            for j0 in range(kend, N, BLOCK):
                cols = j0 + tl.arange(0, BLOCK)
                cmask = cols < N
                u_tile = tl.load(
                    A + base + pcols[:, None] * N + cols[None, :],
                    mask=pmask[:, None] & cmask[None, :],
                    other=0.0,
                )
                tmask = rmask[:, None] & cmask[None, :]
                tile = tl.load(
                    A + base + rows[:, None] * N + cols[None, :], mask=tmask, other=0.0
                )
                tile = tile - tl.dot(l_tile, u_tile, input_precision="ieee")
                tl.store(A + base + rows[:, None] * N + cols[None, :], tile, mask=tmask)

        tl.debug_barrier()

    det = tl.full((), 1.0, dtype=A.dtype.element_ty)
    for i0 in range(0, N, BLOCK):
        d = i0 + tl.arange(0, BLOCK)
        diag = tl.load(A + base + d * N + d, mask=d < N, other=1.0)
        det = det * tl.reduce(diag, 0, combine_fn=_reduce_mul)
    det = tl.where(swap_count % 2 == 0, det, -det)
    tl.store(out + pid, det)


def linalg_det(A):
    logger.debug("GEMS LINALG_DET")
    return _linalg_det_impl(A)


def linalg_det_out(A, *, out=None):
    logger.debug("GEMS LINALG_DET_OUT")
    if out is None:
        raise TypeError("linalg_det(): out must be provided for out variant")
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


def _linalg_det_impl(A):
    if A.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"linalg_det only supports float32 and float64, got {A.dtype}")

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

    use_panel = n > _DET_BLOCK_MAX and not (
        A.dtype == torch.float64 and runtime_device.vendor_name == "metax"
    )

    grid = (batch_count,)
    with torch_device_fn.device(A.device):
        if A.dtype == torch.float32 and n <= _DET_BLOCK_MAX:
            block_n = max(16, triton.next_power_of_2(n))
            num_warps = min(8, max(1, (block_n * block_n * 4) // 4096))
            _det_register_kernel[grid](
                A_work, out, n, BLOCK_N=block_n, num_warps=num_warps
            )
        elif use_panel:
            _det_panel_kernel[grid](A_work, out, n, PANEL=32, BLOCK=64, num_warps=4)
        else:
            block = min(64, max(8, triton.next_power_of_2(n)))
            num_warps = min(4, max(1, (block * block * 8) // 4096))
            _det_blocked_kernel[grid](A_work, out, n, BLOCK=block, num_warps=num_warps)

    return out.reshape(batch_shape)
