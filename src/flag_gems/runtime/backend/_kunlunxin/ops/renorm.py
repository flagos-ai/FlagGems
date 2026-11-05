# Copyright 2026 FlagOS Contributors
#
# Kunlunxin (XPU) override of renorm / renorm_.
#
# Root cause: the generic kernels (flag_gems/ops/renorm.py, renorm_.py) compute
# the per-row p-norm via `tl_extra_shim.pow(...)` / `tl_extra_shim.sqrt(...)`.
# On XPU these lower to `ld.lld: error: undefined symbol: Unsupported` at link
# time (same failure family as tl_extra_shim.lgamma) -> all 270 cases crash.
#
# Fix: express the power with `exp`/`log` intrinsics that do lower on XPU:
#   |x|^p        = exp(p * log(|x|))     (0 handled explicitly)
#   sum^(1/p)    = exp(log(sum) / p)
# One program per row: pass 1 accumulates sum(|x|^p), pass 2 rescales.
#
# Performance fix (2026-08-12, XPU 4): for dim != 0 the previous wrapper turned
# the permuted sub-tensor side with `.contiguous()` (both for the input and for
# the final output), which inside the gems context dispatches to the slow gems
# `copy_` override (~150us per transposed copy on XPU). Replace both transposed
# copies with a native strided copy via `torch.ops.aten._copy_from` (gems never
# overrides `_copy_from`) on `torch.empty_strided` buffers (gems never
# overrides `empty_strided` either) - the same native-copy key used by
# slice_backward / resize / pad / pixel_unshuffle.
#
# Performance fix (2026-08-29, XPU 5, `renorm_` only): the row kernel was
# launched with BLOCK_SIZE = min(next_pow2(N), 1024), i.e. a tile exactly as
# wide as one sub-tensor. Measured on XPU 5, the per-program cost of this
# kernel is governed almost entirely by the *tile byte width*, not by the
# amount of live data:
#   tile 256 B -> ~0.83 us/program   tile 512 B -> ~0.34 us/program
#   tile >= 1024 B -> ~0.11 us/program
# so short sub-tensors (N <= 128 for fp32, N <= 256 for fp16/bf16) were paying
# 3-8x the necessary cost. Widening the (masked) tile to at least 1024 bytes
# removes that penalty. The widened tail lanes of the *last* program would
# address past the end of the buffer, so BLOCK is only inflated for buffers
# this file allocates itself, and those are over-allocated by _TILE_PAD
# elements. In addition, for N <= BLOCK the whole sub-tensor fits in one tile,
# so it is loaded once and kept in registers (the old kernel read X twice), and
# for p == 2 the |x|^p / sum^(1/p) pair becomes x*x / sqrt (no exp/log).
# `renorm` (out-of-place) deliberately keeps the original launch path.
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Minimum tile width in bytes; below this the XPU per-program cost explodes.
_MIN_TILE_BYTES = 1024
# Slack (in elements) appended to buffers this file allocates, so the masked
# tail lanes of the last program can never address outside the allocation.
_TILE_PAD = 1024


@libentry()
@triton.jit(do_not_specialize=["p", "maxnorm"])
def renorm_row_kernel(X, Y, N, p, maxnorm, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    base = pid * N

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + base + cols, mask=mask, other=0.0).to(tl.float32)
        ax = tl.abs(x)
        powered = tl.where(ax == 0.0, 0.0, tl.exp(p * tl.log(ax)))
        acc += tl.where(mask, powered, 0.0)

    s = tl.sum(acc)
    norm = tl.where(s == 0.0, 0.0, tl.exp(tl.log(s) / p))
    scale = tl.where(norm > maxnorm, maxnorm / norm, 1.0)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + base + cols, mask=mask, other=0.0).to(tl.float32)
        y = x * scale
        tl.store(Y + base + cols, y.to(X.dtype.element_ty), mask=mask)


def _launch(x_flat, out_flat, M, N, p, maxnorm):
    BLOCK = min(triton.next_power_of_2(N), 1024)
    grid = (M,)
    with torch_device_fn.device(x_flat.device):
        renorm_row_kernel[grid](
            x_flat,
            out_flat,
            N,
            float(p),
            float(maxnorm),
            BLOCK_SIZE=BLOCK,
            buffer_size_limit=2048,
            isCloseVectorization=True,
        )


@libentry()
@triton.jit(do_not_specialize=["p", "maxnorm"])
def renorm_row_tile_kernel(
    X, Y, N, p, maxnorm, BLOCK_SIZE: tl.constexpr, IS_P2: tl.constexpr
):
    """One program per sub-tensor, whole sub-tensor in a single tile.

    Requires N <= BLOCK_SIZE. The row is read once and kept in registers, so
    there is no global store->load inside a program (no tl.debug_barrier
    needed even when X is Y).
    """
    pid = tl.program_id(0).to(tl.int64)
    base = pid * N
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X + base + cols, mask=mask, other=0.0).to(tl.float32)
    if IS_P2:
        powered = x * x
    else:
        ax = tl.abs(x)
        powered = tl.where(ax == 0.0, 0.0, tl.exp(p * tl.log(ax)))
    s = tl.sum(tl.where(mask, powered, 0.0))
    if IS_P2:
        norm = tl.sqrt(s)
    else:
        norm = tl.where(s == 0.0, 0.0, tl.exp(tl.log(s) / p))
    scale = tl.where(norm > maxnorm, maxnorm / norm, 1.0)
    tl.store(Y + base + cols, (x * scale).to(X.dtype.element_ty), mask=mask)


def _row_block(N, itemsize, total, inflate):
    """Tile width for one sub-tensor. Always a power of two (TritonXPU
    mis-lowers non-pow2 constexpr tile widths)."""
    np2 = triton.next_power_of_2(N)
    if np2 >= 1024:
        return 1024
    if not inflate:
        return np2
    block = min(max(np2, _MIN_TILE_BYTES // itemsize), 1024)
    while block > total and block > np2:
        block //= 2
    return block


def _padded_flat(shape, dtype, device):
    """Contiguous buffer of `shape` carved out of an allocation that has
    _TILE_PAD extra elements at the end (gems does not override
    `empty_strided`)."""
    numel = 1
    for s in shape:
        numel *= s
    buf = torch.empty_strided((numel + _TILE_PAD,), (1,), dtype=dtype, device=device)
    return buf[:numel].view(shape)


def _launch_rows(x_flat, out_flat, M, N, p, maxnorm, inflate):
    """`renorm_` launch path: single-tile kernel when the sub-tensor fits,
    otherwise the original multi-tile loop kernel.

    `inflate=True` is only legal when the buffer behind x_flat/out_flat has
    _TILE_PAD slack elements after the payload.
    """
    BLOCK = _row_block(N, x_flat.element_size(), M * N, inflate)
    grid = (M,)
    with torch_device_fn.device(x_flat.device):
        if N <= BLOCK:
            renorm_row_tile_kernel[grid](
                x_flat,
                out_flat,
                N,
                float(p),
                float(maxnorm),
                BLOCK_SIZE=BLOCK,
                IS_P2=(float(p) == 2.0),
                buffer_size_limit=2048,
                isCloseVectorization=True,
            )
        else:
            renorm_row_kernel[grid](
                x_flat,
                out_flat,
                N,
                float(p),
                float(maxnorm),
                BLOCK_SIZE=BLOCK,
                buffer_size_limit=2048,
                isCloseVectorization=True,
            )


def _contiguous_strides(shape):
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return strides


def _native_transposed_copy(src_view, dst):
    """One native strided copy (transpose-capable) bypassing the gems copy_
    override; gems never overrides `_copy_from` so this reaches the vendor's
    native copy engine."""
    torch.ops.aten._copy_from(src_view, dst, False)


def renorm(input, p, dim, maxnorm):
    logger.debug("GEMS_KUNLUNXIN RENORM")
    dim = dim % input.ndim

    if input.numel() == 0:
        return torch.empty_strided(
            input.shape,
            _contiguous_strides(input.shape),
            dtype=input.dtype,
            device=input.device,
        )

    if dim == 0:
        # Sub-tensors are the rows of the input itself: no transposition.
        if input.is_contiguous():
            x_flat = input
        else:
            x_flat = torch.empty_strided(
                input.shape,
                _contiguous_strides(input.shape),
                dtype=input.dtype,
                device=input.device,
            )
            _native_transposed_copy(input, x_flat)
        M = x_flat.shape[0]
        N = x_flat[0].numel()
        out_flat = torch.empty_like(x_flat)
        _launch(x_flat, out_flat, M, N, p, maxnorm)
        return out_flat

    perm = [dim] + [i for i in range(input.ndim) if i != dim]
    inv_perm = [0] * input.ndim
    for i, pi in enumerate(perm):
        inv_perm[pi] = i

    x_perm = input.permute(perm)  # strided view
    x_shape = x_perm.shape
    x_flat = torch.empty_strided(
        x_shape,
        _contiguous_strides(x_shape),
        dtype=input.dtype,
        device=input.device,
    )
    _native_transposed_copy(x_perm, x_flat)

    M = x_shape[0]
    N = x_perm[0].numel()
    out_flat = torch.empty_like(x_flat)
    _launch(x_flat, out_flat, M, N, p, maxnorm)

    out = torch.empty_strided(
        input.shape,
        _contiguous_strides(input.shape),
        dtype=input.dtype,
        device=input.device,
    )
    _native_transposed_copy(out_flat.reshape(x_shape).permute(inv_perm), out)
    return out


def renorm_(input, p, dim, maxnorm):
    logger.debug("GEMS_KUNLUNXIN RENORM_")
    dim = dim % input.ndim

    if input.numel() == 0:
        return input

    if dim == 0:
        if input.is_contiguous():
            # In place on the caller's storage: there is no slack behind the
            # payload, so the tile must not be inflated past next_pow2(N).
            M = input.shape[0]
            N = input[0].numel()
            _launch_rows(input, input, M, N, p, maxnorm, inflate=False)
            return input
        x_flat = _padded_flat(input.shape, input.dtype, input.device)
        _native_transposed_copy(input, x_flat)
        M = x_flat.shape[0]
        N = x_flat[0].numel()
        _launch_rows(x_flat, x_flat, M, N, p, maxnorm, inflate=True)
        _native_transposed_copy(x_flat, input)
        return input

    perm = [dim] + [i for i in range(input.ndim) if i != dim]
    inv_perm = [0] * input.ndim
    for i, pi in enumerate(perm):
        inv_perm[pi] = i

    x_perm = input.permute(perm)  # strided view
    x_shape = x_perm.shape
    x_flat = _padded_flat(x_shape, input.dtype, input.device)
    _native_transposed_copy(x_perm, x_flat)

    M = x_shape[0]
    N = x_perm[0].numel()
    _launch_rows(x_flat, x_flat, M, N, p, maxnorm, inflate=True)

    _native_transposed_copy(x_flat.reshape(x_shape).permute(inv_perm), input)
    return input
