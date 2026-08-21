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
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


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
        _launch(x_flat, x_flat, M, N, p, maxnorm)
        if x_flat is not input:
            _native_transposed_copy(x_flat, input)
        return input

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
    _launch(x_flat, x_flat, M, N, p, maxnorm)

    _native_transposed_copy(x_flat.reshape(x_shape).permute(inv_perm), input)
    return input
