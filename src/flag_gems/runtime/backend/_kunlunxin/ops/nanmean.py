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

"""Kunlunxin XPU3 override for ``nanmean`` / ``nanmean_out``.

The generic kernels do not build on this backend for the shapes the suite uses:
the 2-D tile that reduces a non-inner axis fails the XPU unroll and lowering
passes, and the inner-axis kernel fails a reduce type check once the tile is
wider than the vector length.  Both are tile-shape problems, not arithmetic
ones.

This override keeps one reduction shape and reaches it by moving the reduced
axes to the end first:

* a full reduction runs a two-stage 1-D reduction (per-block partial sums, then
  a single program that folds them), and
* every other reduction is a 2-D tile ``[BLOCK_M, BLOCK_N]`` reduced along the
  last axis, with ``BLOCK_N`` capped so the reduce stays inside the vector
  length and ``BLOCK_M`` scaled to keep the tile a sensible size.

Sums and counts accumulate in fp32 and are divided once, so an all-NaN slice
yields ``0 / 0`` = NaN like ATen.  Complex inputs are split into real and
imaginary parts, reduced by the same kernels with a single shared count, and
reassembled.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.copy import copy_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# The reduce has to stay inside one vector length, so the inner tile is capped.
_BLOCK_N_MAX = 1024
_BLOCK_M_MAX = 64
_TILE_BUDGET = 8192
_GLOBAL_BLOCK = 4096
_FOLD_BLOCK_MAX = 4096


@libentry()
@triton.jit(do_not_specialize=["M", "N"])
def _nanmean_inner_kernel(inp, out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Mean over the last axis of a row-major ``[M, N]`` tensor."""
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    row_mask = rows < M
    sum_acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    count_acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for start in range(0, N, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)[None, :]
        mask = row_mask & (cols < N)
        val = tl.load(inp + rows * N + cols, mask=mask, other=0.0).to(tl.float32)
        valid = mask & (val == val)
        sum_acc += tl.where(valid, val, 0.0)
        count_acc += valid.to(tl.float32)

    result = tl.sum(sum_acc, axis=1, dtype=tl.float32) / tl.sum(
        count_acc, axis=1, dtype=tl.float32
    )
    tl.store(out + rows, result[:, None], mask=row_mask)


@libentry()
@triton.jit(do_not_specialize=["M", "N"])
def _nanmean_parts_kernel(
    real_ptr,
    imag_ptr,
    real_out,
    imag_out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Mean of the real and imaginary parts over the last axis, one shared count."""
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    row_mask = rows < M
    real_acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    imag_acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    count_acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for start in range(0, N, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)[None, :]
        mask = row_mask & (cols < N)
        offset = rows * N + cols
        r = tl.load(real_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        i = tl.load(imag_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        valid = mask & (r == r) & (i == i)
        real_acc += tl.where(valid, r, 0.0)
        imag_acc += tl.where(valid, i, 0.0)
        count_acc += valid.to(tl.float32)

    count = tl.sum(count_acc, axis=1, dtype=tl.float32)
    tl.store(
        real_out + rows,
        (tl.sum(real_acc, axis=1, dtype=tl.float32) / count)[:, None],
        mask=row_mask,
    )
    tl.store(
        imag_out + rows,
        (tl.sum(imag_acc, axis=1, dtype=tl.float32) / count)[:, None],
        mask=row_mask,
    )


@libentry()
@triton.jit(do_not_specialize=["M"])
def _nanmean_block_kernel(inp, mid, M, mid_size, BLOCK: tl.constexpr):
    """Per-block partial sum and valid count of a flat tensor."""
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < M
    val = tl.load(inp + idx, mask=mask, other=0.0).to(tl.float32)
    valid = mask & (val == val)
    tl.store(mid + pid, tl.sum(tl.where(valid, val, 0.0), axis=0, dtype=tl.float32))
    tl.store(
        mid + mid_size + pid, tl.sum(valid.to(tl.float32), axis=0, dtype=tl.float32)
    )


@libentry()
@triton.jit(do_not_specialize=["mid_size"])
def _nanmean_fold_kernel(mid, out, mid_size, BLOCK: tl.constexpr):
    """Fold the per-block partials into the final scalar."""
    sum_acc = tl.zeros((), dtype=tl.float32)
    count_acc = tl.zeros((), dtype=tl.float32)
    for start in range(0, mid_size, BLOCK):
        idx = start + tl.arange(0, BLOCK)
        mask = idx < mid_size
        sum_acc += tl.sum(
            tl.load(mid + idx, mask=mask, other=0.0), axis=0, dtype=tl.float32
        )
        count_acc += tl.sum(
            tl.load(mid + mid_size + idx, mask=mask, other=0.0),
            axis=0,
            dtype=tl.float32,
        )
    tl.store(out, sum_acc / count_acc)


def _normalize_dims(dim, ndim):
    if isinstance(dim, (list, tuple)) and len(dim) == 0:
        return []
    if dim is None:
        return list(range(ndim))
    if isinstance(dim, int):
        dim = [dim]
    dims = []
    for d in dim:
        if ndim == 0:
            if d not in (-1, 0):
                raise IndexError(
                    f"Dimension out of range (expected to be in range of [-1, 0], but got {d})"
                )
            wrapped = 0
        else:
            if d < -ndim or d >= ndim:
                raise IndexError(
                    "Dimension out of range (expected to be in range of "
                    f"[-{ndim}, {ndim - 1}], but got {d})"
                )
            wrapped = d % ndim
        if wrapped in dims:
            raise RuntimeError(
                f"dim {wrapped} appears multiple times in the list of dims"
            )
        dims.append(wrapped)
    return sorted(dims, reverse=True)


def _squeeze_dims(result, dims):
    for d in sorted(dims, reverse=True):
        result = result.squeeze(dim=d)
    return result


def _nanmean_full(inp, dtype):
    """Mean over every element of ``inp``."""
    if inp.numel() == 0:
        return torch.full([], float("nan"), dtype=dtype, device=inp.device)
    inp = inp.contiguous()
    M = inp.numel()
    out = torch.empty([], dtype=dtype, device=inp.device)
    mid_size = triton.cdiv(M, _GLOBAL_BLOCK)
    mid = torch.empty(2 * mid_size, dtype=torch.float32, device=inp.device)
    with torch_device_fn.device(inp.device):
        _nanmean_block_kernel[(mid_size,)](inp, mid, M, mid_size, BLOCK=_GLOBAL_BLOCK)
        # The fold walks the partials in tiles: a single program cannot hold a
        # tile as wide as the whole partial array for a large input.
        _nanmean_fold_kernel[(1,)](
            mid,
            out,
            mid_size,
            BLOCK=min(triton.next_power_of_2(mid_size), _FOLD_BLOCK_MAX),
        )
    return out


def _tile(block_m_max, reduced):
    block_n = min(triton.next_power_of_2(reduced), _BLOCK_N_MAX)
    block_m = max(1, min(block_m_max, _TILE_BUDGET // block_n))
    return block_m, block_n


def _nanmean_real(inp, dim, keepdim, dtype):
    """Mean of a real tensor over ``dim``, accumulated in fp32."""
    dims = _normalize_dims(dim, inp.ndim)

    if inp.ndim == 0 or len(dims) == 0 or len(dims) == inp.ndim:
        result = _nanmean_full(inp, dtype)
        if keepdim and inp.ndim > 0:
            result = result.reshape([1] * inp.ndim)
        return result

    shape = list(inp.shape)
    reduced = 1
    for d in dims:
        reduced *= shape[d]
    out_shape = [1 if d in dims else s for d, s in enumerate(shape)]

    if reduced == 0:
        # An empty reduction has no valid element, so every output is NaN.
        out = torch.full(out_shape, float("nan"), dtype=dtype, device=inp.device)
        return out if keepdim else _squeeze_dims(out, dims)

    # Move the reduced axes to the end so one kernel shape covers every case;
    # the copy is what makes the reduction inner.
    order = [d for d in range(inp.ndim) if d not in dims] + list(reversed(dims))
    moved = inp.permute(order).contiguous()
    M = moved.numel() // reduced
    if M == 0:
        out = torch.empty(out_shape, dtype=dtype, device=inp.device)
        return out if keepdim else _squeeze_dims(out, dims)

    block_m, block_n = _tile(_BLOCK_M_MAX, reduced)
    out = torch.empty(M, dtype=dtype, device=inp.device)
    with torch_device_fn.device(inp.device):
        _nanmean_inner_kernel[(triton.cdiv(M, block_m),)](
            moved, out, M, reduced, BLOCK_M=block_m, BLOCK_N=block_n
        )

    out = out.reshape(out_shape)
    return out if keepdim else _squeeze_dims(out, dims)


def _split_parts(inp):
    """Real and imaginary parts of ``inp`` as real tensors."""
    real_dtype = torch.float32
    real = torch.empty(inp.shape, dtype=real_dtype, device=inp.device)
    imag = torch.empty_like(real)
    if inp.is_complex():
        physical = inp.conj() if inp.is_conj() else inp
        parts = torch.view_as_real(physical)
        copy_(real, parts[..., 0])
        copy_(imag, parts[..., 1])
    else:
        copy_(real, inp)
        imag.zero_()
    return real, imag


def _assemble_complex(real, imag, dtype):
    result = torch.empty(real.shape, device=real.device, dtype=dtype)
    parts = torch.view_as_real(result)
    copy_(parts[..., 0], real)
    copy_(parts[..., 1], imag)
    return result


def _complex_nanmean(inp, dim, keepdim, dtype):
    dims = _normalize_dims(dim, inp.ndim)
    real, imag = _split_parts(inp)
    shape = list(inp.shape)
    reduced = 1
    for d in dims:
        reduced *= shape[d]
    out_shape = [1 if d in dims else s for d, s in enumerate(shape)]

    if reduced == 0 or real.numel() == 0:
        out = torch.full(out_shape, float("nan"), dtype=dtype, device=inp.device)
        out = out if keepdim else _squeeze_dims(out, dims)
        return out, out

    order = [d for d in range(real.ndim) if d not in dims] + list(reversed(dims))
    real_m = real.permute(order).contiguous()
    imag_m = imag.permute(order).contiguous()
    M = real_m.numel() // reduced
    block_m, block_n = _tile(_BLOCK_M_MAX, reduced)
    real_out = torch.empty(M, dtype=torch.float32, device=inp.device)
    imag_out = torch.empty(M, dtype=torch.float32, device=inp.device)
    with torch_device_fn.device(inp.device):
        _nanmean_parts_kernel[(triton.cdiv(M, block_m),)](
            real_m,
            imag_m,
            real_out,
            imag_out,
            M,
            reduced,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
        )
    return _squeeze_pair(
        real_out.reshape(out_shape), imag_out.reshape(out_shape), dims, keepdim
    )


def _squeeze_pair(real, imag, dims, keepdim):
    if keepdim:
        return real, imag
    return _squeeze_dims(real, dims), _squeeze_dims(imag, dims)


def nanmean(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS_KUNLUNXIN NANMEAN")
    if not (inp.is_floating_point() or inp.is_complex()):
        raise NotImplementedError(
            "nanmean(): expected input to have floating point or complex dtype but got "
            f"{inp.dtype}"
        )
    if dtype is not None and not (dtype.is_floating_point or dtype.is_complex):
        raise RuntimeError(
            "nanmean(): could not infer output dtype. Optional dtype must be either "
            f"a floating point or complex dtype. Got: {dtype}"
        )
    if inp.requires_grad and torch.is_grad_enabled():
        return _NanmeanAutograd.apply(inp, dim, keepdim, dtype)

    out_dtype = dtype or inp.dtype
    if inp.is_complex() or (dtype is not None and dtype.is_complex):
        real_mean, imag_mean = _complex_nanmean(inp, dim, keepdim, out_dtype)
        if not out_dtype.is_complex:
            # ATen keeps the real part when a real dtype is requested.
            return real_mean.to(out_dtype)
        if not inp.is_complex():
            imag_mean = torch.zeros_like(imag_mean)
        return _assemble_complex(real_mean, imag_mean, out_dtype)

    return _nanmean_real(inp, dim, keepdim, out_dtype)


def nanmean_out(inp, dim=None, keepdim=False, *, dtype=None, out=None):
    logger.debug("GEMS_KUNLUNXIN NANMEAN_OUT")
    if out is None:
        raise RuntimeError("nanmean(): missing required out tensor")
    if torch.is_grad_enabled() and (inp.requires_grad or out.requires_grad):
        raise RuntimeError(
            "nanmean(): functions with out= arguments don't support automatic differentiation"
        )
    if out.device != inp.device:
        raise RuntimeError(
            "nanmean: expected result tensor to be on the same device as input"
        )
    if dtype is not None and dtype != out.dtype:
        raise RuntimeError(
            "nanmean: provided dtype must match dtype of result. Got "
            f"{out.dtype} and {dtype}."
        )
    result = nanmean(inp, dim=dim, keepdim=keepdim, dtype=dtype or out.dtype)
    if out.shape != result.shape:
        out.resize_(result.shape)
    if out.is_complex():
        copy_(torch.view_as_real(out), torch.view_as_real(result))
    else:
        copy_(out, result)
    return out


class _NanmeanAutograd(torch.autograd.Function):
    """Backward of ``nanmean``: the gradient is spread over the valid elements."""

    @staticmethod
    def forward(ctx, inp, dim, keepdim, dtype):
        dtype = dtype or inp.dtype
        dims = _normalize_dims(dim, inp.ndim)
        if not dims:
            dims = list(range(inp.ndim))
        ctx.dims = dims
        ctx.keepdim = keepdim
        ctx.input_dtype = inp.dtype
        real, imag = _split_parts(inp)
        valid = (real == real) & (imag == imag)
        ctx.save_for_backward(valid)
        return nanmean(inp.detach(), dim, keepdim, dtype=dtype)

    @staticmethod
    def backward(ctx, grad):
        (valid,) = ctx.saved_tensors
        if not ctx.keepdim:
            for dim in sorted(ctx.dims):
                grad = grad.unsqueeze(dim)
        count = valid.sum(dim=ctx.dims, keepdim=True).to(grad.dtype)
        return (grad / count) * valid, None, None, None


__all__ = ["nanmean", "nanmean_out"]
