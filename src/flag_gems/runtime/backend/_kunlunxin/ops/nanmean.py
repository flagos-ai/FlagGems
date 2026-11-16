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

"""Kunlunxin XPU3 override for ``aten::nanmean`` / ``aten::nanmean.out``.

Generic ``src/flag_gems/ops/nanmean.py`` routes any single-axis reduction that
is *not* the innermost axis through ``nanmean_dim_non_inner_kernel``.  That
kernel uses a ``[BLOCK_N, BLOCK_K]`` 2-D tile and ``tl.sum(axis=0)``; on XPU3 it
is unusable:

* silent wrong values for the 2-D tile path — e.g. ``(4096, 256)`` reduced over
  ``dim=0`` mismatches 13% of the output for every float dtype;
* compile failures (``triton_xpu.reduce`` type inference / ``TritonXPU
  UnrollControl`` / ``ConvertTritonXPUToLLVM``) for the wider tiles, e.g.
  ``(200, 40999, 3)`` over ``dim=0`` or ``dim=1``.

Both are the known ``pattern-xpu3-2d-tile-silent-miscompile`` family.  The
generic *global* path and the *inner-dim* path (``K == 1``) are correct on XPU
and are kept.

This override keeps the generic host-side contract (validation, dtype
inference, keepdim/squeeze, empty / zero-sized handling, complex path,
autograd) and replaces the reduction kernel with the proven row-reduce shape
used by the Kunlunxin ``mean`` override: reduced dims are compressed to the
trailing axis with ``dim_compress`` (identity when the reduced axis is already
last), then one program reduces one row with a 1-D lane tile, a persisted
``[BLOCK_M, BLOCK_N]`` accumulator and a single ``tl.sum(axis=1)`` *outside* the
loop (the in-loop axis-1 reduce miscompiles for fp16/bf16).  NaN is detected
with an integer bit compare instead of the unordered float compare that crashes
the XPU LLVM backend (see the ``nansum`` override).
"""

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.nanmean import (  # noqa: E402
    _NanmeanAutograd as _gen_autograd,
    _complex_mean,
    _masked_input_parts,
    _nanmean_global,
    _normalize_dims,
    _squeeze_dims,
)
from flag_gems.ops.nanmean import copy_ as _gen_copy
from flag_gems.ops.nanmean import sum_dim as _gen_sum_dim
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

_TILE_BUDGET = 32768


def _block_n(N, dt):
    if N > 8192:
        return min(triton.next_power_of_2(N), 2048)
    if dt == torch.float32:
        return min(triton.next_power_of_2(N), 512)
    return min(triton.next_power_of_2(N), 256)


def _block_m(M, wide):
    # A 64-bit accumulator halves the number of lanes that fit the tile budget.
    cap = 32 if wide else 128
    return min(triton.next_power_of_2(M), cap)


@libentry()
@triton.jit
def _nanmean_row_kernel(
    X,
    Out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ACC64: tl.constexpr,
):
    pid = ext.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid.to(tl.int64) * N
    Out = Out + pid
    row_mask = pid < M

    if ACC64:
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float64)
        cnt = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float64)
    else:
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        cnt = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        mask = row_mask & (cols < N)
        a = tl.load(X + cols, mask=mask, other=0.0)
        if ACC64:
            a = a.to(tl.float64)
            b = a.to(tl.int64, bitcast=True)
            is_nan = (b & 0x7FFFFFFFFFFFFFFF) > 0x7FF0000000000000
            cnt += (~is_nan & mask).to(tl.float64)
        else:
            a = a.to(tl.float32)
            b = a.to(tl.int32, bitcast=True)
            is_nan = (b & 0x7FFFFFFF) > 0x7F800000
            cnt += (~is_nan & mask).to(tl.float32)
        acc += tl.where(is_nan, 0.0, a)

    s = tl.sum(acc, axis=1)[:, None]
    c = tl.sum(cnt, axis=1)[:, None]
    res = s / c
    tl.store(Out, res, row_mask)


def _nanmean_dim(inp, dim, keepdim, dtype):
    if dtype is None:
        dtype = inp.dtype

    dims = _normalize_dims(dim, inp.ndim)

    if inp.ndim == 0:
        out = _nanmean_global(inp, dtype=dtype)
        return out.reshape([1] * inp.ndim) if keepdim else out

    # dim=None / dim=[] / full-dimensional reduction -> global nanmean.
    if len(dims) == 0 or len(dims) == inp.ndim:
        out = _nanmean_global(inp, dtype=dtype)
        if keepdim:
            out = out.reshape([1] * inp.ndim)
        return out

    shape = list(inp.shape)
    N = 1
    for d in dims:
        N *= shape[d]
        shape[d] = 1

    if N == 0:
        out = torch.full(shape, float("nan"), dtype=dtype, device=inp.device)
        return out if keepdim else _squeeze_dims(out, dims)

    if math.prod(shape) == 0:
        out = torch.empty(shape, dtype=dtype, device=inp.device)
        return out if keepdim else _squeeze_dims(out, dims)

    # Move the reduced dims to the trailing axis (identity + no copy when the
    # reduced axis is already last).  Row-major over the batch dims then maps
    # 1:1 onto the flat output.
    x = dim_compress(inp, dims)
    M = x.numel() // N

    out_shape = [1 if i in dims else s for i, s in enumerate(inp.shape)]
    if not keepdim:
        out_shape = [s for i, s in enumerate(out_shape) if i not in dims]

    if M == 0:
        return torch.empty(out_shape, dtype=dtype, device=inp.device)

    out = torch.empty(out_shape, dtype=dtype, device=inp.device)
    acc64 = dtype == torch.float64
    block_n = _block_n(N, dtype)
    block_m = max(min(_block_m(M, acc64), _TILE_BUDGET // block_n), 1)
    grid = (triton.cdiv(M, block_m),)
    with torch_device_fn.device(inp.device):
        _nanmean_row_kernel[grid](
            x,
            out,
            M,
            N,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            ACC64=acc64,
            buffer_size_limit=2048,
        )
    return out


class _NanmeanAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, dim, keepdim, dtype):
        dtype = dtype or inp.dtype
        dims = _normalize_dims(dim, inp.ndim)
        if not dims:
            dims = list(range(inp.ndim))
        ctx.dims = dims
        ctx.keepdim = keepdim
        ctx.input_dtype = inp.dtype
        _, _, valid = _masked_input_parts(inp, dtype)
        count = _gen_sum_dim(valid, dim=dims, keepdim=True)
        ctx.save_for_backward(valid, count)
        with torch.no_grad():
            if inp.is_complex() or dtype.is_complex:
                return _complex_mean(inp.detach(), dim, keepdim, dtype)
            return _nanmean_impl(inp.detach(), dim, keepdim, dtype)

    @staticmethod
    def backward(ctx, grad):
        return _gen_autograd.backward(ctx, grad)


def _nanmean_impl(inp, dim, keepdim, dtype):
    if inp.is_complex() or (dtype is not None and dtype.is_complex):
        return _complex_mean(inp, dim, keepdim, dtype or inp.dtype)
    if dim is None:
        result = _nanmean_global(inp, dtype=dtype)
        if keepdim:
            result = result.reshape([1] * inp.ndim)
        return result
    return _nanmean_dim(inp, dim, keepdim, dtype)


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
    return _nanmean_impl(inp, dim, keepdim, dtype)


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
        out_parts = torch.view_as_real(out)
        result_parts = torch.view_as_real(result)
        _gen_copy(out_parts, result_parts)
    else:
        _gen_copy(out, result)
    return out
