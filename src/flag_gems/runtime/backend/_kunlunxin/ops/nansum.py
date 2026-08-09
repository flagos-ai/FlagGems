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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Kunlunxin(XPU) nansum override.
#
# The generic flag_gems/ops/nansum.py is unusable on XPU: it zeroes NaN with
# `tl.where(val != val, 0.0, val)`, whose floating-point unordered compare
# (setuo) crashes the XPU LLVM backend:
#   LLVM ERROR: Cannot select: ... setcc ... setuo   (make_elf abort)
# and its reduce tiles run up to 32768 *masked* lanes, which are numerically
# unreliable on XPU (a masked 32768-lane tl.sum silently drops lanes for
# 16-bit inputs; verified on device: bf16 M=10000 -> 9984).
#
# This override:
#  * zeroes NaN via integer bit comparison (fp32: (b & 0x7fffffff) >
#    0x7f800000; fp64: (b & 0x7fff_ffff_ffff_ffff) > 0x7ff0_0000_0000_0000),
#    which lowers cleanly on XPU (verified in isolation);
#  * never reduces more than 8192 masked lanes (verified exact for fp16/bf16/
#    fp32 on device: masked 8192-lane single-tile tl.sum with other=0 is
#    exact), and uses fully unmasked 32768-lane tiles elsewhere (the
#    documented safe tl.sum point with buffer_size_limit=2048);
#  * keeps the flat two-stage split-reduce structure and the row-reduce
#    (BLOCK_M=128 / BLOCK_N<=8192, reduce-OUTSIDE) launch used by the
#    kunlunxin sum override, and uses the native `_copy_from` engine for the
#    strided transpose needed on non-last dims (gems never registers
#    `_copy_from`; same pattern as the sum_dim fix).
# ---------------------------------------------------------------------------

_BLOCK_1D = 8192   # max masked-lane reduce tile (verified exact on XPU)
_CHUNK = 32768      # unmasked tl.sum safe point (buffer_size_limit=2048)
_BLOCK_M = 128
_BLOCK_N_MAX = 8192
_SMALL_M = 4096
_HUGE_N = 32768
_SMALL_BLOCK_M = 8


@libentry()
@triton.jit
def nansum_masked1d_kernel(inp, out, N, BLOCK_SIZE: tl.constexpr):
    # Single static masked tile, BLOCK_SIZE <= 8192, reduce EXACT on XPU
    # (verified for fp16/bf16/fp32; NaN zeroed by int bit compare).
    if tl.constexpr(
        (inp.dtype.element_ty == tl.float64)
        or (inp.dtype.element_ty == tl.int64)
    ):
        cdtype = tl.float64
    else:
        cdtype = tl.float32

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    val = tl.load(inp + offs, mask=mask, other=0.0).to(cdtype)
    if tl.constexpr(
        (inp.dtype.element_ty == tl.float64)
        or (inp.dtype.element_ty == tl.int64)
    ):
        b = val.to(tl.int64, bitcast=True)
        is_nan = (b & 0x7FFFFFFFFFFFFFFF) > 0x7FF0000000000000
    else:
        b = val.to(tl.int32, bitcast=True)
        is_nan = (b & 0x7FFFFFFF) > 0x7F800000
    val = tl.where(is_nan, 0.0, val)
    tl.store(out, tl.sum(val))


@libentry()
@triton.jit
def nansum_chunk_kernel(inp, partial, BLOCK_SIZE: tl.constexpr):
    # Unmasked chunk reduce (BLOCK_SIZE == 32768, safe tl.sum point with
    # buffer_size_limit=2048); each program reduces BLOCK_SIZE consecutive
    # elements and stores one partial.
    if tl.constexpr(
        (inp.dtype.element_ty == tl.float64)
        or (inp.dtype.element_ty == tl.int64)
    ):
        cdtype = tl.float64
    else:
        cdtype = tl.float32

    pid = ext.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    val = tl.load(inp + offs).to(cdtype)
    if tl.constexpr(
        (inp.dtype.element_ty == tl.float64)
        or (inp.dtype.element_ty == tl.int64)
    ):
        b = val.to(tl.int64, bitcast=True)
        is_nan = (b & 0x7FFFFFFFFFFFFFFF) > 0x7FF0000000000000
    else:
        b = val.to(tl.int32, bitcast=True)
        is_nan = (b & 0x7FFFFFFF) > 0x7F800000
    val = tl.where(is_nan, 0.0, val)
    tl.store(partial + pid, tl.sum(val))


@libentry()
@triton.jit
def nansum_rows_kernel(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Row-reduce with reduce-OUTSIDE (elementwise accumulate into a persisted
    # [BLOCK_M, BLOCK_N] tile, single tl.sum after the loop) — exact for all N
    # on XPU; full 8192-lane column blocks unmasked, the < 8192 tail masked.
    if tl.constexpr(
        (inp.dtype.element_ty == tl.float64)
        or (inp.dtype.element_ty == tl.int64)
    ):
        cdtype = tl.float64
    else:
        cdtype = tl.float32

    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows * N
    out = out + rows
    row_mask = rows < M

    _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=cdtype)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        mask = row_mask and (cols < N)
        val = tl.load(inp + cols, mask, other=0.0).to(cdtype)
        if tl.constexpr(
            (inp.dtype.element_ty == tl.float64)
            or (inp.dtype.element_ty == tl.int64)
        ):
            b = val.to(tl.int64, bitcast=True)
            is_nan = (b & 0x7FFFFFFFFFFFFFFF) > 0x7FF0000000000000
        else:
            b = val.to(tl.int32, bitcast=True)
            is_nan = (b & 0x7FFFFFFF) > 0x7F800000
        val = tl.where(is_nan, 0.0, val)
        _sum += val
    tl.store(out, tl.sum(_sum, axis=1)[:, None], row_mask)


def _is_fp64_like(t):
    return t in (torch.float64, torch.int64)


def _launch_nansum_rows(inp, out, M, N):
    block_n = min(triton.next_power_of_2(N), _BLOCK_N_MAX)
    if M <= _SMALL_M and N >= _HUGE_N:
        block_m = _SMALL_BLOCK_M
    else:
        block_m = _BLOCK_M
    grid = (triton.cdiv(M, block_m),)
    with torch_device_fn.device(inp.device):
        nansum_rows_kernel[grid](
            inp, out, M, N, block_m, block_n, buffer_size_limit=2048
        )


def _permute_last(inp, dims):
    """Bring reduced dims to the end via native strided copy (not gems copy_)."""
    ndim = inp.ndim
    keep = [d for d in range(ndim) if d not in dims]
    perm = keep + sorted(dims)
    shape = list(inp.shape)
    N = 1
    for d in dims:
        N *= shape[d]
    M = inp.numel() // N
    view = inp.permute(*perm).reshape(M, N)
    buf = torch.empty(M * N, dtype=inp.dtype, device=inp.device).reshape(M, N)
    torch.ops.aten._copy_from(view, buf, False)
    return buf, M, N


def _nansum_global(inp, out_dtype):
    """Full-tensor (flat) reduction; returns a scalar tensor of out_dtype."""
    M = inp.numel()
    out = torch.empty((), dtype=out_dtype, device=inp.device)
    if M == 0:
        return out

    if M <= _BLOCK_1D:
        # Single masked 8192-lane tile: one launch, exact.
        with torch_device_fn.device(inp.device):
            nansum_masked1d_kernel[(1, 1, 1)](
                inp, out, M, _BLOCK_1D, buffer_size_limit=2048
            )
        return out

    cd = torch.float64 if _is_fp64_like(inp.dtype) else torch.float32
    # Unmasked chunks only: the masked-lane chunk kernel is unreliable with
    # grid > 1 on XPU even for fully-true masks (verified on device:
    # masked 8192-lane chunks at M=16384 give partial[1] ~ 0.13), so every
    # multi-CTA stage uses the unmasked nansum_chunk_kernel, and tails are
    # zero-padded through the native `_copy_from` engine.
    chunk = _BLOCK_1D if M <= 4 * _CHUNK else _CHUNK
    full = M // chunk
    tail = M - full * chunk
    nb = full + (1 if tail else 0)
    partial = torch.empty(nb, dtype=cd, device=inp.device)
    with torch_device_fn.device(inp.device):
        if full:
            nansum_chunk_kernel[(full, 1, 1)](
                inp, partial, chunk, buffer_size_limit=2048
            )
        if tail:
            tail_buf = torch.zeros(chunk, dtype=cd, device=inp.device)
            flat = inp.reshape(-1)
            torch.ops.aten._copy_from(flat[full * chunk :], tail_buf[:tail], False)
            nansum_chunk_kernel[(1, 1, 1)](
                tail_buf, partial[full:], chunk, buffer_size_limit=2048
            )
        if nb == 1:
            torch.ops.aten._copy_from(partial, out, False)
        elif nb <= _BLOCK_1D:
            nansum_masked1d_kernel[(1, 1, 1)](
                partial, out, nb, triton.next_power_of_2(nb),
                buffer_size_limit=2048,
            )
        else:
            if nb != _CHUNK:
                big = torch.zeros(_CHUNK, dtype=cd, device=inp.device)
                torch.ops.aten._copy_from(partial, big[:nb], False)
                partial = big
            nansum_chunk_kernel[(1, 1, 1)](
                partial, out, _CHUNK, buffer_size_limit=2048
            )
    return out


def _nansum_dim_impl(inp, dim, out_dtype):
    """Single-dim reduction; returns tensor of shape with dim removed."""
    shape = list(inp.shape)
    N = shape[dim]
    ndim = inp.ndim
    out_shape = list(shape)
    out_shape[dim] = 1
    out = torch.empty(out_shape, dtype=out_dtype, device=inp.device)
    if not inp.is_contiguous():
        inp = inp.contiguous()
    if dim == ndim - 1:
        _launch_nansum_rows(inp, out, inp.numel() // N, N)
    else:
        buf, M2, N2 = _permute_last(inp, [dim])
        _launch_nansum_rows(buf, out, M2, N2)
    return out


def nansum(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS_KUNLUNXIN NANSUM")
    if dtype is None:
        if inp.dtype == torch.bool:
            inp = inp.to(torch.int64)
            out_dtype = torch.int64
        else:
            out_dtype = inp.dtype
    else:
        out_dtype = dtype

    if dim is None or dim == []:
        res = _nansum_global(inp, out_dtype)
        if dim == [] and keepdim:
            res = res.reshape([1] * inp.ndim)
        return res

    dims = [dim] if isinstance(dim, int) else list(dim)
    dims = [d % inp.ndim for d in dims]

    if inp.numel() == 0:
        shape = list(inp.shape)
        if keepdim:
            for d in dims:
                shape[d] = 1
        else:
            for d in sorted(dims, reverse=True):
                shape.pop(d)
        return torch.zeros(shape, dtype=out_dtype, device=inp.device)

    if set(dims) == set(range(inp.ndim)):
        res = _nansum_global(inp, out_dtype)
        if keepdim:
            res = res.reshape([1] * inp.ndim)
        return res

    if len(dims) == 1:
        res = _nansum_dim_impl(inp, dims[0], out_dtype)
        if keepdim:
            return res
        return res.squeeze(dim=dims[0])

    # Multi-dim: reduce one dim at a time (descending keeps lower dims valid).
    data = inp.contiguous()
    for d in sorted(dims, reverse=True):
        data = _nansum_dim_impl(data, d, out_dtype).squeeze(d)
    if keepdim:
        target = list(inp.shape)
        for d in dims:
            target[d] = 1
        return data.reshape(target)
    return data


def nansum_out(inp, dim=None, keepdim=False, *, dtype=None, out=None):
    logger.info("GEMS_KUNLUNXIN NANSUM_OUT")
    result = nansum(inp, dim=dim, keepdim=keepdim, dtype=dtype)
    if out.shape != result.shape:
        out.resize_(result.shape)
    torch.ops.aten._copy_from(result, out.reshape(result.shape), False)
    return out