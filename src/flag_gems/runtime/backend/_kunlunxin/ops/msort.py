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

"""Kunlunxin XPU3 ``msort`` override.

``torch.msort`` sorts each column along the first (``dim=0``) axis in ascending
order and returns values only (no indices).  The generic Triton implementation
relies on ``tl.xor_sum``/``tl.sort``/``tl.flip`` for its in-register bitonic
network, all of which fail to compile on this backend (``uni_sram`` /
silent no-op), so the operator is currently completely broken here.

This override provides two *real kernel* paths (no ATen/CPU fallback):

* ``_network_msort`` — a compare-exchange bitonic network expressed purely with
  plane 2D loads/stores and element-wise ``tl.where`` (the only pattern the XPU
  compiler lowers reliably).  One kernel launch per network step with a host
  side ping-pong of scratch buffers, which keeps every kernel free of
  cross-thread read-after-write hazards.  Selected for the benchmark regime
  (float dtypes, power-of-two first dim, wide column count) where it is the
  fastest of the two paths.
* ``_radix_msort`` — the proven backend radix sorter (``sort_stable``) with a
  NaN canonicalisation + index gather so that NaN payloads and the
  NaNs-sort-last contract are preserved.  Used for every other shape/dtype and
  is the universal correct fallback.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.contiguous import contiguous
from flag_gems.ops.copy import copy_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from .sort import sort_stable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Radix fallback
# ---------------------------------------------------------------------------


@libentry()
@triton.jit
def _canonicalize_nan_kernel(inp, out, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    off = tl.minimum(offsets, n_elements - 1)
    values = tl.load(inp + off, mask=mask, other=0)
    values = tl.where(values != values, float("nan"), values)
    tl.store(out + offsets, values, mask=mask)


@libentry()
@triton.jit
def _gather_last_kernel(
    inp, indices, out, n_elements, n_inner, BLOCK_SIZE: tl.constexpr
):
    # ``inp``/``indices``/``out`` are laid out as [n_outer, n_inner] row-major;
    # gather ``out[o, r] = inp[o, indices[o, r]]`` along the inner axis.
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    idx = tl.load(indices + offsets, mask=mask, other=0)
    outer = offsets // n_inner
    src = outer * n_inner + idx
    src = tl.minimum(src, n_elements - 1)
    values = tl.load(inp + src, mask=mask, other=0)
    tl.store(out + offsets, values, mask=mask)


def _radix_msort(inp, out):
    # Sorting along the first axis is the same as sorting the transposed tensor
    # along its (contiguous) last axis.  The backend radix sorter's dim-0
    # permute path is unreliable on this platform, but the last-axis path is
    # exact, so operate on the transposed contiguous layout throughout.
    n_rows = inp.shape[0]
    n_cols = inp.numel() // n_rows
    flat = inp.reshape(n_rows, n_cols)
    xt = flat.transpose(0, 1).contiguous()  # [n_cols, n_rows]
    block = 1024
    grid = (triton.cdiv(xt.numel(), block),)
    if inp.dtype.is_floating_point:
        canonical = torch.empty_like(xt)
        out_t = torch.empty_like(xt)
        with torch_device_fn.device(inp.device):
            _canonicalize_nan_kernel[grid](
                xt, canonical, xt.numel(), BLOCK_SIZE=block
            )
            _, indices = sort_stable(canonical, stable=False, dim=-1, descending=False)
            indices = contiguous(indices)
            _gather_last_kernel[grid](
                xt, indices, out_t, xt.numel(), n_rows, BLOCK_SIZE=block
            )
        result = out_t.transpose(0, 1).contiguous()
    else:
        with torch_device_fn.device(inp.device):
            values, _ = sort_stable(xt, stable=False, dim=-1, descending=False)
        result = values.transpose(0, 1).contiguous()
    copy_(out, result.reshape(out.shape))


# ---------------------------------------------------------------------------
# Bitonic network (float only, plane 2D loads/stores + element-wise min/max)
# ---------------------------------------------------------------------------


@triton.jit
def _mono_key(v, DT: tl.constexpr):
    if DT == 1:  # float32
        bits = v.to(tl.uint32, bitcast=True)
        return tl.where(
            (bits & 0x80000000) != 0, (~bits) & 0xFFFFFFFF, bits ^ 0x80000000
        ).to(tl.int64)
    elif DT == 2:  # float16
        bits = v.to(tl.uint16, bitcast=True).to(tl.uint32)
        return tl.where((bits & 0x8000) != 0, (~bits) & 0xFFFF, bits ^ 0x8000).to(
            tl.int64
        )
    else:  # bfloat16 via exact widening to float32
        bits = v.to(tl.float32).to(tl.uint32, bitcast=True)
        return tl.where(
            (bits & 0x80000000) != 0, (~bits) & 0xFFFFFFFF, bits ^ 0x80000000
        ).to(tl.int64)


@libentry()
@triton.jit
def _seed_kernel(
    inp,
    k0,
    v0,
    n_rows,
    n_cols,
    cpad,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    DT: tl.constexpr,
):
    col = tl.program_id(0) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    cmask = col < n_cols
    rows = tl.arange(0, BLOCK_ROWS)
    numel = n_rows * n_cols
    off_in = rows[:, None] * n_cols + col[None, :]
    off_ld = tl.minimum(off_in, numel - 1)
    m = (rows[:, None] < n_rows) & cmask[None, :]
    vals = tl.load(inp + off_ld, mask=m, other=0)
    if DT == 2:
        NAN_KEY: tl.constexpr = 0xFFFE
    else:
        NAN_KEY: tl.constexpr = 0xFFFFFFFE
    mono = _mono_key(vals, DT)
    nank = tl.full(mono.shape, value=NAN_KEY, dtype=tl.int64)
    keyed = tl.where(vals != vals, nank, mono)
    padk = tl.full(mono.shape, value=9223372036854775807, dtype=tl.int64)
    keys = tl.where(m, keyed, padk)
    boff = rows[:, None] * cpad + col[None, :]
    tl.store(k0 + boff, keys)
    tl.store(v0 + boff, vals)


@libentry()
@triton.jit
def _step_kernel(
    src_k,
    src_v,
    dst_k,
    dst_v,
    cpad,
    STAGE: tl.constexpr,
    STEP: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    col = tl.program_id(0) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    half: tl.constexpr = BLOCK_ROWS // 2
    bit: tl.constexpr = 1 << STEP
    j = tl.arange(0, half)
    a_rows = ((j >> STEP) << (STEP + 1)) | (j & (bit - 1))
    b_rows = a_rows | bit
    off_a = a_rows[:, None] * cpad + col[None, :]
    off_b = b_rows[:, None] * cpad + col[None, :]
    ka = tl.load(src_k + off_a)
    kb = tl.load(src_k + off_b)
    va = tl.load(src_v + off_a)
    vb = tl.load(src_v + off_b)
    take_min = ((a_rows & (1 << STAGE)) == 0)[:, None]
    cond = tl.where(take_min, ka > kb, ka < kb)
    tl.store(dst_k + off_a, tl.where(cond, kb, ka))
    tl.store(dst_k + off_b, tl.where(cond, ka, kb))
    tl.store(dst_v + off_a, tl.where(cond, vb, va))
    tl.store(dst_v + off_b, tl.where(cond, va, vb))


@libentry()
@triton.jit
def _final_kernel(
    cur_v,
    out,
    n_rows,
    n_cols,
    cpad,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    col = tl.program_id(0) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    cmask = col < n_cols
    rows = tl.arange(0, BLOCK_ROWS)
    numel = n_rows * n_cols
    off_in = rows[:, None] * n_cols + col[None, :]
    off_st = tl.minimum(off_in, numel - 1)
    m = (rows[:, None] < n_rows) & cmask[None, :]
    boff = rows[:, None] * cpad + col[None, :]
    fv = tl.load(cur_v + boff)
    tl.store(out + off_st, fv, mask=m)


_NETWORK_DT = {
    torch.float32: 1,
    torch.float16: 2,
    torch.bfloat16: 3,
}


def _network_eligible(inp, n_rows, n_cols):
    if inp.dtype not in _NETWORK_DT:
        return False
    if n_rows & (n_rows - 1) != 0 or n_rows > 512:
        return False
    # BLOCK_COLS < 64 hits the store-spill path; keep the column tile wide.
    return n_cols >= 64


def _network_msort(inp, out):
    n_rows = inp.shape[0]
    n_cols = inp.numel() // n_rows
    block_rows = n_rows
    block_cols = 64
    grid = (triton.cdiv(n_cols, block_cols),)
    cpad = grid[0] * block_cols
    dev = inp.device
    k0 = torch.empty(block_rows * cpad, device=dev, dtype=torch.int64)
    k1 = torch.empty(block_rows * cpad, device=dev, dtype=torch.int64)
    v0 = torch.empty(block_rows * cpad, device=dev, dtype=inp.dtype)
    v1 = torch.empty(block_rows * cpad, device=dev, dtype=inp.dtype)
    dt = _NETWORK_DT[inp.dtype]
    with torch_device_fn.device(dev):
        _seed_kernel[grid](
            inp, k0, v0, n_rows, n_cols, cpad,
            BLOCK_ROWS=block_rows, BLOCK_COLS=block_cols, DT=dt,
        )
        cur = (k0, v0)
        dst = (k1, v1)
        log_rows = block_rows.bit_length() - 1
        for stage in range(1, log_rows + 1):
            for step in range(stage - 1, -1, -1):
                _step_kernel[grid](
                    cur[0], cur[1], dst[0], dst[1], cpad,
                    STAGE=stage, STEP=step,
                    BLOCK_ROWS=block_rows, BLOCK_COLS=block_cols,
                )
                cur, dst = dst, cur
        _final_kernel[grid](
            cur[1], out, n_rows, n_cols, cpad,
            BLOCK_ROWS=block_rows, BLOCK_COLS=block_cols,
        )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def _msort_contiguous(inp, out):
    if inp.ndim == 0:
        copy_(out, inp)
        return
    n_rows = inp.shape[0]
    if inp.numel() == 0 or n_rows == 1:
        copy_(out, inp)
        return
    n_cols = inp.numel() // n_rows
    if _network_eligible(inp, n_rows, n_cols):
        _network_msort(inp.reshape(n_rows, n_cols), out.reshape(n_rows, n_cols))
    else:
        _radix_msort(inp.reshape(n_rows, n_cols), out.reshape(n_rows, n_cols))


def msort(inp):
    logger.debug("GEMS_KUNLUNXIN MSORT")
    if inp.is_complex():
        raise RuntimeError('"msort" not implemented for complex dtypes')
    out = torch.empty_like(inp, memory_format=torch.preserve_format)
    if inp.numel() == 0:
        return out
    if inp.is_contiguous() and out.is_contiguous():
        _msort_contiguous(inp, out)
    else:
        contiguous_inp = contiguous(inp)
        contiguous_out = torch.empty_like(contiguous_inp)
        _msort_contiguous(contiguous_inp, contiguous_out)
        copy_(out, contiguous_out)
    return out


def msort_out(inp, *, out):
    """``aten::msort.out`` — reuse the retained ``msort`` kernels, write through.

    The generic ``msort_out`` routes through the generic ``msort_kernel`` which
    cannot compile on XPU3 (``uni_sram``), so ``msort_out`` fails even though the
    Kunlunxin ``msort`` itself is wired.  This variant applies the ``out=``
    contract (dtype/device validation, resize, alias safety, non-contiguous
    output) on top of the already-retained ``_msort_contiguous`` kernels.
    """
    logger.debug("GEMS_KUNLUNXIN MSORT.OUT")
    if inp.is_complex():
        raise RuntimeError('"msort" not implemented for complex dtypes')
    if out.dtype != inp.dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {inp.dtype}, but got {out.dtype} instead"
        )
    if out.device != inp.device:
        raise RuntimeError(
            f"Expected out tensor to have device {inp.device}, but got {out.device} instead"
        )
    if (
        inp.numel()
        and inp.untyped_storage().data_ptr() == out.untyped_storage().data_ptr()
    ):
        # Sort in place would expose partially sorted data to the compare-exchange
        # network; snapshot the source before any resize/write-through.
        source = torch.empty(inp.shape, dtype=inp.dtype, device=inp.device)
        copy_(source, inp)
        inp = source
    if out.shape != inp.shape:
        out.resize_(inp.shape)
    if inp.numel() == 0:
        return out
    if inp.is_contiguous() and out.is_contiguous():
        _msort_contiguous(inp, out)
    else:
        contiguous_inp = contiguous(inp)
        contiguous_out = torch.empty_like(contiguous_inp)
        _msort_contiguous(contiguous_inp, contiguous_out)
        copy_(out, contiguous_out)
    return out
