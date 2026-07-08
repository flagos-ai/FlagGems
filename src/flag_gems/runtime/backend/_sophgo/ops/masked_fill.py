import logging
from numbers import Number

import torch
import triton
import triton.language as tl

from flag_gems.ops.masked_fill import (
    masked_fill as _fallback_masked_fill,
    masked_fill_ as _fallback_masked_fill_,
)
from flag_gems.utils import broadcastable_to, libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit(do_not_specialize=["value"])
def _masked_fill_same_shape_select_kernel(
    inp,
    mask_ptr,
    out,
    n_elements,
    value,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < n_elements

    fill_mask = tl.load(mask_ptr + offsets, mask=valid, other=0).to(tl.int1)
    cur = tl.load(inp + offsets, mask=valid, other=0.0)
    out_val = tl.where(fill_mask, value, cur)
    tl.store(out + offsets, out_val, mask=valid)


@libentry()
@triton.jit(do_not_specialize=["value"])
def _masked_fill_same_shape_kernel(
    inp,
    mask_ptr,
    out,
    n_elements,
    value,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < n_elements

    fill_mask = tl.load(mask_ptr + offsets, mask=valid, other=0).to(tl.int1)
    keep_mask = valid & (not fill_mask)
    cur = tl.load(inp + offsets, mask=keep_mask, other=0.0)
    tl.store(out + offsets, cur, mask=keep_mask)
    tl.store(out + offsets, value, mask=valid & fill_mask)


@libentry()
@triton.jit(do_not_specialize=["value"])
def _masked_fill_same_shape_inplace_kernel(
    inp,
    mask_ptr,
    n_elements,
    value,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < n_elements

    fill_mask = tl.load(mask_ptr + offsets, mask=valid, other=0).to(tl.int1)
    tl.store(inp + offsets, value, mask=valid & fill_mask)


@libentry()
@triton.jit(do_not_specialize=["value"])
def _masked_fill_lastdim_broadcast_kernel(
    inp,
    mask_ptr,
    out,
    total_outer: tl.constexpr,
    inner: tl.constexpr,
    value,
    BLOCK_INNER: tl.constexpr,
):
    outer = tle.program_id(0)
    inner_offsets = tle.program_id(1) * BLOCK_INNER + tl.arange(0, BLOCK_INNER)
    offsets = outer * inner + inner_offsets
    valid = (outer < total_outer) & (inner_offsets < inner)

    fill_mask = tl.load(mask_ptr + outer, mask=outer < total_outer, other=0).to(tl.int1)
    keep_mask = valid & (not fill_mask)
    cur = tl.load(inp + offsets, mask=keep_mask, other=0.0)
    tl.store(out + offsets, cur, mask=keep_mask)
    tl.store(out + offsets, value, mask=valid & fill_mask)


@libentry()
@triton.jit(do_not_specialize=["value"])
def _masked_fill_lastdim_broadcast_inplace_kernel(
    inp,
    mask_ptr,
    total_outer: tl.constexpr,
    inner: tl.constexpr,
    value,
    BLOCK_INNER: tl.constexpr,
):
    outer = tle.program_id(0)
    inner_offsets = tle.program_id(1) * BLOCK_INNER + tl.arange(0, BLOCK_INNER)
    offsets = outer * inner + inner_offsets
    valid = (outer < total_outer) & (inner_offsets < inner)

    fill_mask = tl.load(mask_ptr + outer, mask=outer < total_outer, other=0).to(tl.int1)
    tl.store(inp + offsets, value, mask=valid & fill_mask)


def _is_tpu_tensor(x):
    return isinstance(x, torch.Tensor) and x.device.type in ("tpu", "sophgo")


def _can_use_fast_path(inp, mask, value):
    if not (_is_tpu_tensor(inp) and isinstance(mask, torch.Tensor)):
        return False
    if not (inp.is_contiguous() and mask.is_contiguous()):
        return False
    if not (inp.dtype is torch.float32 and mask.dtype is torch.bool):
        return False
    if torch.is_tensor(value):
        return value.ndim == 0
    return isinstance(value, Number)


def _same_shape_block_size(n_elements):
    if n_elements <= 65536:
        return 256
    return 1024


def _lastdim_broadcast_inner(inp, mask):
    if inp.ndim == 0 or mask.ndim != inp.ndim:
        return None
    if tuple(mask.shape[:-1]) != tuple(inp.shape[:-1]):
        return None
    if mask.shape[-1] != 1 or inp.shape[-1] == 1:
        return None
    return inp.shape[-1]


def _as_scalar_value(value):
    return value.item() if torch.is_tensor(value) else value


def _launch_same_shape(inp, mask, value, inplace):
    n_elements = inp.numel()
    block_size = _same_shape_block_size(n_elements)
    grid = (triton.cdiv(n_elements, block_size),)
    value = float(_as_scalar_value(value))
    if inplace:
        _masked_fill_same_shape_inplace_kernel[grid](
            inp,
            mask,
            n_elements,
            value,
            BLOCK_SIZE=block_size,
        )
        return inp

    out = torch.empty_like(inp)
    _masked_fill_same_shape_kernel[grid](
        inp,
        mask,
        out,
        n_elements,
        value,
        BLOCK_SIZE=block_size,
    )
    return out


def _launch_same_shape_small_select(inp, mask, value):
    n_elements = inp.numel()
    block_size = 256
    grid = (triton.cdiv(n_elements, block_size),)
    out = torch.empty_like(inp)
    _masked_fill_same_shape_select_kernel[grid](
        inp,
        mask,
        out,
        n_elements,
        float(_as_scalar_value(value)),
        BLOCK_SIZE=block_size,
    )
    return out


def _launch_lastdim_broadcast(inp, mask, value, inner, inplace):
    total_outer = inp.numel() // inner
    block_inner = 256
    grid = (total_outer, triton.cdiv(inner, block_inner))
    mask_for_kernel = mask.to(torch.int)
    value = float(_as_scalar_value(value))
    if inplace:
        _masked_fill_lastdim_broadcast_inplace_kernel[grid](
            inp,
            mask_for_kernel,
            total_outer,
            inner,
            value,
            BLOCK_INNER=block_inner,
        )
        return inp

    out = torch.empty_like(inp)
    _masked_fill_lastdim_broadcast_kernel[grid](
        inp,
        mask_for_kernel,
        out,
        total_outer,
        inner,
        value,
        BLOCK_INNER=block_inner,
    )
    return out


def masked_fill(inp, mask, value):
    logger.debug("SOPHGO MASKED_FILL")
    assert (
        (torch.is_tensor(value) and value.ndim == 0)
        or isinstance(value, int)
        or isinstance(value, float)
    ), "masked_fill only supports a 0-dimensional value tensor or a scalar"
    assert broadcastable_to(
        mask.shape, inp.shape
    ), "The shape of mask must be broadcastable with the shape of the underlying tensor"

    if inp.ndim == 0 or inp.numel() == 0:
        return _fallback_masked_fill(inp, mask, value)

    if _can_use_fast_path(inp, mask, value):
        if tuple(mask.shape) == tuple(inp.shape):
            if inp.numel() <= 65536:
                return _launch_same_shape_small_select(inp, mask, value)
            return _fallback_masked_fill(inp, mask, value)
        inner = _lastdim_broadcast_inner(inp, mask)
        if inner is not None:
            return _launch_lastdim_broadcast(inp, mask, value, inner, inplace=False)

    return _fallback_masked_fill(inp, mask, value)


def masked_fill_(inp, mask, value):
    logger.debug("SOPHGO MASKED_FILL_")
    assert (
        (torch.is_tensor(value) and value.ndim == 0)
        or isinstance(value, int)
        or isinstance(value, float)
    ), "masked_fill_ only supports a 0-dimensional value tensor or a scalar"
    assert broadcastable_to(
        mask.shape, inp.shape
    ), "The shape of mask must be broadcastable with the shape of the underlying tensor"

    if inp.ndim == 0 or inp.numel() == 0:
        return _fallback_masked_fill_(inp, mask, value)

    if _can_use_fast_path(inp, mask, value):
        if tuple(mask.shape) == tuple(inp.shape):
            if inp.numel() <= 65536:
                return _launch_same_shape(inp, mask, value, inplace=True)
            return _fallback_masked_fill_(inp, mask, value)
        inner = _lastdim_broadcast_inner(inp, mask)
        if inner is not None:
            return _launch_lastdim_broadcast(inp, mask, value, inner, inplace=True)

    return _fallback_masked_fill_(inp, mask, value)
