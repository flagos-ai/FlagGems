import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _fractional_max_pool2d_forward_kernel(
    input_ptr,
    output_ptr,
    indices_ptr,
    random_samples_ptr,
    numel,
    input_channels: tl.constexpr,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    output_height: tl.constexpr,
    output_width: tl.constexpr,
    kernel_height: tl.constexpr,
    kernel_width: tl.constexpr,
    alpha_height,
    alpha_width,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    store_mask = offsets < numel
    safe_offsets = tl.where(store_mask, offsets, 0)

    output_column = safe_offsets % output_width
    ohw = safe_offsets // output_width
    output_row = ohw % output_height
    nc = ohw // output_height

    sample_height = tl.load(random_samples_ptr + nc * 2).to(tl.float32)
    sample_width = tl.load(random_samples_ptr + nc * 2 + 1).to(tl.float32)

    sample_alpha_h = (sample_height * alpha_height).to(tl.int32)
    sample_alpha_w = (sample_width * alpha_width).to(tl.int32)
    start_height = ((output_row.to(tl.float32) + sample_height) * alpha_height).to(
        tl.int32
    ) - sample_alpha_h
    start_width = ((output_column.to(tl.float32) + sample_width) * alpha_width).to(
        tl.int32
    ) - sample_alpha_w
    start_height = tl.where(
        output_row == output_height - 1,
        input_height - kernel_height,
        start_height,
    )
    start_width = tl.where(
        output_column == output_width - 1,
        input_width - kernel_width,
        start_width,
    )

    plane_base = input_ptr + nc * (input_height * input_width)
    max_value = tl.full((BLOCK_SIZE,), -float("inf"), tl.float32)
    max_index = tl.full((BLOCK_SIZE,), -1, tl.int64)
    for kernel_row in tl.static_range(0, kernel_height):
        input_row = start_height + kernel_row
        for kernel_column in tl.static_range(0, kernel_width):
            input_column = start_width + kernel_column
            value = tl.load(plane_base + input_row * input_width + input_column).to(
                tl.float32
            )
            update = value > max_value
            max_value = tl.where(update, value, max_value)
            max_index = tl.where(
                update,
                input_row.to(tl.int64) * input_width + input_column,
                max_index,
            )

    tl.store(
        output_ptr + offsets,
        max_value.to(output_ptr.dtype.element_ty),
        mask=store_mask,
    )
    tl.store(indices_ptr + offsets, max_index, mask=store_mask)


@libentry()
@triton.jit
def _fractional_max_pool2d_backward_scatter_kernel(
    grad_output_ptr,
    indices_ptr,
    grad_input_ptr,
    n_out,
    out_per_nc,
    in_hw,
    BLOCK: tl.constexpr,
):
    """One lane per output position, non-atomic scatter (Kunlunxin/XPU).

    Fast path, valid whenever ``alpha >= k`` in both dims (``in - k >= k *
    (out - 1)``, or ``out == 1``).  The pool window of output ``o`` is
    ``[start(o), start(o) + k)`` with ``start(o) = trunc((o + s) * alpha) -
    trunc(s * alpha)``; consecutive starts satisfy ``start(o + 1) - start(o)
    >= floor(alpha) >= k`` (``start(out - 1)`` is clamped to ``in - k`` but
    ``(out - 1) * alpha == in - k`` up to sub-ulp fp32 rounding, so the final
    gap is ``>= k`` as well).  Hence the k-wide windows are pairwise disjoint,
    every input position belongs to at most one window, each output's argmax
    index is distinct, and the plain (non-atomic) stores can never race --
    exact, deterministic, O(n_out).  This is the same pattern as the proven
    ``adaptive_max_pool2d`` scatter fast path (``tl.atomic_add`` scatter loses
    updates on this backend ~1e-5 per op, seed-dependent).
    """
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_out
    # ``n_out`` is not always a multiple of ``BLOCK`` (the (2, 4, 20, 20) shape of
    # tests/test_fractional_max_pool2d.py gives n_out = 800 with BLOCK = 256), and the
    # tail lanes used to be masked on the store only: the two loads below then read up
    # to BLOCK-1 elements past the end of ``grad_output``/``indices``.  Out-of-bounds
    # accesses are a device-wedge hazard on this backend, so predicate the loads too.
    idx = tl.load(indices_ptr + offsets, mask=mask, other=0).to(tl.int32)
    val = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    nc = offsets // out_per_nc
    tl.store(
        grad_input_ptr + nc * in_hw + idx,
        val.to(grad_input_ptr.dtype.element_ty),
        mask=mask,
    )


@libentry()
@triton.jit
def _fractional_max_pool2d_backward_gather_kernel(
    grad_output_ptr,
    indices_ptr,
    grad_input_ptr,
    n_elems,
    in_h,
    in_w,
    out_h,
    out_w,
    inv_alpha_h,
    inv_alpha_w,
    k_h: tl.constexpr,
    k_w: tl.constexpr,
    MAX_H: tl.constexpr,
    MAX_W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Gather-based fractional max pool 2d backward (Kunlunxin/XPU).

    General fallback (used when ``alpha < k`` in some dim, i.e. overlapping
    windows where the scatter fast path is not race-free).  One lane per input
    position (flat 1-D over n, c, h, w).  Every output's argmax is looked up
    in the per-plane flat index ``h * in_w + w`` produced by the Gems forward
    and the upstream gradient is accumulated in registers wherever it equals
    this lane's position, then written with one masked store.  This is the
    exact, deterministic, race-free pattern of the proven
    ``adaptive_max_pool2d_backward`` gather kernel (``tl.atomic_add`` scatter
    loses updates on this backend ~1e-5 per op, seed-dependent).

    The candidate output box is bounded *without* knowing the random samples.
    With ``start(o) = trunc((o + s) * alpha) - trunc(s * alpha)`` we have
    ``o*alpha - 1 <= start(o) <= o*alpha + 1``, and ``start(out - 1)`` is the
    clamped ``in - k``; hence every output row whose k-window may contain
    input row ``h`` satisfies ``(h - k_h) / alpha_h <= o <= (h + 1) / alpha_h``.
    `inv_alpha` is passed host-side as ``1 / alpha`` (0 when ``out == 1`` or
    ``in == k``, where the only candidate is ``o = 0``), the ``+-2`` margin
    absorbs fp32 rounding, and each candidate is verified exactly against the
    stored indices, so the result is identical to scanning every output
    (even across overlapping windows, multiple outputs can map to the same
    input position and their gradients all accumulate).  ``MAX_*`` is the
    host-computed tight bound ``(k + 1) / alpha + 5`` (see the host function);
    the scan range is rolled (not ``tl.static_range``): the XPU unroll control
    pass fails when the per-program body is fully unrolled.
    """
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elems
    safe_offsets = tl.where(mask, offsets, 0)

    in_hw = in_h * in_w
    nc = safe_offsets // in_hw
    rem = safe_offsets % in_hw
    h = rem // in_w
    w = rem % in_w
    my_flat = h * in_w + w

    oh_lo = tl.maximum(
        0, (tl.maximum(h - k_h, 0).to(tl.float32) * inv_alpha_h).to(tl.int32) - 2
    )
    oh_hi = tl.minimum(
        out_h - 1, ((h + 1).to(tl.float32) * inv_alpha_h).to(tl.int32) + 2
    )
    ow_lo = tl.maximum(
        0, (tl.maximum(w - k_w, 0).to(tl.float32) * inv_alpha_w).to(tl.int32) - 2
    )
    ow_hi = tl.minimum(
        out_w - 1, ((w + 1).to(tl.float32) * inv_alpha_w).to(tl.int32) + 2
    )

    out_per_nc = out_h * out_w
    gop = grad_output_ptr + nc * out_per_nc
    iop = indices_ptr + nc * out_per_nc

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for oh in range(0, MAX_H):
        o_h = oh_lo + oh
        h_ok = (o_h <= oh_hi) & (o_h < out_h)
        c_h = tl.minimum(o_h, out_h - 1)
        for ow in range(0, MAX_W):
            o_w = ow_lo + ow
            w_ok = (o_w <= ow_hi) & (o_w < out_w)
            c_w = tl.minimum(o_w, out_w - 1)
            o_off = c_h * out_w + c_w
            idx = tl.load(iop + o_off).to(tl.int32)
            val = tl.load(gop + o_off).to(tl.float32)
            active = mask & h_ok & w_ok
            acc += tl.where(active & (idx == my_flat), val, 0.0)

    tl.store(
        grad_input_ptr + offsets,
        acc.to(grad_input_ptr.dtype.element_ty),
        mask=mask,
    )


def _parse_size(value):
    if isinstance(value, (int, float)):
        return value, value
    return value[0], value[1]


def fractional_max_pool2d(
    input,
    kernel_size,
    output_size=None,
    output_ratio=None,
    return_indices=True,
    _random_samples=None,
):
    logger.debug("GEMS_KUNLUNXIN FRACTIONAL_MAX_POOL2D")
    if isinstance(output_ratio, torch.Tensor) and _random_samples is None:
        _random_samples = output_ratio
        output_ratio = None
    assert input.dim() == 4, f"Expected 4D input, got {input.dim()}D"
    input = input.contiguous()
    batch_size, channels, input_height, input_width = input.shape
    kernel_height, kernel_width = _parse_size(kernel_size)
    if output_size is not None:
        output_height, output_width = _parse_size(output_size)
    elif output_ratio is not None:
        ratio_height, ratio_width = _parse_size(output_ratio)
        output_height = int(input_height * ratio_height)
        output_width = int(input_width * ratio_width)
    else:
        raise ValueError("Either output_size or output_ratio must be specified")
    assert output_height + kernel_height - 1 <= input_height
    assert output_width + kernel_width - 1 <= input_width

    if _random_samples is None:
        _random_samples = torch.rand(
            batch_size,
            channels,
            2,
            device=input.device,
            dtype=input.dtype,
        )
    else:
        assert _random_samples.shape == (batch_size, channels, 2)
        _random_samples = _random_samples.to(dtype=input.dtype).contiguous()

    output = torch.empty(
        (batch_size, channels, output_height, output_width),
        device=input.device,
        dtype=input.dtype,
    )
    indices = torch.empty(
        (batch_size, channels, output_height, output_width),
        device=input.device,
        dtype=torch.int64,
    )
    if output.numel() == 0:
        return (output, indices) if return_indices else output
    alpha_height = (
        (input_height - kernel_height) / (output_height - 1)
        if output_height > 1
        else 0.0
    )
    alpha_width = (
        (input_width - kernel_width) / (output_width - 1) if output_width > 1 else 0.0
    )
    numel = output.numel()
    block_size = 128 if numel >= 8192 else 64
    grid = (triton.cdiv(numel, block_size),)
    with torch_device_fn.device(input.device):
        _fractional_max_pool2d_forward_kernel[grid](
            input,
            output,
            indices,
            _random_samples.reshape(batch_size * channels, 2),
            numel,
            channels,
            input_height,
            input_width,
            output_height,
            output_width,
            kernel_height,
            kernel_width,
            alpha_height,
            alpha_width,
            BLOCK_SIZE=block_size,
            num_warps=1,
            isCloseVectorization=True,
            buffer_size_limit=2048,
        )
    if return_indices:
        return output, indices
    return output


def fractional_max_pool2d_backward(
    grad_output,
    input,
    kernel_size,
    output_size,
    indices,
):
    logger.debug("GEMS_KUNLUNXIN FRACTIONAL_MAX_POOL2D_BACKWARD")
    input = input.contiguous()
    grad_output = grad_output.contiguous()
    indices = indices.contiguous()
    batch_size, channels, input_height, input_width = input.shape
    kernel_height, kernel_width = _parse_size(kernel_size)
    output_height, output_width = _parse_size(output_size)
    if input.numel() == 0 or grad_output.numel() == 0:
        return torch.zeros_like(input)
    out_per_nc = output_height * output_width
    in_hw = input_height * input_width
    n_out = grad_output.numel()
    scatter_h = (output_height == 1) or (
        input_height - kernel_height >= kernel_height * (output_height - 1)
    )
    scatter_w = (output_width == 1) or (
        input_width - kernel_width >= kernel_width * (output_width - 1)
    )
    with torch_device_fn.device(input.device):
        if scatter_h and scatter_w:
            grad_input = torch.zeros_like(input)
            _fractional_max_pool2d_backward_scatter_kernel[(triton.cdiv(n_out, 256),)](
                grad_output,
                indices,
                grad_input,
                n_out,
                out_per_nc,
                in_hw,
                BLOCK=256,
                num_warps=4,
                isCloseVectorization=True,
                buffer_size_limit=2048,
            )
        else:
            inv_alpha_h = (
                (output_height - 1) / (input_height - kernel_height)
                if input_height > kernel_height
                else 0.0
            )
            inv_alpha_w = (
                (output_width - 1) / (input_width - kernel_width)
                if input_width > kernel_width
                else 0.0
            )
            max_h = max(
                1,
                min(
                    output_height, int(math.ceil((kernel_height + 1) * inv_alpha_h)) + 6
                ),
            )
            max_w = max(
                1,
                min(output_width, int(math.ceil((kernel_width + 1) * inv_alpha_w)) + 6),
            )
            n_elems = input.numel()
            grad_input = torch.empty_like(input)
            _fractional_max_pool2d_backward_gather_kernel[(triton.cdiv(n_elems, 128),)](
                grad_output,
                indices,
                grad_input,
                n_elems,
                input_height,
                input_width,
                output_height,
                output_width,
                inv_alpha_h,
                inv_alpha_w,
                kernel_height,
                kernel_width,
                MAX_H=max_h,
                MAX_W=max_w,
                BLOCK=128,
                num_warps=2,
                isCloseVectorization=True,
                buffer_size_limit=2048,
            )
    return grad_input
