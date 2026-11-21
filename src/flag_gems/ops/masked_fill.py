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
from flag_gems.utils import broadcastable_to, pointwise_dynamic

logger = logging.getLogger(__name__)

# One 128-bit transaction per thread: 4 warps over 1024 elements is 8 elements
# per thread for 16-bit dtypes. The mid-size class prefers fewer, fatter
# programs because its kernel run is too short to amortize the launch.
_FLAT_BLOCK = 1024
_FLAT_WARPS = 4
_MID_BLOCK = 4096
_MID_WARPS = 8
_MID_MIN_NUMEL = 1 << 17
_MID_MAX_NUMEL = 1 << 28
_INT32_MAX = 1 << 31


@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, "NO_OPMATH")])
@triton.jit
def masked_fill_kernel(inp, expand_mask, value):
    inp = tl.where(expand_mask == 1, value, inp)
    return inp


@pointwise_dynamic(is_tensor=[True, True, True], promotion_methods=[(0, "NO_OPMATH")])
@triton.jit
def _masked_fill_tensor_value_kernel(inp, mask, value):
    return tl.where(mask == 1, value.to(inp.dtype), inp)


@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, "NO_OPMATH")])
@triton.jit
def masked_fill_backward_kernel(grad_output, expand_mask, zero):
    # d/d input of ``where(mask, value, input)``: the upstream gradient is
    # propagated unchanged wherever the mask is false and is zeroed out on
    # every selected position. ``zero`` keeps the output dtype identical to
    # ``grad_output`` for every supported floating dtype. This generic path
    # covers arbitrary rank, non-contiguous inputs and every broadcast mask.
    return tl.where(expand_mask == 1, zero, grad_output)


@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, "NO_OPMATH")])
@triton.jit
def masked_fill_value_grad_kernel(grad_output, expand_mask, zero):
    # d/d value of ``where(mask, value, input)`` with a 0-dim value broadcast
    # over the masked positions: keep the upstream gradient only where the
    # mask is true. The reduction to a 0-dim tensor happens in ``sum``.
    return tl.where(expand_mask == 1, grad_output, zero)


@triton.jit
def masked_fill_backward_flat_kernel(
    grad_ptr,
    mask_ptr,
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    INT64: tl.constexpr,
):
    """Flat ``grad_output`` with a mask of the same shape and layout."""
    if INT64:
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        grad = tl.load(grad_ptr + offs)
        keep = tl.load(mask_ptr + offs)
        tl.store(out_ptr + offs, tl.where(keep, tl.zeros((BLOCK,), grad.dtype), grad))
    else:
        in_bounds = offs < n_elements
        grad = tl.load(grad_ptr + offs, mask=in_bounds, other=0.0)
        keep = tl.load(mask_ptr + offs, mask=in_bounds, other=0)
        tl.store(
            out_ptr + offs,
            tl.where(keep, tl.zeros((BLOCK,), grad.dtype), grad),
            mask=in_bounds,
        )


@triton.jit
def masked_fill_backward_wrap_kernel(
    grad_ptr,
    mask_ptr,
    out_ptr,
    n_elements,
    MASK_N: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    INT64: tl.constexpr,
):
    """Flat ``grad_output`` with a mask that repeats every ``MASK_N`` elements.

    Only leading dimensions may be broadcast: the mask is contiguous and its
    trailing run is a suffix of ``grad_output.shape``.
    """
    if INT64:
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        grad = tl.load(grad_ptr + offs)
        keep = tl.load(mask_ptr + (offs % MASK_N))
        tl.store(out_ptr + offs, tl.where(keep, tl.zeros((BLOCK,), grad.dtype), grad))
    else:
        in_bounds = offs < n_elements
        grad = tl.load(grad_ptr + offs, mask=in_bounds, other=0.0)
        keep = tl.load(mask_ptr + (offs % MASK_N), mask=in_bounds, other=0)
        tl.store(
            out_ptr + offs,
            tl.where(keep, tl.zeros((BLOCK,), grad.dtype), grad),
            mask=in_bounds,
        )


@triton.jit
def masked_fill_backward_sum_flat_kernel(
    grad_ptr,
    mask_ptr,
    part_ptr,
    n_elements,
    BLOCK: tl.constexpr,
    INT64: tl.constexpr,
):
    """Partial sums of the upstream gradient over the selected positions."""
    if INT64:
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_bounds = offs < n_elements
    grad = tl.load(grad_ptr + offs, mask=in_bounds, other=0.0)
    keep = tl.load(mask_ptr + offs, mask=in_bounds, other=0)
    values = tl.where(keep, grad.to(tl.float32), 0.0)
    tl.store(part_ptr + tl.program_id(0), tl.sum(values))


@triton.jit
def masked_fill_backward_sum_wrap_kernel(
    grad_ptr,
    mask_ptr,
    part_ptr,
    n_elements,
    MASK_N: tl.constexpr,
    BLOCK: tl.constexpr,
    INT64: tl.constexpr,
):
    """Partial sums with a mask that repeats every ``MASK_N`` elements."""
    if INT64:
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_bounds = offs < n_elements
    grad = tl.load(grad_ptr + offs, mask=in_bounds, other=0.0)
    keep = tl.load(mask_ptr + (offs % MASK_N), mask=in_bounds, other=0)
    values = tl.where(keep, grad.to(tl.float32), 0.0)
    tl.store(part_ptr + tl.program_id(0), tl.sum(values))


def masked_fill(inp, mask, value):
    logger.debug("GEMS MASKED_FILL")
    if torch.is_grad_enabled() and (
        inp.requires_grad or (torch.is_tensor(value) and value.requires_grad)
    ):
        _validate_masked_fill_args(inp, mask, value, "masked_fill")
        return MaskedFillFunction.apply(inp, mask, value)
    if torch.is_tensor(value) and value.ndim == 0 and value.device == inp.device:
        _validate_masked_fill_args(inp, mask, value, "masked_fill")
        return _masked_fill_tensor_value_kernel(inp, mask.expand(inp.shape), value)
    assert (
        (torch.is_tensor(value) and value.ndim == 0)
        or isinstance(value, int)
        or isinstance(value, float)
    ), "masked_fill_ only supports a 0-dimensional value tensor"
    if torch.is_tensor(value):
        # Value can be a tensor or a scalar
        value = value.item()
    assert broadcastable_to(
        mask.shape, inp.shape
    ), "The shape of mask must be broadcastable with the shape of the underlying tensor"

    if inp.ndim == 0:
        # inp is a single-value
        return (
            torch.tensor(value, dtype=inp.dtype, device=inp.device)
            if mask.item()
            else inp.clone()
        )

    expand_mask = mask.expand(inp.shape)
    return masked_fill_kernel(inp, expand_mask, value)


def _validate_masked_fill_args(inp, mask, value, name):
    assert (
        (torch.is_tensor(value) and value.ndim == 0)
        or isinstance(value, int)
        or isinstance(value, float)
    ), f"{name} only supports a 0-dimensional value tensor"
    assert broadcastable_to(
        mask.shape, inp.shape
    ), "The shape of mask must be broadcastable with the shape of the underlying tensor"


def _masked_fill_forward(inp, mask, value):
    """Keep the tensor value on the device inside the autograd forward."""
    expand_mask = mask.expand(inp.shape)
    if torch.is_tensor(value):
        if value.device == inp.device:
            return _masked_fill_tensor_value_kernel(inp, expand_mask, value)
        if value.device.type != "cpu":
            raise RuntimeError(
                "masked_fill(): value must be on the input device or CPU"
            )
        value = value.item()  # Only a host-resident scalar reaches this branch.
    return masked_fill_kernel(inp, expand_mask, value)


def _flat_run_length(grad_shape, mask_shape):
    """Trailing run length over which a contiguous mask repeats.

    Returns ``None`` when the mask cannot be addressed by a single modulo of
    the flat ``grad_output`` index, i.e. when some leading dimension of
    ``grad_output`` is not broadcast or the mask is not a suffix of its shape.
    """
    rank = len(grad_shape)
    if len(mask_shape) > rank:
        return None
    padded = (1,) * (rank - len(mask_shape)) + tuple(mask_shape)
    run = 1
    dim = rank - 1
    while dim >= 0 and padded[dim] == grad_shape[dim]:
        run *= grad_shape[dim]
        dim -= 1
    for leading in range(dim + 1):
        if padded[leading] != 1:
            return None
    return run


def _flat_tile(numel):
    if _MID_MIN_NUMEL < numel < _MID_MAX_NUMEL:
        return _MID_BLOCK, _MID_WARPS
    return _FLAT_BLOCK, _FLAT_WARPS


def _flat_ready(grad_output, mask, out):
    """Vector-wide alignment of every pointer in the flat fast path."""
    mask_ptr = 0 if mask is None else mask.data_ptr()
    return ((grad_output.data_ptr() | out.data_ptr() | mask_ptr) & 15) == 0


def _masked_fill_input_grad(grad_output, mask):
    """Gradient with respect to ``input``: zero on masked positions."""
    numel = grad_output.numel()
    if grad_output.ndim == 0:
        # A 0-dim input has a single element: the mask either zeroes it or
        # propagates it unchanged.
        return masked_fill_backward_kernel(grad_output, mask, 0)
    if numel == 0:
        return torch.empty_like(grad_output)

    out = None
    wrap = None
    if grad_output.is_contiguous() and mask.is_contiguous():
        if grad_output.shape == mask.shape:
            wrap = 0
        else:
            wrap = _flat_run_length(grad_output.shape, mask.shape)

    if wrap is not None and numel > 0:
        out = torch.empty_like(grad_output)
        if _flat_ready(grad_output, mask, out):
            block, warps = _flat_tile(numel)
            even = (numel % block) == 0
            int64 = numel >= _INT32_MAX
            logger.debug("GEMS MASKED_FILL_BACKWARD")
            if wrap == 0:
                kernel = masked_fill_backward_flat_kernel
            else:
                kernel = masked_fill_backward_wrap_kernel
            # constexpr order follows the kernel signature:
            # ``(MASK_N, BLOCK, EVEN, INT64)`` for the wrapped variant and
            # ``(BLOCK, EVEN, INT64)`` otherwise.
            consts = (wrap, block, even, int64) if wrap else (block, even, int64)
            with torch_device_fn.device(grad_output.device):
                kernel[(triton.cdiv(numel, block),)](
                    grad_output, mask, out, numel, *consts, num_warps=warps
                )
            return out

    expand_mask = mask.expand(grad_output.shape)
    logger.debug("GEMS MASKED_FILL_BACKWARD")
    return masked_fill_backward_kernel(grad_output, expand_mask, 0)


def _masked_fill_value_grad(grad_output, mask):
    """Gradient with respect to a 0-dim differentiable ``value`` tensor.

    ``value`` is broadcast over every selected element, so its gradient is the
    sum of the upstream gradients at the masked positions
    (``ATen``: ``grad.masked_select(mask).sum()``).
    """
    numel = grad_output.numel()
    if grad_output.ndim == 0:
        # The single element is either fully replaced by ``value`` or not
        # reached by the broadcast at all.
        return masked_fill_value_grad_kernel(grad_output, mask, 0)

    from flag_gems.ops.sum import sum as _sum

    if numel == 0:
        return torch.zeros([], dtype=torch.float32, device=grad_output.device)

    wrap = None
    if grad_output.is_contiguous() and mask.is_contiguous():
        if grad_output.shape == mask.shape:
            wrap = 0
        else:
            wrap = _flat_run_length(grad_output.shape, mask.shape)

    if wrap is not None and numel > 0:
        block, warps = _flat_tile(numel)
        num_parts = triton.cdiv(numel, block)
        parts = torch.empty(num_parts, dtype=torch.float32, device=grad_output.device)
        int64 = numel >= _INT32_MAX
        logger.debug("GEMS MASKED_FILL_VALUE_BACKWARD")
        if wrap == 0:
            kernel = masked_fill_backward_sum_flat_kernel
            consts = (block, int64)
        else:
            kernel = masked_fill_backward_sum_wrap_kernel
            consts = (wrap, block, int64)
        with torch_device_fn.device(grad_output.device):
            kernel[(num_parts,)](
                grad_output, mask, parts, numel, *consts, num_warps=warps
            )
        # ATen accumulates the selected elements in float32; the partial sums
        # above are already float32, so the final reduction keeps that dtype.
        return _sum(parts)

    logger.debug("GEMS MASKED_FILL_VALUE_BACKWARD")
    masked_grad = masked_fill_value_grad_kernel(
        grad_output, mask.expand(grad_output.shape), 0
    )
    return _sum(masked_grad, dtype=torch.float32)


class MaskedFillFunction(torch.autograd.Function):
    """Autograd wrapper around the Triton ``masked_fill`` kernel.

    ``torch.masked_fill`` has no standalone ``aten::masked_fill_backward``
    schema: ATen derives the backward from ``derivatives.yaml`` as
    ``grad_input = grad.masked_fill(mask, 0)`` and, for the ``.Tensor``
    overload, ``grad_value = grad.masked_select(mask).sum()``. This wrapper
    keeps the FlagGems entry differentiable with a fused Triton kernel for the
    input gradient and a fused masked reduction for a 0-dim ``value``.

    The ATen-generated formula stays in place for the ``torch.masked_fill``
    dispatch path: its input gradient is itself ``masked_fill``, so it
    redispatches into the FlagGems kernel below. Only the direct FlagGems
    entry can also replace the ``masked_select().sum()`` value gradient.
    """

    @staticmethod
    def forward(ctx, inp, mask, value):
        logger.debug("GEMS MASKED_FILL_FORWARD")
        ctx.save_for_backward(mask)
        ctx.value_dtype = value.dtype if torch.is_tensor(value) else None
        return _masked_fill_forward(inp, mask, value)

    @staticmethod
    def backward(ctx, grad_output):
        logger.debug("GEMS MASKED_FILL_AUTOGRAD_BACKWARD")
        (mask,) = ctx.saved_tensors

        grad_inp = None
        if ctx.needs_input_grad[0]:
            grad_inp = _masked_fill_input_grad(grad_output, mask)

        grad_value = None
        if ctx.needs_input_grad[2]:
            grad_value = _masked_fill_value_grad(grad_output, mask)
            if ctx.value_dtype is not None and grad_value.dtype != ctx.value_dtype:
                grad_value = grad_value.to(ctx.value_dtype)

        return grad_inp, None, grad_value


def masked_fill_(inp, mask, value):
    logger.debug("GEMS MASKED_FILL")
    assert (
        (torch.is_tensor(value) and value.ndim == 0)
        or isinstance(value, int)
        or isinstance(value, float)
    ), "masked_fill_ only supports a 0-dimensional value tensor"
    if torch.is_tensor(value):
        # Value can be a tensor or a scalar
        value = value.item()
    assert broadcastable_to(
        mask.shape, inp.shape
    ), "The shape of mask must be broadcastable with the shape of the underlying tensor"

    if inp.ndim == 0:
        # inp is a single-value
        if mask.item():
            inp[()] = value
        return inp

    expand_mask = mask.expand(inp.shape)
    return masked_fill_kernel(inp, expand_mask, value, out0=inp)


def masked_fill_backward(
    grad_output, mask, value=None, *, needs_input_grad=(True, True)
):
    """Return the input gradient and, for a tensor fill value, its scalar gradient."""
    need_input, need_value = needs_input_grad
    grad_input = _masked_fill_input_grad(grad_output, mask) if need_input else None
    grad_value = None
    if need_value and torch.is_tensor(value):
        grad_value = _masked_fill_value_grad(grad_output, mask).to(value.dtype)
    return grad_input, grad_value
