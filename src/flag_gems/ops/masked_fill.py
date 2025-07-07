import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import broadcastable_to, pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, "NO_OPMATH")])
@triton.jit
def masked_fill_kernel(inp, expand_mask, value):
    inp = tl.where(expand_mask == 1, value, inp)
    return inp


def masked_fill(inp, mask, value):
    logger.debug("GEMS MASKED FILL")
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


def masked_fill_(inp, mask, value):
    logger.debug("GEMS MASKED FILL")
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
