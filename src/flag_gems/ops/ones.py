import logging

import torch
import triton
import triton.language as tl
from paddle.tensor.creation import ones as _paddle_ones
from paddle.utils.decorator_utils import size_args_decorator

from flag_gems.ops.zeros import concrete_shape, gpu_place
from flag_gems.runtime import device, device_guard
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.shape_utils import volume

device_ = device
logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def ones_kernel(
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(output_ptr + offsets, 1.0, mask=mask)


def ones(size, *, dtype=None, layout=None, device=None, pin_memory=None):
    logger.debug("GEMS ONES")
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = device_.name

    out = torch.empty(size, device=device, dtype=dtype)
    N = volume(size)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    with device_guard(device):
        ones_kernel[grid](out, N, BLOCK_SIZE)
    return out

def ones_paddle(shape, dtype = None, place = None) -> 'Tensor':
    return ones(shape, dtype=dtype, device=place)


# paddle.ones never reaches _C_ops.ones: it goes through paddle.full to _C_ops.full,
# so taking over _C_ops.ones alone leaves the public API on Paddle's own kernel.
# This replaces the public entry and keeps its signature.
@size_args_decorator
def ones_paddle_api(
    shape,
    dtype=None,
    name=None,
    *,
    out=None,
    device=None,
    requires_grad=False,
    pin_memory=False,
):
    place = gpu_place(device)
    if place is None or out is not None or pin_memory:
        # The kernel fills device memory allocated here; CPU and pinned tensors
        # and the out= form stay on Paddle's implementation.
        return _paddle_ones(
            shape,
            dtype,
            name,
            out=out,
            device=device,
            requires_grad=requires_grad,
            pin_memory=pin_memory,
        )
    res = ones(concrete_shape(shape), dtype=dtype, device=place)
    res.stop_gradient = not requires_grad
    return res