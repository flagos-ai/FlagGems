import logging

import paddle
import torch
import triton
import triton.language as tl
from paddle.base.framework import _current_expected_place, _get_paddle_place
from paddle.tensor.creation import zeros as _paddle_zeros
from paddle.utils.decorator_utils import size_args_decorator

from flag_gems.runtime import device, device_guard
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.shape_utils import volume

device_ = device
logger = logging.getLogger(__name__)


@triton.jit
def zeros_kernel(
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(output_ptr + offsets, 0.0, mask=mask)


def zeros(size, *, dtype=None, layout=None, device=None, pin_memory=None):
    logger.debug("GEMS ZEROS")
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device(device_.name)

    out = torch.empty(size, device=device, dtype=dtype)
    N = volume(size)
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    with device_guard(device):
        zeros_kernel[grid_fn](out, N, BLOCK_SIZE=1024)
    return out

def zeros_paddle(shape: list[int], dtype = None, place = None) -> 'Tensor':
    return zeros(shape, dtype=dtype, device=place)


def gpu_place(device):
    """Resolve a paddle device denotation; None when it is not a GPU place."""
    place = _current_expected_place() if device is None else _get_paddle_place(device)
    if isinstance(place, paddle.CUDAPlace) or place.is_gpu_place():
        return place
    return None


def concrete_shape(shape):
    """paddle accepts a shape Tensor or a list holding 0-D Tensors; the kernels need ints."""
    if isinstance(shape, paddle.Tensor):
        return shape.tolist()
    return [int(s) for s in shape]


# paddle.zeros never reaches _C_ops.zeros: it goes through paddle.full to
# _C_ops.full, so taking over _C_ops.zeros alone leaves the public API on Paddle's
# own kernel. This replaces the public entry and keeps its signature.
@size_args_decorator
def zeros_paddle_api(
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
        return _paddle_zeros(
            shape,
            dtype,
            name,
            out=out,
            device=device,
            requires_grad=requires_grad,
            pin_memory=pin_memory,
        )
    res = zeros(concrete_shape(shape), dtype=dtype, device=place)
    res.stop_gradient = not requires_grad
    return res