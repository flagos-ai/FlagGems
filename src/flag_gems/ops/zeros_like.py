import logging

import torch
import triton
from paddle.tensor.creation import zeros_like as _paddle_zeros_like
from paddle.utils.decorator_utils import param_one_alias

from flag_gems.ops.zeros import gpu_place, zeros_kernel
from flag_gems.runtime import device_guard
from flag_gems.utils.shape_utils import volume

logger = logging.getLogger(__name__)


def zeros_like(
    x, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None
):
    logger.debug("GEMS ZEROS_LIKE")
    if device is None:
        device = x.device
    if dtype is None:
        dtype = x.dtype
    out = torch.empty_like(x, device=device, dtype=dtype)
    # paddle's numel() returns a Tensor, which Triton would take as a pointer
    N = volume(x.shape)
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    with device_guard(x):
        zeros_kernel[grid_fn](out, N, BLOCK_SIZE=1024)
    return out

def zeros_like_paddle(x: 'Tensor', dtype = None, place = None) -> 'Tensor':
    return zeros_like(x, dtype=dtype, device=place)


# paddle.zeros_like never reaches _C_ops.zeros_like: it goes through
# paddle.full_like to _C_ops.full_like, so taking over _C_ops.zeros_like alone
# leaves the public API on Paddle's own kernel. This replaces the public entry and
# keeps its signature.
@param_one_alias(["x", "input"])
def zeros_like_paddle_api(
    x, dtype=None, name=None, *, device=None, requires_grad=False, pin_memory=False
):
    place = gpu_place(x.place if device is None else device)
    if place is None or pin_memory:
        # The kernel fills device memory allocated here; CPU and pinned tensors
        # stay on Paddle's implementation.
        return _paddle_zeros_like(
            x,
            dtype,
            name,
            device=device,
            requires_grad=requires_grad,
            pin_memory=pin_memory,
        )
    res = zeros_like(x, dtype=dtype, device=place)
    res.stop_gradient = not requires_grad
    return res