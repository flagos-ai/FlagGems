import logging

import torch
import triton

from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.codegen_config_utils import CodeGenConfig

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


config_for_general = CodeGenConfig(
    1024,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=False,
    # num_warps=2
)


@pointwise_dynamic(
    is_tensor=[True, True, False],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_for_general,
)
@triton.jit
def add_func(x, y, alpha):
    return x + y * alpha


config_for_broadcast = CodeGenConfig(
    128,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    # num_warps=4
)


@pointwise_dynamic(
    is_tensor=[True, True, False],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_for_broadcast,
)
@triton.jit
def add_func_broadcast(x, y, alpha):
    return x + y * alpha


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def add_func_tensor_scalar(x, y, alpha):
    return x + y * alpha


@pointwise_dynamic(
    is_tensor=[False, True, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def add_func_scalar_tensor(x, y, alpha):
    return x + y * alpha


def get_best_strided_output_tensor(A, B):
    def get_best_strides(A, B, broadcast_shape):
        if A.shape == broadcast_shape:
            return A.stride()
        elif B.shape == broadcast_shape:
            return B.stride()
        return None

    broadcast_shape = torch.broadcast_shapes(A.shape, B.shape)
    dtype = torch.float32
    out = torch.empty(broadcast_shape, device=A.device, dtype=dtype)
    best_stride = get_best_strides(A, B, broadcast_shape)
    if best_stride is not None:
        out = out.as_strided(broadcast_shape, best_stride)
    return out


def is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0


def should_use_broadcast_configs(A, B):
    # In scenarios where broadcasting is involved and the last two dimensions
    # of the two input tensors are the same, we use 1D tiling with a smaller
    # max_tile_size config for better performance.
    need_broadcast = A.shape != B.shape
    has_equal_last_dimentions = (
        len(A.shape) >= 2 and len(B.shape) >= 2 and A.shape[-2:] == B.shape[-2:]
    )
    return (
        need_broadcast
        and has_equal_last_dimentions
        and not is_power_of_two(A.shape[-1])
        and torch.result_type(A, B) in [torch.float16, torch.float32]
    )


def add(A, B, *, alpha=1):
    logger.debug("GEMS ADD")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if B.device != A.device:
            B = B.to(A.device)
        if should_use_broadcast_configs(A, B):
            out = get_best_strided_output_tensor(A, B)
            add_func_broadcast(A, B, alpha, out0=out)
            return out.to(torch.result_type(A, B))
        else:
            return add_func(A, B, alpha)
    elif isinstance(A, torch.Tensor):
        return add_func_tensor_scalar(A, B, alpha)
    elif isinstance(B, torch.Tensor):
        return add_func_scalar_tensor(A, B, alpha)
    else:
        return torch.tensor(A + B * alpha)


def add_(A, B, *, alpha=1):
    logger.debug("GEMS ADD_")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if B.device != A.device:
            B = B.to(A.device)
        return add_func(A, B, alpha, out0=A)
    elif isinstance(A, torch.Tensor):
        return add_func_tensor_scalar(A, B, alpha, out0=A)
    # elif isinstance(B, torch.Tensor):
    #     return add_func_scalar_tensor(A, B, alpha, out0=A)
    else:
        raise ValueError("Unreachable.")
