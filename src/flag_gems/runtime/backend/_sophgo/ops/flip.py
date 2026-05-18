import logging

import torch
import triton

from flag_gems.utils.codegen_config_utils import CodeGenConfig
from flag_gems.utils.pointwise_dynamic import pointwise_dynamic
from flag_gems.utils.tensor_wrapper import StridedBuffer

config_ = CodeGenConfig(
    64,
    (512, 1, 1),
    32,
    False,
    prefer_1d_tile=int(triton.__version__[0]) < 3,
)


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")], config=config_)
@triton.jit
def copy_func(x):
    return x


def _merge_to_4d(A: torch.Tensor, dims):
    """Merge consecutive non-flip dims to reduce rank to <=4."""
    ndim = A.ndim
    flip_set = set(dims)

    while ndim > 4:
        merge_idx = None
        for i in range(ndim - 2, -1, -1):
            if i not in flip_set and (i + 1) not in flip_set:
                merge_idx = i
                break
        if merge_idx is None:
            merge_idx = ndim - 2

        new_shape = list(A.shape)
        new_shape[merge_idx] = A.shape[merge_idx] * A.shape[merge_idx + 1]
        new_shape.pop(merge_idx + 1)

        new_dims = []
        for d in dims:
            if d <= merge_idx:
                new_dims.append(d)
            elif d == merge_idx + 1:
                if merge_idx not in new_dims:
                    new_dims.append(merge_idx)
            else:
                new_dims.append(d - 1)

        A = A.reshape(new_shape)
        dims = new_dims
        flip_set = set(dims)
        ndim = A.ndim

    return A, dims


def flip(A: torch.Tensor, dims) -> torch.Tensor:
    logging.debug("GEMS FLIP")
    orig_shape = A.shape

    if A.ndim > 4:
        A, dims = _merge_to_4d(A.contiguous(), dims)

    ndim = A.ndim
    norm_dims = [d if d >= 0 else ndim + d for d in dims]

    strides = list(A.stride())
    offset = 0
    for d in norm_dims:
        if A.size(d) > 1 and A.stride(d) != 0:
            offset += strides[d] * (A.shape[d] - 1)
            strides[d] = -strides[d]

    if offset == 0 or A.numel() <= 1:
        result = A.clone()
    else:
        out = torch.empty_like(A)
        flipped_A = StridedBuffer(A, strides=strides, offset=offset)
        overload = copy_func.instantiate(ndim)
        overload(flipped_A, out0=out)
        result = out

    if result.shape != orig_shape:
        result = result.reshape(orig_shape)
    return result