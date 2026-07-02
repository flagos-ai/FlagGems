import math
import os

from .tile_impl import tile as _original_tile

# the uplimit f32 can present the precision of i32
_F32_PRECISION_NUMEL_THRESHOLD = 2**24


def _compute_output_numel(x, dims):
    in0_shape = list(x.shape)
    dims_shape = list(dims)
    diff = len(in0_shape) - len(dims_shape)
    if diff > 0:
        dims_shape = [1] * diff + dims_shape
    elif diff < 0:
        in0_shape = [1] * (-diff) + in0_shape
    out_shape = [s * d for s, d in zip(in0_shape, dims_shape)]
    return math.prod(out_shape)


def tile(inp, dims):
    original_precision_priority = os.environ.get("PRECISION_PRIORITY", None)
    out_numel = _compute_output_numel(inp, dims)
    if out_numel > _F32_PRECISION_NUMEL_THRESHOLD:
        os.environ["PRECISION_PRIORITY"] = "1"

    try:
        return _original_tile(inp, dims)
    finally:
        if original_precision_priority is not None:
            os.environ["PRECISION_PRIORITY"] = original_precision_priority
        else:
            os.environ.pop("PRECISION_PRIORITY", None)
