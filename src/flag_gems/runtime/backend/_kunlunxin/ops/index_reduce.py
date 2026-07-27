import logging
import math

import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _index_reduce_kernel(
    inp,
    index,
    source,
    output,
    output_dim_size,
    inner_size,
    SOURCE_DIM_SIZE: tl.constexpr,
    REDUCE: tl.constexpr,
    INCLUDE_SELF: tl.constexpr,
):
    output_offset = tl.program_id(0)
    inner_offset = output_offset % inner_size
    output_dim_offset = (output_offset // inner_size) % output_dim_size
    outer_offset = output_offset // (output_dim_size * inner_size)
    self_value = tl.load(inp + output_offset).to(tl.float32)

    if REDUCE == 0:
        accumulator = self_value if INCLUDE_SELF else 1.0
    elif REDUCE == 1:
        accumulator = self_value if INCLUDE_SELF else 0.0
    elif REDUCE == 2:
        accumulator = self_value if INCLUDE_SELF else -float("inf")
    else:
        accumulator = self_value if INCLUDE_SELF else float("inf")
    count = 1 if INCLUDE_SELF else 0

    for source_dim_offset in tl.static_range(0, SOURCE_DIM_SIZE):
        selected = tl.load(index + source_dim_offset) == output_dim_offset
        source_offset = (
            (outer_offset * SOURCE_DIM_SIZE + source_dim_offset) * inner_size
            + inner_offset
        )
        value = tl.load(source + source_offset).to(tl.float32)
        if REDUCE == 0:
            accumulator = tl.where(selected, accumulator * value, accumulator)
        elif REDUCE == 1:
            accumulator = tl.where(selected, accumulator + value, accumulator)
        elif REDUCE == 2:
            accumulator = tl.where(selected, tl.maximum(accumulator, value), accumulator)
        else:
            accumulator = tl.where(selected, tl.minimum(accumulator, value), accumulator)
        count += selected.to(tl.int32)

    if REDUCE == 1:
        accumulator /= count
    if not INCLUDE_SELF:
        accumulator = tl.where(count == 0, self_value, accumulator)
    tl.store(output + output_offset, accumulator)


_REDUCTIONS = {"prod": 0, "mean": 1, "amax": 2, "amin": 3}


def index_reduce_(inp, dim, index, source, reduce, *, include_self=True):
    logger.debug("GEMS_KUNLUNXIN INDEX_REDUCE_")
    if reduce not in _REDUCTIONS:
        raise RuntimeError(f"index_reduce(): Expected reduce to be one of prod, mean, amax or amin but got {reduce}.")
    if inp.ndim == 0:
        raise IndexError("index_reduce_(): Expected self to have non-zero dimensionality")

    dim %= inp.ndim
    if index.ndim != 1 or index.numel() != source.shape[dim]:
        raise RuntimeError("index_reduce_(): Expected index to be a vector matching source.size(dim)")
    if any(
        source.shape[axis] != inp.shape[axis]
        for axis in range(inp.ndim)
        if axis != dim
    ):
        raise RuntimeError("index_reduce_(): source must match self outside the reduced dimension")

    input_contiguous = inp.contiguous()
    source = source.contiguous()
    index = index.contiguous()
    result = input_contiguous.clone()
    inner_size = math.prod(inp.shape[dim + 1 :])
    with torch_device_fn.device(inp.device):
        _index_reduce_kernel[(result.numel(),)](
            input_contiguous,
            index,
            source,
            result,
            inp.shape[dim],
            inner_size,
            SOURCE_DIM_SIZE=index.numel(),
            REDUCE=_REDUCTIONS[reduce],
            INCLUDE_SELF=include_self,
            isCloseVectorization=True,
            buffer_size_limit=2048,
        )
    inp.copy_(result)
    return inp
