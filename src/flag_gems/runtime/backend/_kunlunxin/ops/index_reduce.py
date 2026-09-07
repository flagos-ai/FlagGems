import logging
import math

import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_BLOCK_SIZE = 64


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
    BLOCK: tl.constexpr = 64
    pid = tl.program_id(0)
    nblocks = tl.cdiv(output_dim_size * inner_size, BLOCK)
    outer = pid // nblocks
    block_id = pid - outer * nblocks
    offs = block_id * BLOCK + tl.arange(0, BLOCK)
    within = offs < output_dim_size * inner_size
    output_dim_offset = offs // inner_size
    inner_offset = offs % inner_size
    base = outer * output_dim_size * inner_size

    self_value = tl.load(inp + base + offs, mask=within, other=0.0).to(tl.float32)

    if REDUCE == 0:
        accumulator = self_value if INCLUDE_SELF else tl.full((BLOCK,), 1.0, dtype=tl.float32)
    elif REDUCE == 1:
        accumulator = self_value if INCLUDE_SELF else tl.zeros((BLOCK,), dtype=tl.float32)
    elif REDUCE == 2:
        accumulator = self_value if INCLUDE_SELF else tl.full((BLOCK,), float("-inf"), dtype=tl.float32)
    else:
        accumulator = self_value if INCLUDE_SELF else tl.full((BLOCK,), float("inf"), dtype=tl.float32)
    count = tl.full((BLOCK,), 1 if INCLUDE_SELF else 0, dtype=tl.int32)

    for source_dim_offset in tl.range(0, SOURCE_DIM_SIZE):
        selected = tl.load(index + source_dim_offset).to(tl.int64) == output_dim_offset
        selected = selected & within
        source_offset = (
            outer * SOURCE_DIM_SIZE + source_dim_offset
        ) * inner_size + inner_offset
        value = tl.load(source + source_offset, mask=within, other=0.0).to(tl.float32)
        if REDUCE == 0:
            accumulator = tl.where(selected, accumulator * value, accumulator)
        elif REDUCE == 1:
            accumulator = tl.where(selected, accumulator + value, accumulator)
        elif REDUCE == 2:
            accumulator = tl.where(
                selected, tl.maximum(accumulator, value), accumulator
            )
        else:
            accumulator = tl.where(
                selected, tl.minimum(accumulator, value), accumulator
            )
        count += selected.to(tl.int32)

    if REDUCE == 1:
        accumulator = accumulator / tl.maximum(count, 1).to(tl.float32)
    if not INCLUDE_SELF:
        accumulator = tl.where(count == 0, self_value, accumulator)
    tl.store(output + base + offs, accumulator, mask=within)


_REDUCTIONS = {"prod": 0, "mean": 1, "amax": 2, "amin": 3}


def index_reduce_(inp, dim, index, source, reduce, *, include_self=True):
    logger.debug("GEMS_KUNLUNXIN INDEX_REDUCE_")
    if reduce not in _REDUCTIONS:
        raise RuntimeError(
            f"index_reduce(): Expected reduce to be one of prod, mean, amax or amin but got {reduce}."
        )
    if inp.ndim == 0:
        raise IndexError(
            "index_reduce_(): Expected self to have non-zero dimensionality"
        )

    dim %= inp.ndim
    if index.ndim != 1 or index.numel() != source.shape[dim]:
        raise RuntimeError(
            "index_reduce_(): Expected index to be a vector matching source.size(dim)"
        )
    if any(
        source.shape[axis] != inp.shape[axis] for axis in range(inp.ndim) if axis != dim
    ):
        raise RuntimeError(
            "index_reduce_(): source must match self outside the reduced dimension"
        )

    input_contiguous = inp.contiguous()
    source = source.contiguous()
    index = index.contiguous()
    result = input_contiguous.clone()
    inner_size = math.prod(inp.shape[dim + 1 :])
    nblocks = math.ceil(inp.shape[dim] * inner_size / _BLOCK_SIZE)
    outer_size = math.prod(inp.shape[:dim])
    with torch_device_fn.device(inp.device):
        _index_reduce_kernel[(outer_size * nblocks,)](
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
