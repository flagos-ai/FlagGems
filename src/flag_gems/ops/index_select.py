import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("index_select"))
@triton.jit
def index_select_kernel(
    inp, out, M, N, K, index, index_len, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_x = tle.program_id(axis=0)
    pid_y = tle.program_id(axis=1)
    rows_offsets = pid_x * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    rows_mask = rows_offsets < M
    cols_offsets = pid_y * BLOCK_N + tl.arange(0, BLOCK_N)

    out_mask = rows_mask & (cols_offsets < index_len)

    indices = tl.load(index + cols_offsets, mask=(cols_offsets < index_len), other=0)
    valid_lower_bound = indices >= 0
    valid_upper_bound = indices < N
    index_valid_mask = valid_lower_bound & valid_upper_bound

    # `rows_offsets` enumerates the (outer, inner) pairs around the indexed axis,
    # where K is the number of trailing elements. K == 1 is the last-axis case and
    # degenerates to the plain row-major layout, so indexing a non-last axis no
    # longer needs the tensor transposed into place and back.
    outer = rows_offsets // K
    inner = rows_offsets % K
    inp_off = outer * N * K + indices[None, :] * K + inner
    out_off = outer * index_len * K + cols_offsets[None, :] * K + inner

    final_mask = out_mask & index_valid_mask
    selected = tl.load(inp + inp_off, mask=final_mask, other=0.0)
    tl.store(out + out_off, selected, mask=final_mask)


def index_select(inp, dim, index):
    logger.debug("GEMS INDEX SELECT")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index.ndim <= 1, "Index should have dimension 1 or 0"

    if index.ndim == 0:
        index = index.unsqueeze(0)
    dim = dim % inp.ndim
    inp_shape = list(inp.shape)
    index_len = 1
    for s in index.shape: index_len *= s

    # Index in place with strided addressing: dim_compress would copy the whole
    # tensor to move the axis last, and the result would need a second copy back.
    inp = inp.contiguous()
    N = inp_shape[dim]
    K = math.prod(inp_shape[dim + 1 :])
    M = math.prod(inp_shape[:dim]) * K

    out_shape = list(inp_shape)
    out_shape[dim] = index_len
    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(index_len, meta["BLOCK_N"]),
    )
    index_select_kernel[grid](inp, out, M, N, K, index, index_len)
    return out

def index_select_paddle(inp, index, dim, out=None):
    return index_select(inp, dim, index)
