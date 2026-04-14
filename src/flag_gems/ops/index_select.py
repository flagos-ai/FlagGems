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
    inp, out, M, N, index, index_len, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_x = tle.program_id(axis=0)
    pid_y = tle.program_id(axis=1)
    rows_offsets = pid_x * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    rows_mask = rows_offsets < M
    cols_offsets = pid_y * BLOCK_N + tl.arange(0, BLOCK_N)

    out_mask = rows_mask and (cols_offsets < index_len)

    indices = tl.load(index + cols_offsets, mask=(cols_offsets < index_len), other=0)
    valid_lower_bound = indices >= 0
    valid_upper_bound = indices < N
    index_valid_mask = valid_lower_bound & valid_upper_bound

    inp_off = rows_offsets * N + indices[None, :]
    out_off = rows_offsets * index_len + cols_offsets[None, :]

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
    index_len = index.numel()

    # with dim_compress
    inp = dim_compress(inp, dim)
    N = inp_shape[dim]
    M = inp.numel() // N
    out_shape = list(inp.shape)
    out_shape[inp.ndim - 1] = index_len
    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(index_len, meta["BLOCK_N"]),
    )
    index_select_kernel[grid](inp, out, M, N, index, index_len)
    if dim != out.ndim - 1:
        order = [i for i in range(out.ndim - 1)]
        order.insert(dim, out.ndim - 1)
        out = out.permute(order).contiguous()
        return out.reshape(out.shape)
    else:
        return out


def dim_compress_backward(inp, dims):
    if isinstance(dims, int):
        dims = [dims]
    dim = inp.ndim
    stride = inp.stride()
    batch_dim = [i for i in range(dim) if i not in dims]
    sorted_reduction_dim = sorted(dims, key=lambda x: stride[x], reverse=True)
    order = sorted_reduction_dim + batch_dim
    return inp.permute(order).contiguous()


# kernel
@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("index_select_backward"),
    key=["M", "N"],
    reset_to_zero=["out"],
)
@triton.jit
def index_select_backward_kernel(
    grad,
    out,
    M,
    N,
    num_blocks_per_CTA,
    index,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_x = tle.program_id(axis=0)
    pid_y = tle.program_id(axis=1)
    rows_offsets = pid_x * BLOCK_M + tl.arange(0, BLOCK_M)
    cols_offsets = pid_y * BLOCK_N + tl.arange(0, BLOCK_N)

    grad_mask = (rows_offsets < M) and (cols_offsets < N)
    indices = tl.load(index + rows_offsets, mask=(rows_offsets < M), other=0)

    for i in range(0, num_blocks_per_CTA):
        grad_off = (pid_x * num_blocks_per_CTA + i) * N + cols_offsets
        out_off = (indices * num_blocks_per_CTA + i) * N + cols_offsets
        selected = tl.load(grad + grad_off, mask=grad_mask, other=0.0)
        tl.atomic_add(out + out_off, selected, mask=grad_mask)


# function
def index_select_backward(grad, self_sizes, dim, index):
    logging.debug("GEMS INDEX SELECT BACKWARD")
    assert dim >= -len(self_sizes) and dim < len(self_sizes), "Invalid dim"
    assert index.ndim <= 1, "Index should have dimension 1 or 0"
    if index.ndim == 0:
        index = index.unsqueeze(0)
    grad_init = grad.ndim
    if grad_init == 1:
        grad = grad.unsqueeze(1)
    index_shape = list(index.shape)
    dim = dim % len(self_sizes)
    grad_shape = list(grad.shape)
    assert grad_shape[dim] == index_shape[0], "Index out of range"
    grad = dim_compress_backward(grad, dim)
    grad_shape = list(grad.shape)
    out_shape = list(grad.shape)
    shape_for_block_counting = list(grad.shape[1:])
    shape_for_block_counting[-1] = 1
    num_blocks_per_CTA = math.prod(shape_for_block_counting)
    N = grad_shape[grad.ndim - 1]
    M = grad.numel() // N // num_blocks_per_CTA
    out_shape[0] = self_sizes[dim]
    grad_type = grad.dtype
    grad = grad.to(torch.float32)
    out = torch.zeros(out_shape, dtype=torch.float32, device=grad.device)
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )
    index_select_backward_kernel[grid](grad, out, M, N, num_blocks_per_CTA, index)
    out = out.to(grad_type)
    if grad_init == 1:
        out = out.squeeze(1)
    if dim != 0:
        order = [i for i in range(1, out.ndim)]
        order.insert(dim, 0)
        return out.permute(order)
    else:
        return out
