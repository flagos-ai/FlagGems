import logging
import math

import torch
import triton
import triton.language as tl
import triton.language.core as core

try:
    # TODO: Triton 2.1 does not implement _log2.
    # Remove the try-catch block once all vendors upgrade to a newer version of Triton.
    from triton.language.standard import _log2, zeros_like
except ImportError:
    pass

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))
_MIN_FLOAT32_VAL = tl.constexpr(torch.finfo(torch.float32).min)
_MAX_FLOAT32_VAL = tl.constexpr(torch.finfo(torch.float32).max)
_MIN_FLOAT16_VAL = tl.constexpr(torch.finfo(torch.float16).min)
_MAX_FLOAT16_VAL = tl.constexpr(torch.finfo(torch.float16).max)
_MIN_BFLOAT16_VAL = tl.constexpr(torch.finfo(torch.bfloat16).min)
_MAX_BFLOAT16_VAL = tl.constexpr(torch.finfo(torch.bfloat16).max)
_MIN_INT16_VAL = tl.constexpr(torch.iinfo(torch.int16).min)
_MAX_INT16_VAL = tl.constexpr(torch.iinfo(torch.int16).max)
_MIN_INT32_VAL = tl.constexpr(torch.iinfo(torch.int32).min)
_MAX_INT32_VAL = tl.constexpr(torch.iinfo(torch.int32).max)
_MIN_INT64_VAL = tl.constexpr(torch.iinfo(torch.int64).min)
_MAX_INT64_VAL = tl.constexpr(torch.iinfo(torch.int64).max)


@triton.jit
def _get_finfo_val(
    dtype,
    return_max,
):
    if dtype is tl.float32:
        if return_max:
            return _MAX_FLOAT32_VAL
        else:
            return _MIN_FLOAT32_VAL
    elif dtype is tl.float16:
        if return_max:
            return _MAX_FLOAT16_VAL
        else:
            return _MIN_FLOAT16_VAL
    elif dtype is tl.bfloat16:
        if return_max:
            return _MAX_BFLOAT16_VAL
        else:
            return _MIN_BFLOAT16_VAL


@triton.jit
def _get_iinfo_val(
    dtype,
    return_max,
):
    if dtype is tl.int16:
        if return_max:
            return _MAX_INT16_VAL
        else:
            return _MIN_INT16_VAL
    elif dtype is tl.int32:
        if return_max:
            return _MAX_INT32_VAL
        else:
            return _MIN_INT32_VAL
    elif dtype is tl.int64:
        if return_max:
            return _MAX_INT64_VAL
        else:
            return _MIN_INT64_VAL


@libentry()
@triton.jit
def topk_stage1_kernel(
    y_ptr,
    index_ptr,
    x_ptr,
    k,
    N: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    DESCENDING: tl.constexpr,
):
    cur_batch = tle.program_id(0)
    cur_chunk_idx = tle.program_id(1)
    chunk_num = tle.num_programs(1)

    y_ptr += cur_batch * chunk_num * k + cur_chunk_idx * k
    index_ptr += cur_batch * chunk_num * k + cur_chunk_idx * k

    chunk_offset = cur_chunk_idx * CHUNK_SIZE
    x_ptr += cur_batch * N + chunk_offset

    cols = tl.arange(0, CHUNK_SIZE)
    mask = (chunk_offset + cols) < N

    mask_val = _get_finfo_val(x_ptr.dtype.element_ty, return_max=not DESCENDING)
    x_val = tl.load(x_ptr + cols, mask=mask, other=mask_val).to(tl.float32)
    for k_idx in range(k):
        if DESCENDING:
            chunk_select_val = tl.max(x_val)
            chunk_select_idx = tl.argmax(x_val, axis=0)
        else:
            chunk_select_val = tl.min(x_val)
            chunk_select_idx = tl.argmin(x_val, axis=0)

        tl.store(y_ptr + k_idx, chunk_select_val)
        tl.store(index_ptr + k_idx, chunk_select_idx + chunk_offset)

        if DESCENDING:
            x_val = tl.where(
                cols == chunk_select_idx,
                _get_finfo_val(tl.float32, return_max=False),
                x_val,
            )
        else:
            x_val = tl.where(
                cols == chunk_select_idx,
                _get_finfo_val(tl.float32, return_max=True),
                x_val,
            )


"""
Note(Zhengzekang):
Refer from triton2.2 official `sort` implementation:
https://github.com/triton-lang/triton/blob/release/2.2.x/python/triton/language/standard.py#L392-L404
Just add indices to sort with values.
"""


@triton.jit
def _compare_and_swap(x, ids, flip, i: core.constexpr, n_dims: core.constexpr):
    n_outer: core.constexpr = x.numel >> n_dims
    shape: core.constexpr = [n_outer * 2**i, 2, 2 ** (n_dims - i - 1)]

    # tl.device_print("shape is: ", shape)
    y = core.reshape(x, shape)
    y_idx = core.reshape(ids, shape)

    # slice left/right with 'stride' 2**(n_dims - i - 1)
    mask = core.arange(0, 2)[None, :, None]
    left = core.broadcast_to(tl.sum(y * (1 - mask), 1)[:, None, :], shape).to(x.dtype)
    right = core.broadcast_to(tl.sum(y * mask, 1)[:, None, :], shape).to(x.dtype)
    left = core.reshape(left, x.shape)
    right = core.reshape(right, x.shape)

    left_idx = core.broadcast_to(tl.sum(y_idx * (1 - mask), 1)[:, None, :], shape).to(
        ids.dtype
    )
    right_idx = core.broadcast_to(tl.sum(y_idx * mask, 1)[:, None, :], shape).to(
        ids.dtype
    )
    left_idx = core.reshape(left_idx, ids.shape)
    right_idx = core.reshape(right_idx, ids.shape)

    # actual compare-and-swap
    if core.constexpr(x.dtype.primitive_bitwidth) == 8:
        idtype = core.int8
    elif core.constexpr(x.dtype.primitive_bitwidth) == 16:
        idtype = core.int16
    elif core.constexpr(x.dtype.primitive_bitwidth) == 32:
        idtype = core.int32
    elif core.constexpr(x.dtype.primitive_bitwidth) == 64:
        idtype = core.int64
    else:
        raise ValueError("Unsupported dtype")

    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)

    cond = (left > right) ^ flip
    ret = ix ^ core.where(cond, ileft ^ iright, zeros_like(ix))

    if core.constexpr(ids.dtype.primitive_bitwidth) == 8:
        idx_dtype = core.int8
    elif core.constexpr(ids.dtype.primitive_bitwidth) == 16:
        idx_dtype = core.int16
    elif core.constexpr(ids.dtype.primitive_bitwidth) == 32:
        idx_dtype = core.int32
    elif core.constexpr(ids.dtype.primitive_bitwidth) == 64:
        idx_dtype = core.int64
    else:
        raise ValueError("Unsupported dtype")

    ileft_idx = left_idx.to(idx_dtype, bitcast=True)
    iright_idx = right_idx.to(idx_dtype, bitcast=True)
    ix_idx = ids.to(idx_dtype, bitcast=True)
    ret_idx = ix_idx ^ core.where(cond, ileft_idx ^ iright_idx, zeros_like(ix_idx))

    return ret.to(x.dtype, bitcast=True), ret_idx.to(ids.dtype, bitcast=True)


@triton.jit
def _bitonic_merge(
    x, ids, stage: core.constexpr, order: core.constexpr, n_dims: core.constexpr
):
    """
    order_type 0 == ascending
    order_type 1 == descending
    order_type 2 == alternating
    """
    n_outer: core.constexpr = x.numel >> n_dims
    core.static_assert(stage <= n_dims)
    # flip denotes whether to re-arrange sub-sequences of elements in ascending or
    # descending order.
    # if flip = 00000000... then all elements will be re-arranged ascendingly at this stage
    # if flip = 00110011... then all the elements will be re-arranged alternatingly (with
    # a stride of 2) at this stage
    if order == 2:
        shape: core.constexpr = [n_outer * 2 ** (n_dims - 1 - stage), 2, 2**stage]
        flip = core.reshape(
            core.broadcast_to(core.arange(0, 2)[None, :, None], shape), x.shape
        )
    else:
        flip = order
    # perform `stage` rounds of `compare-and-swap`
    for i in core.static_range(stage):
        x, ids = _compare_and_swap(x, ids, flip, i + (n_dims - stage), n_dims)
    return x, ids


@triton.jit
def argsort(x, ids, dim: tl.constexpr, descending: core.constexpr):
    # handle default dimension or check that it is the most minor dim
    _dim: core.constexpr = dim
    n_dims: core.constexpr = _log2(x.shape[_dim])
    for i in core.static_range(1, n_dims + 1):
        x, ids = _bitonic_merge(x, ids, i, 2 if i < n_dims else descending, n_dims)
    return x, ids


@libentry()
@triton.jit
def topk_stage2_kernel(
    y_ptr,
    index_ptr,
    chunk_x,
    chunk_index,
    sort_dim: tl.constexpr,
    k: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DESCENDING: tl.constexpr,
):
    cur_batch = tle.program_id(0)
    chunk_x += cur_batch * N
    chunk_index += cur_batch * N
    y_ptr += cur_batch * k
    index_ptr += cur_batch * k

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    mask_val = _get_finfo_val(chunk_x.dtype.element_ty, return_max=not DESCENDING)
    mask_index_val = _MIN_INT32_VAL if DESCENDING else _MAX_INT32_VAL

    chunk_x_val = tl.load(chunk_x + cols, mask=mask, other=mask_val).to(tl.float32)
    chunk_index_val = tl.load(chunk_index + cols, mask=mask, other=mask_index_val).to(
        tl.int32
    )

    sorted_chunk_x, sorted_chunk_index = argsort(
        chunk_x_val, chunk_index_val, 0, descending=DESCENDING
    )
    tl.store(y_ptr + cols, sorted_chunk_x, mask=cols < k)
    tl.store(index_ptr + cols, sorted_chunk_index, mask=cols < k)


def topk(x, k, dim=-1, largest=True, sorted=True):
    logger.debug("GEMS TOPK")
    # If dim equals to last dim, we set it to -1.
    if dim < 0:
        dim = dim + x.ndim

    assert dim == x.ndim - 1, "Currently only support topk in last dimension"
    assert sorted, "Currently only support sorted == True"

    descending = True
    if not largest:
        descending = False

    topk_elem_cnt = x.shape[dim]
    batch_size = math.prod(x.shape) // topk_elem_cnt

    # Note(Zhengzekang): Maybe we should add a heuristic search in selecting a proper chunk size.
    if topk_elem_cnt < 1024:
        chunk_size = 256
    else:
        chunk_size = 1024

    # Note(Zhengzekang): We should promise chunk_size is larger than k.
    if chunk_size < k:
        chunk_size = triton.next_power_of_2(k)

    chunk_num = triton.cdiv(topk_elem_cnt, chunk_size)

    stage1_out = torch.empty(batch_size * chunk_num * k, device=x.device, dtype=x.dtype)
    stage1_out_idx = torch.empty(
        batch_size * chunk_num * k, device=x.device, dtype=torch.int64
    )

    out_shape = x.shape[:-1] + (k,)
    stage2_out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
    stage2_out_idx = torch.empty(out_shape, device=x.device, dtype=torch.int64)

    with torch_device_fn.device(x.device):
        topk_stage1_kernel[
            batch_size,
            chunk_num,
        ](
            stage1_out,  # pointer to the output
            stage1_out_idx,  # pointer to the output
            x,  # pointer to the input
            k,
            topk_elem_cnt,
            chunk_size,
            descending,
        )
    stage2_elem_cnt = chunk_num * k
    BLOCK_SIZE = triton.next_power_of_2(stage2_elem_cnt)

    with torch_device_fn.device(x.device):
        topk_stage2_kernel[batch_size,](
            stage2_out,
            stage2_out_idx,
            stage1_out,
            stage1_out_idx,
            dim,
            k,
            stage2_elem_cnt,
            BLOCK_SIZE,
            descending,
        )

    return (stage2_out, stage2_out_idx)
