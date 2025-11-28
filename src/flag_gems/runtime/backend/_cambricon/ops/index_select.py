import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry, libtuner

from ..utils import MAX_NRAM_SIZE, TOTAL_CORE_NUM

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


def get_max_block_size(dtype_size):
    return MAX_NRAM_SIZE // 3 // dtype_size


def config_prune(configs, named_args, **kwargs):
    N = named_args["N"]
    dtype_size = named_args["dtype_size"]
    max_block_size = get_max_block_size(dtype_size)

    pruned_configs = []
    index_block_size = []
    for config in configs:
        bs = config.kwargs["BLOCK_SIZE"]
        ibs = (bs + N - 1) // N
        if ibs not in index_block_size and ibs * N <= max_block_size:
            index_block_size.append(ibs)
            pruned_configs.append(config)

    in_n_elements = named_args["in_n_elements"]

    # make sure at least one config is at the load-balance sweet point
    if in_n_elements % TOTAL_CORE_NUM == 0:
        bs = min(max(in_n_elements // TOTAL_CORE_NUM, 1) * N, max_block_size)
    else:
        bs = min(max(in_n_elements // TOTAL_CORE_NUM, 1) * N + 1, max_block_size)
    if (bs + N - 1) // N not in index_block_size:
        pruned_configs.append(
            triton.Config(kwargs={"BLOCK_SIZE": bs}, num_stages=1, num_warps=1)
        )

    return pruned_configs


@triton.jit
def ld_st_1(indices, N: tl.constexpr, weight_ptr, in_mask, in_offsets, out_ptr):
    weight_offsets = indices[:, None] * N + tl.arange(0, N)
    embedding_weight = tl.load(weight_ptr + weight_offsets, in_mask[:, None])
    out_offsets = in_offsets[:, None] * N + tl.arange(0, N)
    tl.store(out_ptr + out_offsets, embedding_weight, in_mask[:, None])


@libentry()
@libtuner(
    configs=[
        # [512, 65536]
        triton.Config(kwargs={"BLOCK_SIZE": 512 * 2**i}, num_stages=1, num_warps=1)
        for i in range(0, 8, 2)
    ],
    key=["N"],
    prune_configs_by={
        "early_config_prune": config_prune,
    },
)
@triton.jit
def one_batch_index_select_kernel(  # 2D
    out_ptr,
    in_ptr,
    in_n_elements,
    weight_ptr,
    N: tl.constexpr,
    dtype_size,
    inp_numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_jobs = tl.num_programs(axis=0)

    INDEX_BLOCK_SIZE: tl.constexpr = (BLOCK_SIZE + N - 1) // N

    step = num_jobs * INDEX_BLOCK_SIZE
    iters = tl.cdiv(in_n_elements, step)

    # TODO: remove dtype_size once contiguous DMA is ensured
    small_out = inp_numel.to(tl.int64) * dtype_size <= 2**31

    for i in tl.range(iters):
        iter_start = i * step
        iter_end = iter_start + step

        if iter_end <= in_n_elements:
            block_offset = iter_start + pid * INDEX_BLOCK_SIZE
            block_len = INDEX_BLOCK_SIZE
        else:
            rem_n_elements = in_n_elements - iter_start
            base_num = rem_n_elements // num_jobs
            remn_num = rem_n_elements % num_jobs
            extra_one = pid < remn_num

            block_offset = iter_start + (
                (base_num + 1) * pid if extra_one else (base_num * pid + remn_num)
            )
            block_len = base_num + extra_one

        in_offsets = block_offset + tl.arange(0, INDEX_BLOCK_SIZE)
        in_mask = in_offsets < (block_offset + block_len)
        indices = tl.load(in_ptr + in_offsets, in_mask, other=0.0)
        if indices.dtype != tl.int32 and small_out:
            indices_int32 = indices.to(tl.int32)
            ld_st_1(indices_int32, N, weight_ptr, in_mask, in_offsets, out_ptr)
        else:
            ld_st_1(indices, N, weight_ptr, in_mask, in_offsets, out_ptr)


def config_prune(configs, named_args, **kwargs):
    # TODO: bad perf when BLOCK_BATCH is 1
    batch_dim = max(named_args["batch_dim"], 2)
    index_dim = named_args["index_dim"]
    c_dim = named_args["c_dim"]
    dtype_size = named_args["dtype_size"]

    # difficult to include these critical configs while keeping number of configs small
    lb_block_batch_1 = triton.cdiv(batch_dim, TOTAL_CORE_NUM)
    lb_block_batch_2 = max(batch_dim // TOTAL_CORE_NUM, 1)
    lb_block_index_1 = triton.cdiv(index_dim, TOTAL_CORE_NUM)
    lb_block_index_2 = max(index_dim // TOTAL_CORE_NUM, 1)

    max_bs = get_max_block_size(dtype_size)

    block_batches = set([lb_block_batch_1, lb_block_batch_2, batch_dim])
    block_indices = set([lb_block_index_1, lb_block_index_2, index_dim])
    block_cs = set([c_dim, min(max_bs, c_dim)])

    new_configs = []
    for config in configs:
        block_batch = config.kwargs["BLOCK_BATCH"]
        block_index = config.kwargs["BLOCK_INDEX"]
        block_c = config.kwargs["BLOCK_C"]

        # to keep the autotune space small: if c_dim is not very large, don't split c
        block_c_max = 2048 * 5
        block_c = c_dim if c_dim <= block_c_max else block_c

        if block_batch <= batch_dim and block_index <= index_dim and block_c <= c_dim:
            block_batches.add(block_batch)
            block_indices.add(block_index)
            block_cs.add(block_c)

    for block_batch in block_batches:
        for block_index in block_indices:
            for block_c in block_cs:
                if block_batch * block_index * block_c <= max_bs:
                    new_configs.append(
                        triton.Config(
                            {
                                "BLOCK_BATCH": block_batch,
                                "BLOCK_INDEX": block_index,
                                "BLOCK_C": block_c,
                            },
                            num_warps=1,
                            num_stages=1,
                        )
                    )
    return new_configs


@triton.jit
def ld_st_2(
    inp,
    out,
    batch_offsets,
    index_offsets,
    c_offsets,
    inp_strides_0,
    inp_strides_1,
    out_strides_0,
    out_strides_1,
    index_cur,
    input_output_mask,
):
    input_offsets = (batch_offsets * inp_strides_0)[:, None, None] + (
        (index_cur * inp_strides_1)[:, None] + c_offsets[None, :]
    )[None, :, :]

    output_offsets = (batch_offsets * out_strides_0)[:, None, None] + (
        (index_offsets * out_strides_1)[:, None] + c_offsets[None, :]
    )[None, :, :]

    selected = tl.load(inp + input_offsets, mask=input_output_mask, other=0.0)
    tl.store(out + output_offsets, selected, mask=input_output_mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("index_select"),
    key=["batch_dim", "index_dim", "c_dim"],
    prune_configs_by={"early_config_prune": config_prune},
)
@triton.jit
def multi_batch_index_select_kernel(
    inp,
    index,
    out,
    batch_dim,
    select_dim,
    c_dim,
    index_dim,
    dtype_size,
    inp_numel,
    BLOCK_BATCH: tl.constexpr,
    BLOCK_INDEX: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_x = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)

    block_id_start = pid_x
    block_id_step = num_programs

    block_batch: tl.constexpr = BLOCK_BATCH
    block_index: tl.constexpr = BLOCK_INDEX
    block_c: tl.constexpr = BLOCK_C

    block_num_batch = tl.cdiv(batch_dim, block_batch)
    block_num_index = tl.cdiv(index_dim, block_index)
    block_num_c = tl.cdiv(c_dim, block_c)

    block_num_total = block_num_batch * block_num_index * block_num_c

    inp_strides_0, inp_strides_1 = [select_dim * c_dim, c_dim]
    out_strides_0, out_strides_1 = [index_dim * c_dim, c_dim]
    block_strides_0, block_strides_1 = [block_num_index * block_num_c, block_num_c]

    # TODO: remove dtype_size once contiguous DMA is ensured
    small_out = inp_numel.to(tl.int64) * dtype_size <= 2**31

    for block_id in tl.range(block_id_start, block_num_total, block_id_step):
        block_id_batch = block_id // block_strides_0
        block_id_index = (block_id // block_strides_1) % block_num_index
        block_id_c = block_id % block_num_c

        # arange requires constexpr
        batch_offsets = block_id_batch * block_batch + tl.arange(0, block_batch)
        batch_mask = batch_offsets < batch_dim

        index_offsets = block_id_index * block_index + tl.arange(0, block_index)
        index_mask = index_offsets < index_dim

        c_offsets = block_id_c * block_c + tl.arange(0, block_c)
        c_mask = c_offsets < c_dim

        input_output_mask = (
            batch_mask[:, None, None]
            and (index_mask[:, None] and c_mask[None, :])[None, :, :]
        )

        index_cur = tl.load(index + index_offsets, mask=index_mask, other=0)
        # TODO: remove dtype_size once contiguous DMA is ensured
        if index.dtype != tl.int32 and small_out:
            index_cur_int32 = index_cur.to(tl.int32)
            ld_st_2(
                inp,
                out,
                batch_offsets,
                index_offsets,
                c_offsets,
                inp_strides_0,
                inp_strides_1,
                out_strides_0,
                out_strides_1,
                index_cur_int32,
                input_output_mask,
            )
        else:
            ld_st_2(
                inp,
                out,
                batch_offsets,
                index_offsets,
                c_offsets,
                inp_strides_0,
                inp_strides_1,
                out_strides_0,
                out_strides_1,
                index_cur,
                input_output_mask,
            )


def index_select(inp, dim, index):
    logger.debug("GEMS_CAMBRICON INDEX SELECT")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index.ndim <= 1, "Index should have dimension 1 or 0"
    # TODO: index is on device, should it be a kernel (like cnnl __assert_fail__) to check this?
    assert ((i >= 0 and i < inp.size(dim)) for i in index), "Index out of range"

    # TODO: make sure input is contiguous

    if index.ndim == 0:
        index = index.unsqueeze(0)
    dim = dim % inp.ndim
    inp_shape = list(inp.shape)
    index_dim = index.numel()

    # input  [batch_dim, select_dim, c_dim]
    # output [batch_dim, index_dim, c_dim]
    inp = inp.contiguous()
    index = index.contiguous()
    inp_numel = inp.numel()
    batch_dim = math.prod(inp_shape[:dim])
    select_dim = inp_shape[dim]
    c_dim = math.prod(inp_shape[(dim + 1) :])

    out_shape = inp_shape
    out_shape[dim] = index_dim
    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    if torch.is_floating_point(inp):
        dtype_size = torch.finfo(inp.dtype).bits // 8
    else:
        dtype_size = torch.iinfo(inp.dtype).bits // 8

    if batch_dim == 1 and c_dim <= get_max_block_size(dtype_size):
        # ram: (input, output), half, extra
        # 2D, not split c_dim
        def grid_fn(meta):
            index_block_size_grid = max(meta["BLOCK_SIZE"] // c_dim, 1)
            index_block_num = triton.cdiv(index_dim, index_block_size_grid)
            return (min(index_block_num, TOTAL_CORE_NUM),)

        one_batch_index_select_kernel[grid_fn](
            out, index, index_dim, inp, c_dim, dtype_size, inp_numel
        )
    else:
        grid = lambda meta: (
            min(
                triton.cdiv(batch_dim, meta["BLOCK_BATCH"])
                * triton.cdiv(index_dim, meta["BLOCK_INDEX"])
                * triton.cdiv(c_dim, meta["BLOCK_C"]),
                TOTAL_CORE_NUM,
            ),
        )
        multi_batch_index_select_kernel[grid](
            inp,
            index,
            out,
            batch_dim,
            select_dim,
            c_dim,
            index_dim,
            dtype_size,
            inp_numel,
        )
    return out
