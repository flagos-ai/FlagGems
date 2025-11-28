import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.topk import topk_stage1_kernel, topk_stage2_kernel
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner

from ..utils import TOTAL_CORE_NUM

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


@triton.jit
def get_topk_bubble_res(
    buffer, buffer_ind, k, axis, mask_val, DESCENDING, BLOCK_M, BLOCK_N
):
    kep_buffer_n = buffer
    topk_buffer_index_n = buffer_ind
    ret = tl.empty([BLOCK_M, k], dtype=buffer.dtype)
    ret_ind = tl.empty([BLOCK_M, k], dtype=buffer_ind.dtype)
    for k_ind in tl.range(0, k):
        if DESCENDING:
            sel_val, sel_index = tl.max(kep_buffer_n, axis=axis, return_indices=True)
        else:
            sel_val, sel_index = tl.min(kep_buffer_n, axis=axis, return_indices=True)

        if BLOCK_M > 1:
            mask_sel = tl.arange(0, BLOCK_N)[None, :] == sel_index[:, None]
            tep_sel_index_buffer = tl.where(mask_sel, topk_buffer_index_n, 0)
            sel_index_res = tl.max(tep_sel_index_buffer, axis=axis)
            sel_val_res = sel_val
            ret[:, k_ind] = sel_val_res
            ret_ind[:, k_ind] = sel_index_res

            # Update buffer.
            kep_buffer_n = tl.where(mask_sel, mask_val, kep_buffer_n)
        else:
            indices = sel_index[0]
            ret[:, k_ind] = sel_val
            ret_ind[:, k_ind] = topk_buffer_index_n[:, indices]
            # Update buffer.
            kep_buffer_n[:, indices] = mask_val
    return ret, ret_ind


BLOCK_BATCH = [1, 16]
BLOCK_N = [128, 512, 1024, 2048]


def topk_cfggen():
    num_stage = [1, 3]
    configs = [
        triton.Config({"TILE_M": m, "TILE_N": n}, num_warps=1, num_stages=s)
        for m in BLOCK_BATCH
        for n in BLOCK_N
        for s in num_stage
    ]
    return configs


def topk_config_prune(configs, named_args, **kwargs):
    k = named_args["k"]
    N = named_args["N"]
    block_m = named_args["BLOCK_M"]
    new_configs = []

    for config in configs:
        tile_n = config.kwargs["TILE_N"]
        tile_m = config.kwargs["TILE_M"]
        if tile_n < k or tile_m > block_m:
            continue
        if len(new_configs) >= 1:
            last_tn = new_configs[-1].kwargs["TILE_N"]
            last_tm = new_configs[-1].kwargs["TILE_M"]
            if tile_n > N and last_tn >= N and last_tm == tile_m:
                continue
        config.kwargs["TILE_M_NUM"] = triton.cdiv(block_m, tile_m)
        config.kwargs["TILE_N_NUM"] = triton.cdiv(N, tile_n)
        new_configs.append(config)

    if (N not in BLOCK_N) and (N <= max(BLOCK_N)):
        for tm in BLOCK_BATCH:
            new_configs.append(
                triton.Config(
                    {
                        "TILE_M": tm,
                        "TILE_N": N,
                        "TILE_M_NUM": triton.cdiv(block_m, tm),
                        "TILE_N_NUM": 1,
                    },
                    num_warps=1,
                    num_stages=3,
                )
            )
    return new_configs


@libentry()
@libtuner(
    configs=topk_cfggen(),
    key=["k", "N", "M", "BLOCK_M", "DESCENDING"],
    prune_configs_by={"early_config_prune": topk_config_prune},
)
@triton.jit
def topk_bubble_kernel(
    inp_ptr,
    out_ptr,
    out_index_ptr,
    k: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_M_NUM: tl.constexpr,
    TILE_N_NUM: tl.constexpr,
    DESCENDING: tl.constexpr,
):
    pid = tl.program_id(0)
    m_st = pid * BLOCK_M

    mask_val = _get_finfo_val(inp_ptr.dtype.element_ty, return_max=not DESCENDING)
    mask_val = mask_val.to(inp_ptr.dtype.element_ty)

    for m_block_ind in tl.range(0, TILE_M_NUM):
        m_iter_st = m_block_ind * TILE_M + m_st
        m_offset_val = m_iter_st + tl.arange(0, TILE_M)
        m_offset = m_offset_val[:, None]
        m_offset_mask = m_offset < M

        topk_buffer_n = tl.full(
            [TILE_M, TILE_N_NUM * k], value=mask_val, dtype=inp_ptr.dtype.element_ty
        )
        topk_buffer_index_n = tl.full(
            [TILE_M, TILE_N_NUM * k], value=0, dtype=out_index_ptr.dtype.element_ty
        )
        for n_block_ind in tl.range(0, TILE_N_NUM):
            n_st = n_block_ind * TILE_N
            n_offset = n_st + tl.arange(0, TILE_N)[None, :]
            n_offset_mask = n_offset < N

            inp_mask = m_offset_mask & n_offset_mask
            inp_ptrs = inp_ptr + m_offset * N + n_offset
            block_inp_val = tl.load(inp_ptrs, mask=inp_mask, other=mask_val)

            local_buffer, local_buffer_ind = get_topk_bubble_res(
                block_inp_val,
                n_offset.to(out_index_ptr.dtype.element_ty),
                k,
                1,
                mask_val,
                DESCENDING,
                TILE_M,
                TILE_N,
            )
            tep_index = n_block_ind * k
            topk_buffer_n[:, tep_index : tep_index + k] = local_buffer
            topk_buffer_index_n[:, tep_index : tep_index + k] = local_buffer_ind
        if TILE_N_NUM > 1:
            global_res, global_res_ind = get_topk_bubble_res(
                topk_buffer_n,
                topk_buffer_index_n,
                k,
                1,
                mask_val,
                DESCENDING,
                TILE_M,
                TILE_N_NUM * k,
            )
        else:
            global_res = topk_buffer_n
            global_res_ind = topk_buffer_index_n

        # Store topk.
        store_ptrs = m_offset * k + tl.arange(0, k)[None, :]
        store_mask = m_offset_mask
        tl.store(store_ptrs + out_ptr, global_res, store_mask)
        tl.store(store_ptrs + out_index_ptr, global_res_ind, store_mask)


def topk(x, k, dim=-1, largest=True, sorted=True):
    logger.debug("GEMS_CAMBRICON TOPK")
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
    out_shape = x.shape[:-1] + (k,)

    if k <= math.log2(topk_elem_cnt):
        logger.debug("GEMS_CAMBRICON TOPK USING BUBBLE")
        topk_out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
        topk_out_idx = torch.empty(out_shape, device=x.device, dtype=torch.int64)

        def grid_fn(meta):
            return (min(batch_size, TOTAL_CORE_NUM),)

        block_m = triton.cdiv(batch_size, TOTAL_CORE_NUM)
        topk_bubble_kernel[grid_fn](
            x,
            topk_out,
            topk_out_idx,
            k,
            batch_size,
            topk_elem_cnt,
            block_m,
            DESCENDING=descending,
        )
        return (topk_out, topk_out_idx)
    else:
        logger.debug("GEMS_CAMBRICON TOPK USING SORT")
        # Note(Zhengzekang): Maybe we should add a heuristic search in selecting a proper chunk size.
        if topk_elem_cnt < 1024:
            chunk_size = 256
        else:
            chunk_size = 1024

        # Note(Zhengzekang): We should promise chunk_size is larger than k.
        if chunk_size < k:
            chunk_size = triton.next_power_of_2(k)

        chunk_num = triton.cdiv(topk_elem_cnt, chunk_size)

        stage1_out = torch.empty(
            batch_size * chunk_num * k, device=x.device, dtype=x.dtype
        )
        stage1_out_idx = torch.empty(
            batch_size * chunk_num * k, device=x.device, dtype=torch.int64
        )

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
