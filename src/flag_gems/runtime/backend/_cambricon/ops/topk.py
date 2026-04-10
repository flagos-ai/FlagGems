import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.topk import topk_stage1_kernel, topk_stage2_kernel
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils.triton_version_utils import HAS_TLE

if HAS_TLE:
    import triton.experimental.tle.language as tle_gpu
else:
    tle_gpu = None

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
    key=["k", "N", "M", "BLOCK_M"],
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


if HAS_TLE:

    @triton.jit
    def _get_topmask_and_fullmask(x):
        tl.static_assert(
            x.dtype.is_int_unsigned(),
            "floating-point value must be passed as bits",
        )
        tm: tl.constexpr = 1 << (-1 + x.dtype.primitive_bitwidth)
        fm: tl.constexpr = (1 << x.dtype.primitive_bitwidth) - 1
        tm_arr = tl.full(x.shape, tm, dtype=x.dtype)
        fm_arr = tl.full(x.shape, fm, dtype=x.dtype)
        return tm_arr, fm_arr

    @triton.jit
    def _fpval_to_key_with_nan(x, x_bits):
        tm, fm = _get_topmask_and_fullmask(x_bits)
        mask = tl.where((x_bits & tm) != 0, fm, tm)
        key = x_bits ^ mask
        return tl.where(x == x, key, fm)

    @triton.jit
    def _key_to_fpval(x):
        tm, fm = _get_topmask_and_fullmask(x)
        mask = tl.where((x & tm) != 0, tm, fm)
        return x ^ mask

    @libentry()
    @triton.jit
    def topk_kernel_radix_tle(
        X,
        Yv,
        Yi,
        stride_xm,
        stride_ym,
        n_cols,
        K: tl.constexpr,
        K_PAD: tl.constexpr,
        BLOCK_N: tl.constexpr,
        RADIX_BITS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        x_dtype = X.dtype.element_ty
        x_nbits: tl.constexpr = x_dtype.primitive_bitwidth
        if x_nbits < 16:
            y_nbits: tl.constexpr = 32
        else:
            y_nbits: tl.constexpr = x_nbits * 2
        x_utype = tl.dtype(f"uint{x_nbits}")
        x_ultype = tl.dtype(f"uint{y_nbits}")

        RADIX_SIZE: tl.constexpr = 1 << RADIX_BITS
        RADIX_MASK: tl.constexpr = RADIX_SIZE - 1
        bins = tl.arange(0, RADIX_SIZE)
        one = tl.full([BLOCK_N], 1, tl.int32)

        desired = tl.full((), 0, dtype=x_utype)
        desired_mask = tl.full((), 0, dtype=x_utype)
        k_to_find = tl.full((), K, dtype=tl.int32)
        n_tiles = tl.cdiv(n_cols, BLOCK_N)

        smem_counts = tle_gpu.gpu.alloc(
            [RADIX_SIZE],
            dtype=tl.int32,
            layout=None,
            scope=tle_gpu.gpu.smem,
            nv_mma_shared_layout=False,
        )
        smem_count_ptrs = tle_gpu.gpu.local_ptr(smem_counts, (bins,))

        for digit_pos in tl.static_range(x_nbits - RADIX_BITS, -1, -RADIX_BITS):
            tl.store(smem_count_ptrs, tl.zeros([RADIX_SIZE], dtype=tl.int32))
            for t in tl.range(0, n_tiles):
                offs_n = t * BLOCK_N + tl.arange(0, BLOCK_N)
                mask_n = offs_n < n_cols
                x_ptrs = X + pid * stride_xm + offs_n
                x = tl.load(x_ptrs, mask=mask_n, other=float("-inf"))
                x_bits = x.to(x_utype, bitcast=True)
                x_key = _fpval_to_key_with_nan(x, x_bits)
                matches = (x_key & desired_mask) == desired
                digit = ((x_key >> digit_pos) & RADIX_MASK).to(tl.int32)
                valid = mask_n & matches
                count_addrs = tle_gpu.gpu.local_ptr(smem_counts, (digit,))
                tl.atomic_add(count_addrs, one, mask=valid, sem="relaxed", scope="cta")

            counts = tl.load(smem_count_ptrs)

            cumsum_desc = tl.cumsum(counts, axis=0, reverse=True)
            tl.store(smem_count_ptrs, cumsum_desc)

            selected_scalar = 0
            counts_gt_scalar = 0
            found = 0
            for rev in tl.static_range(RADIX_SIZE):
                d = RADIX_SIZE - 1 - rev
                cum_d = tl.load(tle_gpu.gpu.local_ptr(smem_counts, (d,)))
                if d + 1 < RADIX_SIZE:
                    cum_next = tl.load(tle_gpu.gpu.local_ptr(smem_counts, (d + 1,)))
                else:
                    cum_next = 0
                take = (found == 0) & (cum_d >= k_to_find) & (cum_next < k_to_find)
                selected_scalar = tl.where(take, d, selected_scalar)
                counts_gt_scalar = tl.where(take, cum_next, counts_gt_scalar)
                found = tl.where(take, 1, found)

            selected_u = selected_scalar.to(x_utype)
            desired = desired | (selected_u << digit_pos)
            desired_mask = desired_mask | (
                tl.full((), RADIX_MASK, dtype=x_utype) << digit_pos
            )
            k_to_find = k_to_find - counts_gt_scalar

        thr_key = desired

        min_val = tl.full((), float("-inf"), tl.float32).to(x_dtype)
        min_bits = min_val.to(x_utype, bitcast=True)
        min_key = _fpval_to_key_with_nan(min_val, min_bits)
        min_packed = min_key.to(x_ultype) << 16
        offs_k = tl.arange(0, K_PAD)

        smem_selected = tle_gpu.gpu.alloc(
            [K_PAD],
            dtype=x_ultype,
            layout=None,
            scope=tle_gpu.gpu.smem,
            nv_mma_shared_layout=False,
        )
        smem_selected_ptrs = tle_gpu.gpu.local_ptr(smem_selected, (offs_k,))
        tl.store(smem_selected_ptrs, tl.full([K_PAD], min_packed, dtype=x_ultype))

        smem_write_count = tle_gpu.gpu.alloc(
            [1],
            dtype=tl.int32,
            layout=None,
            scope=tle_gpu.gpu.smem,
            nv_mma_shared_layout=False,
        )
        tl.store(tle_gpu.gpu.local_ptr(smem_write_count, (0,)), 0)
        write_count_ptrs = tle_gpu.gpu.local_ptr(
            smem_write_count, (tl.zeros([BLOCK_N], dtype=tl.int32),)
        )

        for t in tl.range(0, n_tiles):
            offs_n = t * BLOCK_N + tl.arange(0, BLOCK_N)
            mask_n = offs_n < n_cols
            x_ptrs = X + pid * stride_xm + offs_n
            x = tl.load(x_ptrs, mask=mask_n, other=float("-inf"))
            x_bits = x.to(x_utype, bitcast=True)
            x_key = _fpval_to_key_with_nan(x, x_bits)
            idx_key = (n_cols - offs_n).to(x_ultype)
            packed = (x_key.to(x_ultype) << 16) | idx_key
            take_gt = mask_n & (x_key > thr_key)
            pos = tl.atomic_add(
                write_count_ptrs, one, mask=take_gt, sem="relaxed", scope="cta"
            )
            write_mask = take_gt & (pos < K_PAD)
            dst_ptrs = tle_gpu.gpu.local_ptr(smem_selected, (pos.to(tl.int32),))
            tl.store(dst_ptrs, packed, mask=write_mask)

        for t in tl.range(0, n_tiles):
            offs_n = t * BLOCK_N + tl.arange(0, BLOCK_N)
            mask_n = offs_n < n_cols
            x_ptrs = X + pid * stride_xm + offs_n
            x = tl.load(x_ptrs, mask=mask_n, other=float("-inf"))
            x_bits = x.to(x_utype, bitcast=True)
            x_key = _fpval_to_key_with_nan(x, x_bits)
            idx_key = (n_cols - offs_n).to(x_ultype)
            packed = (x_key.to(x_ultype) << 16) | idx_key
            take_eq = mask_n & (x_key == thr_key)
            pos = tl.atomic_add(
                write_count_ptrs, one, mask=take_eq, sem="relaxed", scope="cta"
            )
            write_mask = take_eq & (pos < K_PAD)
            dst_ptrs = tle_gpu.gpu.local_ptr(smem_selected, (pos.to(tl.int32),))
            tl.store(dst_ptrs, packed, mask=write_mask)

        selected_packed = tl.load(smem_selected_ptrs)

        topk = tl.sort(selected_packed, dim=0, descending=True)
        idx_mask = tl.full(topk.shape, (1 << 16) - 1, dtype=topk.dtype)
        idx_raw = (topk & idx_mask).to(tl.uint32)
        y_indices = (n_cols - idx_raw.to(tl.int32)).to(tl.int32)
        y_values_raw = (topk >> 16).to(x_utype)
        y_values = _key_to_fpval(y_values_raw).to(x_dtype, bitcast=True)

        mask_k = offs_k < K
        yv_ptrs = Yv + pid * stride_ym + offs_k
        yi_ptrs = Yi + pid * stride_ym + offs_k
        tl.store(yv_ptrs, y_values, mask=mask_k)
        tl.store(yi_ptrs, y_indices, mask=mask_k)


def topk(x, k, dim=-1, largest=True, sorted=True):
    logger.debug("GEMS_CAMBRICON TOPK")
    # If dim equals to last dim, we set it to -1.
    if dim < 0:
        dim = dim + x.ndim

    assert dim == x.ndim - 1, "Currently only support topk in last dimension"
    assert sorted, "Currently only support sorted == True"

    # Early return for k=0 to avoid Triton kernel compilation error.
    # Triton's tl.arange(0, BLOCK_SIZE) requires BLOCK_SIZE > 0.
    # When k=0, stage2_elem_cnt becomes 0, leading to BLOCK_SIZE=0.
    if k == 0:
        out_shape = list(x.shape[:-1]) + [0]
        return (
            torch.empty(out_shape, device=x.device, dtype=x.dtype),
            torch.empty(out_shape, device=x.device, dtype=torch.int64),
        )

    descending = True
    if not largest:
        descending = False

    topk_elem_cnt = x.shape[dim]
    batch_size = math.prod(x.shape) // topk_elem_cnt
    out_shape = x.shape[:-1] + (k,)

    if (
        HAS_TLE
        and sorted
        and descending
        and x.is_cuda
        and x.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and topk_elem_cnt <= 65535
        and triton.next_power_of_2(k) <= 1024
    ):
        k_pad = triton.next_power_of_2(k)
        out_shape = x.shape[:-1] + (k,)
        y_vals = torch.empty(out_shape, device=x.device, dtype=x.dtype)
        y_idx = torch.empty(out_shape, device=x.device, dtype=torch.int32)
        block_n_radix = max(k_pad, min(512, triton.next_power_of_2(topk_elem_cnt)))
        block_n_radix = min(block_n_radix, 1024)

        x_2d = x.reshape(batch_size, topk_elem_cnt)
        with torch_device_fn.device(x.device):
            topk_kernel_radix_tle[(batch_size,)](
                x_2d,
                y_vals,
                y_idx,
                x_2d.stride(0),
                y_vals.stride(0),
                topk_elem_cnt,
                K=k,
                K_PAD=k_pad,
                BLOCK_N=block_n_radix,
                RADIX_BITS=4,
                num_warps=4,
                num_stages=1,
            )
        return (y_vals, y_idx.to(torch.int64))

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
