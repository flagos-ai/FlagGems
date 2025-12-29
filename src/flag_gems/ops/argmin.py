import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.limits import get_dtype_max

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def argmin_kernel_1(
    inp,
    mid_value,
    mid_index,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M

    max_value = get_dtype_max(inp.type.element_ty)
    inp_val = tl.load(inp_ptrs, mask=mask, other=max_value)
    min_val, min_index = tl.min(inp_val, axis=0, return_indices=True)
    min_index = min_index + pid * BLOCK_SIZE
    mid_value_ptr = mid_value + pid
    min_index_ptr = mid_index + pid
    tl.store(mid_value_ptr, min_val)
    tl.store(min_index_ptr, min_index)


@libentry()
@triton.jit
def argmin_kernel_2(
    mid_value,
    mid_index,
    out,
    mid_size,
    BLOCK_MID: tl.constexpr,
):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid_value + offset
    mask = offset < mid_size
    max_value = get_dtype_max(mid_value.type.element_ty)
    mid_val = tl.load(mid_ptrs, mask=mask, other=max_value)
    index_val = tl.argmin(mid_val, axis=0)
    mid_index_ptrs = mid_index + index_val
    out_val = tl.load(mid_index_ptrs)
    tl.store(out, out_val)


def heur_block_n(args):
    return min(4096, triton.next_power_of_2(args["N"]))


@libentry()
@triton.heuristics(runtime.get_heuristic_config("argmin"))
@triton.jit
def argmin_kernel_opt_k1(
    inp,
    out_index,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tle.program_id(0)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    dtype = inp.type.element_ty
    acc_type = tl.float32 if dtype is tl.bfloat16 else dtype
    max_val = get_dtype_max(dtype)

    min_vals = tl.full([BLOCK_M], dtype=acc_type, value=max_val)
    argmin_vals = tl.full([BLOCK_M], dtype=tl.int64, value=0)
    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N + n_offset[None, :]
        inp_vals = tl.load(inp + offset, mask=True)

        local_min, local_argmin = tl.min(
            inp_vals, 1, return_indices=True, return_indices_tie_break_left=True
        )
        update = local_min < min_vals
        min_vals = tl.where(update, local_min, min_vals)
        argmin_vals = tl.where(update, start_n + local_argmin, argmin_vals)

    out_ptr = out_index + m_offset
    tl.store(out_ptr, argmin_vals, mask=True)


@triton.autotune(
    configs=[
        # 针对N=512, K=512的优化配置
        triton.Config(
            {"BLOCK_M": 4, "BLOCK_N": 512, "BLOCK_K": 32}, num_stages=3, num_warps=4
        ),
        # 针对N=1024, K=1024的优化配置
        triton.Config(
            {"BLOCK_M": 8, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=4, num_warps=8
        ),
        # 通用配置
        triton.Config(
            {"BLOCK_M": 8, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 8, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 4, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=4, num_warps=8
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def argmin_split_K_kernel_merged(
    inp,  # 输入指针：(M, N, K)
    out_index,  # 输出指针：(M, 1, K)
    M: tl.constexpr,  # M维度大小
    N: tl.constexpr,  # N维度大小（reduce dim）
    K: tl.constexpr,  # K维度大小
    dtype: tl.constexpr,  # 数据类型（float16/bfloat16/float32）
    BLOCK_M: tl.constexpr,  # M维度块大小
    BLOCK_N: tl.constexpr,  # N维度块大小
    BLOCK_K: tl.constexpr,  # K维度块大小
):
    # 1. 全局线程块ID：处理M和K维度的块索引
    pid_m = tle.program_id(0)  # M维度块索引
    pid_k = tle.program_id(1)  # K维度块索引

    # 2. 线程块内索引：每个线程处理1个(M,K)位置
    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]  # (BLOCK_M, 1)
    k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)[None, :]  # (1, BLOCK_K)

    # 边界检查
    m_mask = m < M
    k_mask = k < K
    mk_mask = m_mask & k_mask

    # 3. 数据类型处理：仅bfloat16提升至float32避免精度损失
    compute_dtype = tl.float32 if dtype == tl.bfloat16 else dtype
    max_val = get_dtype_max(compute_dtype)

    # 4. 初始化全局最小值和索引（寄存器存储）
    global_min = tl.full((BLOCK_M, BLOCK_K), max_val, dtype=compute_dtype)
    global_argmin = tl.full((BLOCK_M, BLOCK_K), 0, dtype=tl.int64)

    # 5. N维度分块处理（适应不同的N和BLOCK_N）
    for start_n in range(0, N, BLOCK_N):
        # 当前N块的索引
        n = start_n + tl.arange(0, BLOCK_N)
        n_mask = n < N

        # 计算内存偏移：确保连续访问（M×N×K + n×K + k）
        # 偏移 = m*N*K + n*K + k
        offset = m * N * K + n[:, None, None] * K + k[None, :, :]

        # 加载数据，使用边界掩码
        inp_vals = tl.load(
            inp + offset,
            mask=(m_mask & n_mask[:, None, None] & k_mask[None, :, :]),
            other=max_val,
        )
        inp_vals = inp_vals.to(compute_dtype)

        # 6. 局部min/argmin计算（沿N维度，dim=0）
        local_min, local_argmin = tl.min(
            inp_vals, 0, return_indices=True, return_indices_tie_break_left=True
        )
        # 转换为全局N索引（局部索引+块起始偏移）
        local_argmin += start_n

        # 7. 高效更新全局结果（向量指令，无分支开销）
        mask = local_min < global_min
        global_min = tl.where(mask, local_min, global_min)
        global_argmin = tl.where(mask, local_argmin, global_argmin)

    # 8. 存储结果，使用边界掩码
    out_offset = m * K + k  # (BLOCK_M, BLOCK_K)
    tl.store(out_index + out_offset, global_argmin, mask=mk_mask)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("argmin"))
@triton.jit
def argmin_kernel(
    inp,
    out_index,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # set offset
    pid_m = tle.program_id(0)
    pid_k = tle.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    dtype = inp.type.element_ty
    acc_type = tl.float32 if dtype is tl.bfloat16 else dtype
    max_value = get_dtype_max(dtype)
    min_values = tl.full([BLOCK_M], dtype=acc_type, value=max_value)
    argmin_values = tl.full([BLOCK_M], dtype=tl.int64, value=0)
    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
        mask = m_offset[:, None] < M and n_offset[None, :] < N
        inp_ptrs = inp + offset
        inp_vals = tl.load(inp_ptrs, mask=mask, other=max_value)
        # tl.bfloat is promoted to tl.float32 by tl.min
        local_min, local_argmin = tl.min(
            inp_vals, 1, return_indices=True, return_indices_tie_break_left=True
        )
        # if return indices is not supported, call a tl.argmin in addition
        # local_argmin = tl.argmin(inp_vals, 1)
        update = local_min < min_values
        min_values = tl.where(update, local_min, min_values)
        argmin_values = tl.where(update, start_n + local_argmin, argmin_values)

    offset_index = m_offset * K + pid_k
    out_index_ptrs = out_index + offset_index
    mask1 = m_offset < M
    tl.store(out_index_ptrs, argmin_values, mask=mask1)


def argmin(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS ARGMIN")
    if dim is None:
        M = inp.numel()
        if dtype is None:
            dtype = inp.dtype
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
        mid_size = triton.cdiv(M, block_size)
        block_mid = triton.next_power_of_2(mid_size)

        mid_value = torch.empty((mid_size,), dtype=dtype, device=inp.device)
        mid_index = torch.empty((mid_size,), dtype=torch.int64, device=inp.device)
        if keepdim:
            shape = list(inp.shape)
            for i in range(0, inp.dim()):
                shape[i] = 1
            out = torch.empty(shape, dtype=torch.int64, device=inp.device)
        else:
            out = torch.empty([], dtype=torch.int64, device=inp.device)

        with torch_device_fn.device(inp.device):
            argmin_kernel_1[(mid_size, 1, 1)](
                inp,
                mid_value,
                mid_index,
                M,
                block_size,
            )
            argmin_kernel_2[(1, 1, 1)](
                mid_value,
                mid_index,
                out,
                mid_size,
                block_mid,
            )
        return out
    else:
        assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
        shape = inp.shape
        dim = dim % inp.ndim
        N = shape[dim]
        M = math.prod(shape[:dim])
        K = inp.numel() // M // N
        inp = inp.contiguous()

        shape_list = list(shape)
        shape_list[dim] = 1
        out_index = torch.empty(shape_list, dtype=torch.int64, device=inp.device)
        if not keepdim:
            out_index = torch.squeeze(out_index, dim)

        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            K,
        )
        if K == 1 and inp.dtype != torch.int32 and inp.dtype != torch.int16:
            with torch_device_fn.device(inp.device):
                argmin_kernel_opt_k1[grid](
                    inp,
                    out_index,
                    M,
                    N,
                )

        else:
            torch2triton_dtype = {
                torch.float16: tl.float16,
                torch.bfloat16: tl.bfloat16,
                torch.float32: tl.float32,
            }
            # 泛化支持其他N和K的组合
            if (
                (N % 64 == 0 or N == 512)
                and (K % 32 == 0)
                and M % 8 == 0
                and inp.dtype != torch.int32
                and inp.dtype != torch.int16
            ):
                triton_dtype = torch2triton_dtype[inp.dtype]
                grid_for_split_K = (
                    triton.cdiv(M, 8),
                    triton.cdiv(K, 32),
                )  # 使用默认参数计算grid
                with torch_device_fn.device(inp.device):
                    argmin_split_K_kernel_merged[grid_for_split_K](
                        inp,
                        out_index,
                        M,
                        N,
                        K,
                        dtype=triton_dtype,
                    )
            else:
                with torch_device_fn.device(inp.device):
                    argmin_kernel[grid](
                        inp,
                        out_index,
                        M,
                        N,
                        K,
                    )

        return out_index
