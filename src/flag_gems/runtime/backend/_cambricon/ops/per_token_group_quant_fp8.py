import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from ..utils import MAX_GRID_SIZE_X

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))

if torch.cuda.is_available() and torch.cuda.get_device_capability() >= (9, 0):
    SUPPORTED_FP8_DTYPE = torch.float8_e4m3fn
else:
    SUPPORTED_FP8_DTYPE = torch.float32


@triton.jit
def _per_token_group_quant_fp8(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    y_num_columns,
    y_row_stride,
    eps,
    fp8_min,
    fp8_max,
    scale_ue8m0,
    BLOCK: tl.constexpr,
    M: tl.constexpr,
):
    groups_per_row = y_num_columns // group_size

    grid_0 = tl.num_programs(0)
    g_id = tl.program_id(0)
    while g_id < M:
        row = g_id // groups_per_row
        row_g_id = g_id % groups_per_row

        y_ptr_offset = (row * y_row_stride) + (row_g_id * group_size)
        y_q_ptr_offset = g_id * group_size
        y_s_ptr_offset = g_id

        cols = tl.arange(0, BLOCK)
        mask = cols < group_size

        y = tl.load(y_ptr + cols + y_ptr_offset, mask=mask, other=0.0).to(tl.float32)
        _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
        y_s = _absmax / fp8_max
        if scale_ue8m0:
            y_s = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s), 1e-10))))
        y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

        tl.store(y_q_ptr + cols + y_q_ptr_offset, y_q, mask=mask)
        tl.store(y_s_ptr + y_s_ptr_offset, y_s)
        g_id += grid_0


@triton.jit
def _per_token_group_quant_fp8_colmajor(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    y_num_columns,
    y_row_stride,
    y_s_col_stride,
    eps,
    fp8_min,
    fp8_max,
    scale_ue8m0,
    BLOCK: tl.constexpr,
    M: tl.constexpr,
):
    groups_per_row = y_num_columns // group_size
    grid_0 = tl.num_programs(0)
    g_id = tl.program_id(0)
    while g_id < M:
        row = g_id // groups_per_row
        group_id = g_id % groups_per_row

        y_ptr_offset = row * y_row_stride + group_id * group_size
        y_q_ptr_offset = g_id * group_size
        y_s_ptr_offset = group_id * y_s_col_stride + row

        cols = tl.arange(0, BLOCK)
        mask = cols < group_size

        y = tl.load(y_ptr + cols + y_ptr_offset, mask=mask, other=0.0).to(tl.float32)
        _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
        y_s = _absmax / fp8_max
        if scale_ue8m0:
            y_s = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s), 1e-10))))
        y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

        tl.store(y_q_ptr + cols + y_q_ptr_offset, y_q, mask=mask)
        tl.store(y_s_ptr + y_s_ptr_offset, y_s)
        g_id += grid_0


def per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: Optional[torch.dtype] = None,
    column_major_scales: bool = False,
    scale_ue8m0: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logger.debug("GEMS_CAMBRICON PER_TOKEN_GROUP_QUANT_FP8")
    # dtype: The dype of output tensor. Note that only `torch.float8_e4m3fn`
    fp8_dtype = SUPPORTED_FP8_DTYPE if dtype is None else dtype
    assert x.shape[-1] % group_size == 0, (
        f"the last dimension of `x` {x.shape[-1]} must be divisible "
        f"by `group_size` {group_size}"
    )
    assert x.stride(-1) == 1, "`x` groups must be contiguous"

    finfo = torch.finfo(fp8_dtype)
    fp8_min = finfo.min
    fp8_max = finfo.max

    x_q = torch.empty_like(x, device=x.device, dtype=fp8_dtype)
    M = x.numel() // group_size
    N = group_size

    if column_major_scales:
        shape = (x.shape[-1] // group_size,) + x.shape[:-1]
        x_s = torch.empty(shape, device=x.device, dtype=torch.float32).permute(-1, -2)
    else:
        shape = x.shape[:-1] + (x.shape[-1] // group_size,)
        x_s = torch.empty(shape, device=x.device, dtype=torch.float32)

    BLOCK = triton.next_power_of_2(N)
    num_warps = min(max(BLOCK // 256, 1), 8)
    num_stages = 1
    grid = min(M, MAX_GRID_SIZE_X // 4)
    if column_major_scales:
        _per_token_group_quant_fp8_colmajor[(grid,)](
            x,
            x_q,
            x_s,
            group_size,
            x.shape[1],
            x.stride(0),
            x_s.stride(1),
            eps,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
            scale_ue8m0=scale_ue8m0,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
            M=M,
        )
    else:
        _per_token_group_quant_fp8[(grid,)](
            x,
            x_q,
            x_s,
            group_size,
            x.shape[1],
            x.stride(0),
            eps,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
            scale_ue8m0=scale_ue8m0,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
            M=M,
        )

    return x_q, x_s
