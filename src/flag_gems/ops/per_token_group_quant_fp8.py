import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.device_info import get_device_capability

if torch_device_fn.is_available() and get_device_capability() >= (9, 0):
    SUPPORTED_FP8_DTYPE = torch.float8_e4m3fn
else:
    SUPPORTED_FP8_DTYPE = torch.float32

logger = logging.getLogger(__name__)

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
    groups_per_program: tl.constexpr,
):
    groups_per_row = y_num_columns // group_size
    pid = tl.program_id(0)
  
    pairs_per_row = groups_per_row // groups_per_program
    row = pid // pairs_per_row
    pair_id = pid % pairs_per_row

    group0 = pair_id * groups_per_program
    g0 = row * groups_per_row + group0
    base = y_ptr + row * y_row_stride
    cols = tl.arange(0, BLOCK)
    mask = cols < group_size

    for i in tl.static_range(0,groups_per_program):
            
        group = group0 + i
        g = g0 + i
        y_ptr_g = base + group * group_size
        y_q_ptr_g = y_q_ptr + g * group_size
        y_s_ptr_g = y_s_ptr + g
        y = tl.load(y_ptr_g + cols, mask=mask, other=0.0).to(tl.float32)

        abs_g = tl.abs(y)
        max_g = tl.max(abs_g)
        
        y_s = tl.maximum(max_g, eps) / fp8_max

        if scale_ue8m0:
            y_s = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s), 1e-10))))
          
        y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)
    
        tl.store(y_q_ptr_g + cols, y_q, mask=mask)
        tl.store(y_s_ptr_g, y_s)
        
    
#kernel优化版本M/8
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
    groups_per_program: tl.constexpr,
):
    groups_per_row = y_num_columns // group_size
    pid = tl.program_id(0)
  
    pairs_per_row = groups_per_row // groups_per_program
    row = pid // pairs_per_row
    pair_id = pid % pairs_per_row

    group0 = pair_id * groups_per_program
    g0 = row * groups_per_row + group0
    base = y_ptr + row * y_row_stride
    cols = tl.arange(0, BLOCK)
    mask = cols < group_size

    for i in tl.static_range(0,groups_per_program):
        group = group0 + i
        g = g0 + i
        y_ptr_g = base + group * group_size
        y_q_ptr_g = y_q_ptr + g * group_size
        y_s_ptr_g = y_s_ptr + group * y_s_col_stride + row
        y = tl.load(y_ptr_g + cols, mask=mask, other=0.0).to(tl.float32)

        abs_g = tl.abs(y)
        max_g = tl.max(abs_g)

        y_s = tl.maximum(max_g, eps) / fp8_max

        if scale_ue8m0:
            y_s = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s), 1e-10))))

        y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)
    
        tl.store(y_q_ptr_g + cols, y_q, mask=mask)
        tl.store(y_s_ptr_g, y_s)
        
    

def per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: Optional[torch.dtype] = None,
    column_major_scales: bool = False,
    scale_ue8m0: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logger.debug("GEMS PER TOKEN GROUP QUANT FP8")
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
    def Groups_per_program(x, group_size) -> int:
        if (x.shape[-1] // group_size) % 8 == 0:
            return 8
        elif (x.shape[-1] // group_size) % 4 == 0:
            return 4
        elif (x.shape[-1] // group_size) % 2 == 0:
            return 2
        else:
            return 1
        
    groups_per_program = Groups_per_program(x, group_size)
    

    # print("HIT per_token_group_quant_fp8")
    if column_major_scales:
        _per_token_group_quant_fp8_colmajor[(M // groups_per_program,)](
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
            groups_per_program=groups_per_program,
        )
    else: 
        _per_token_group_quant_fp8[(M // groups_per_program,)](
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
            groups_per_program=groups_per_program,
        )
       
    return x_q, x_s
