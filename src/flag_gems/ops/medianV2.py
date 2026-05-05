import logging
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.ops.sort import sort_stable

logger = logging.getLogger(__name__)

MAX_RADIX_SELECT_N = 4096

@libentry()
@triton.jit
def median_radix_select_kernel(
    inp_ptr,
    out_val_ptr,
    out_idx_ptr,
    M,
    N,
    BLOCK_N: tl.constexpr,
):
    """
    Radix Select algorithm for finding the median without full sorting.
    """
    pid = tl.program_id(0)
    if pid >= M:
        return

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    row_ptr = inp_ptr + pid * N
    ctype = inp_ptr.dtype.element_ty
    
    vals = tl.load(row_ptr + cols, mask=mask)
    vals_f32 = vals.to(tl.float32)
    
    # Standardize -0.0 to 0.0 to match PyTorch's equality semantics
    vals_f32 = tl.where(vals_f32 == 0.0, 0.0, vals_f32)
    is_nan = vals_f32 != vals_f32
    
    u32 = vals_f32.to(tl.uint32, bitcast=True)
    # Standardize all NaNs to a single positive NaN bit pattern
    u32 = tl.where(is_nan, 0x7FC00000, u32)
    
    sign_mask = u32 & 0x80000000
    ordered = tl.where(sign_mask != 0, ~u32, u32 | 0x80000000)
    
    active = mask
    k = (N - 1) // 2
    current_k = k
    
    for b in tl.static_range(31, -1, -1):
        bit_mask = tl.cast(1, tl.uint32) << b
        bits = (ordered & bit_mask) != 0
        
        active_bits_0 = active & (~bits)
        count_0 = tl.sum(tl.where(active_bits_0, 1, 0))
        
        if current_k < count_0:
            active = active_bits_0
        else:
            active = active & bits
            current_k = current_k - count_0
            
    median_val_ordered = tl.max(tl.where(active, ordered, tl.zeros_like(ordered)))
    
    less_mask = mask & (ordered < median_val_ordered)
    c_less = tl.sum(tl.where(less_mask, 1, 0))
    target_eq_idx = k - c_less
    
    eq_mask = mask & (ordered == median_val_ordered)
    eq_cumsum = tl.cumsum(tl.where(eq_mask, 1, 0))
    
    target_mask = eq_mask & (eq_cumsum == target_eq_idx + 1)
    
    indices = tl.arange(0, BLOCK_N)
    idx_med = tl.sum(tl.where(target_mask, indices, tl.zeros_like(indices)))
    
    u32_med = tl.where(
        (median_val_ordered & 0x80000000) != 0,
        median_val_ordered & ~tl.cast(0x80000000, tl.uint32),
        ~median_val_ordered
    )
    val_med_f32 = u32_med.to(tl.float32, bitcast=True)
    val_med = val_med_f32.to(ctype)
    
    tl.store(out_val_ptr + pid, val_med)
    tl.store(out_idx_ptr + pid, tl.cast(idx_med, tl.int64))

def heur_block_m(args):
    return min(512, triton.next_power_of_2(args["M"]))

@libentry()
@triton.heuristics({"BLOCK_M": heur_block_m})
@triton.jit
def median_gather_kernel(
    sorted_val_ptr,
    sorted_idx_ptr,
    out_val_ptr,
    out_idx_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    row_ids = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = row_ids < M
    median_pos = (N - 1) // 2
    src_offsets = row_ids * N + median_pos

    v = tl.load(sorted_val_ptr + src_offsets, mask=mask)
    i = tl.load(sorted_idx_ptr + src_offsets, mask=mask)

    tl.store(out_val_ptr + row_ids, v, mask=mask)
    tl.store(out_idx_ptr + row_ids, i, mask=mask)

def medianV2_dim(inp, dim=None, keepdim=False):
    """
    medianV2 utilizing Radix Select for small/medium arrays to bypass full sorting.
    """
    logger.debug("GEMS MEDIAN V2 DIM")

    assert dim is not None, "median_dim requires a dim argument"
    assert inp.is_floating_point(), "median only supports floating-point tensors"

    if dim < 0:
        dim = dim + inp.ndim
    assert 0 <= dim < inp.ndim, f"dim {dim} is out of range for ndim={inp.ndim}"

    if dim != inp.ndim - 1:
        work = torch.movedim(inp, dim, -1).contiguous()
    else:
        work = inp.contiguous()

    N = work.shape[-1]
    M = work.numel() // N

    out_shape = list(work.shape[:-1])
    out_val = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
    out_idx = torch.empty(out_shape, dtype=torch.int64, device=inp.device)

    if N == 1:
        out_val.copy_(work.squeeze(-1))
        out_idx.zero_()
    elif N <= MAX_RADIX_SELECT_N:
        BLOCK_N = triton.next_power_of_2(N)
        grid = (M,)
        with torch_device_fn.device(inp.device):
            median_radix_select_kernel[grid](
                work,
                out_val,
                out_idx,
                M,
                N,
                BLOCK_N=BLOCK_N,
            )
    else:
        with torch_device_fn.device(inp.device):
            sorted_vals, sorted_idxs = sort_stable(work, stable=True, dim=-1)

        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
        with torch_device_fn.device(inp.device):
            median_gather_kernel[grid](
                sorted_vals,
                sorted_idxs,
                out_val,
                out_idx,
                M,
                N,
            )

    if keepdim:
        out_val = out_val.unsqueeze(dim)
        out_idx = out_idx.unsqueeze(dim)

    Median = namedtuple("median", ["values", "indices"])
    return Median(values=out_val, indices=out_idx)
