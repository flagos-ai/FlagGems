import triton
import triton.language as tl

from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.limits import get_dtype_min


@triton.jit
def argmax_kernel_1(
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
    min_value = get_dtype_min(inp.type.element_ty)
    inp_val = tl.load(inp_ptrs, mask=mask, other=min_value)
    max_val, max_index = tl.max(inp_val, axis=0, return_indices=True)
    max_index = max_index + pid * BLOCK_SIZE
    mid_value_ptr = mid_value + pid
    max_index_ptr = mid_index + pid
    tl.store(mid_value_ptr, max_val)
    tl.store(max_index_ptr, max_index)


@triton.jit
def argmax_kernel_2(mid_value, mid_index, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid_value + offset
    mask = offset < mid_size
    min_value = get_dtype_min(mid_value.type.element_ty)
    mid_val = tl.load(mid_ptrs, mask=mask, other=min_value)
    index_val = tl.argmax(mid_val, axis=0)
    mid_index_ptrs = mid_index + index_val
    out_val = tl.load(mid_index_ptrs)
    tl.store(out, out_val)


@triton.jit
def argmax_kernel(
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
    min_value = get_dtype_min(dtype)
    max_values = tl.full([BLOCK_M], dtype=acc_type, value=min_value)
    argmax_values = tl.full([BLOCK_M], dtype=tl.int64, value=0)
    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
        mask = m_offset[:, None] < M and n_offset[None, :] < N
        inp_ptrs = inp + offset
        inp_vals = tl.load(inp_ptrs, mask=mask, other=min_value)
        local_max, local_argmax = tl.max(
            inp_vals, 1, return_indices=True, return_indices_tie_break_left=True
        )
        # if return indices is not supported, call a tl.argmax in addition
        # local_argmax = tl.argmax(inp_vals, 1)
        update = local_max > max_values
        max_values = tl.where(update, local_max, max_values)
        argmax_values = tl.where(update, start_n + local_argmax, argmax_values)

    offset_index = m_offset * K + pid_k
    out_index_ptrs = out_index + offset_index
    mask1 = m_offset < M
    tl.store(out_index_ptrs, argmax_values, mask=mask1)
