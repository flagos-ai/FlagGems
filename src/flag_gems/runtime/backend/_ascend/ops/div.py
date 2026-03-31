import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry, pointwise_dynamic

logger = logging.getLogger(f'flag_gems.runtime._ascend.ops.{__name__.split(".")[-1]}')

NUM_VECTOR_CORES = 48


# ============ true_divide (optimized manual kernel) ============

@libentry()
@triton.jit
def true_div_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    N,
    num_tasks,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_SUB: tl.constexpr,
    NCORE: tl.constexpr,
):
    pid = tl.program_id(0)
    for task_id in range(pid, num_tasks, NCORE):
        base_offset = task_id * BLOCK_SIZE
        for sub_idx in range(0, BLOCK_SIZE, BLOCK_SIZE_SUB):
            offsets = base_offset + sub_idx + tl.arange(0, BLOCK_SIZE_SUB)
            mask = offsets < N
            x = tl.load(x_ptr + offsets, mask=mask, care_padding=False)
            y = tl.load(y_ptr + offsets, mask=mask, care_padding=False)
            result = x / y
            tl.store(out_ptr + offsets, result, mask=mask)


@libentry()
@triton.jit
def true_div_scalar_r_kernel(
    x_ptr,
    out_ptr,
    y_scalar,
    N,
    num_tasks,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_SUB: tl.constexpr,
    NCORE: tl.constexpr,
):
    pid = tl.program_id(0)
    for task_id in range(pid, num_tasks, NCORE):
        base_offset = task_id * BLOCK_SIZE
        for sub_idx in range(0, BLOCK_SIZE, BLOCK_SIZE_SUB):
            offsets = base_offset + sub_idx + tl.arange(0, BLOCK_SIZE_SUB)
            mask = offsets < N
            x = tl.load(x_ptr + offsets, mask=mask, care_padding=False)
            result = x / y_scalar
            tl.store(out_ptr + offsets, result, mask=mask)


@libentry()
@triton.jit
def true_div_scalar_l_kernel(
    y_ptr,
    out_ptr,
    x_scalar,
    N,
    num_tasks,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_SUB: tl.constexpr,
    NCORE: tl.constexpr,
):
    pid = tl.program_id(0)
    for task_id in range(pid, num_tasks, NCORE):
        base_offset = task_id * BLOCK_SIZE
        for sub_idx in range(0, BLOCK_SIZE, BLOCK_SIZE_SUB):
            offsets = base_offset + sub_idx + tl.arange(0, BLOCK_SIZE_SUB)
            mask = offsets < N
            y = tl.load(y_ptr + offsets, mask=mask, care_padding=False)
            result = x_scalar / y
            tl.store(out_ptr + offsets, result, mask=mask)


def _compute_grid(N):
    BLOCK_SIZE = 1024
    BLOCK_SIZE_SUB = 1024
    num_tasks = triton.cdiv(N, BLOCK_SIZE)
    ncore = min(num_tasks, NUM_VECTOR_CORES)
    return ncore, num_tasks, BLOCK_SIZE, BLOCK_SIZE_SUB


def _true_div_tt(A, B, out=None):
    A = A.contiguous()
    B = B.contiguous()
    if out is None:
        out = torch.empty_like(A)
    N = A.numel()
    ncore, num_tasks, BLOCK_SIZE, BLOCK_SIZE_SUB = _compute_grid(N)
    true_div_kernel[(ncore,)](A, B, out, N, num_tasks, BLOCK_SIZE, BLOCK_SIZE_SUB, ncore)
    return out


def _true_div_ts(A, scalar, out=None):
    A = A.contiguous()
    if out is None:
        out = torch.empty_like(A)
    N = A.numel()
    ncore, num_tasks, BLOCK_SIZE, BLOCK_SIZE_SUB = _compute_grid(N)
    true_div_scalar_r_kernel[(ncore,)](A, out, scalar, N, num_tasks, BLOCK_SIZE, BLOCK_SIZE_SUB, ncore)
    return out


def _true_div_st(scalar, B, out=None):
    B = B.contiguous()
    if out is None:
        out = torch.empty_like(B)
    N = B.numel()
    ncore, num_tasks, BLOCK_SIZE, BLOCK_SIZE_SUB = _compute_grid(N)
    true_div_scalar_l_kernel[(ncore,)](B, out, scalar, N, num_tasks, BLOCK_SIZE, BLOCK_SIZE_SUB, ncore)
    return out


# Fallback pointwise_dynamic for broadcasting cases
@pointwise_dynamic(promotion_methods=[(0, 1, "INT_TO_FLOAT")])
@triton.jit
def true_div_func(x, y):
    return x / y


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "INT_TO_FLOAT")])
@triton.jit
def true_div_func_tensor_scalar(x, y):
    return x / y


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "INT_TO_FLOAT")])
@triton.jit
def true_div_func_scalar_tensor(x, y):
    return x / y


def true_divide(A, B):
    logger.debug("GEMS_ASCEND TRUE_DIVIDE")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if A.shape == B.shape and A.dtype.is_floating_point:
            return _true_div_tt(A, B)
        return true_div_func(A, B)
    elif isinstance(A, torch.Tensor):
        if A.dtype.is_floating_point:
            return _true_div_ts(A, B)
        return true_div_func_tensor_scalar(A, B)
    elif isinstance(B, torch.Tensor):
        if B.dtype.is_floating_point:
            return _true_div_st(A, B)
        return true_div_func_scalar_tensor(A, B)
    else:
        return torch.tensor(A / B)


def true_divide_out(A, B, out):
    logger.debug("GEMS_ASCEND TRUE_DIVIDE OUT")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if A.shape == B.shape and A.dtype.is_floating_point:
            return _true_div_tt(A, B, out=out)
        return true_div_func(A, B, out0=out)
    elif isinstance(A, torch.Tensor):
        if A.dtype.is_floating_point:
            return _true_div_ts(A, B, out=out)
        return true_div_func_tensor_scalar(A, B, out0=out)
    elif isinstance(B, torch.Tensor):
        if B.dtype.is_floating_point:
            return _true_div_st(A, B, out=out)
        return true_div_func_scalar_tensor(A, B, out0=out)
    else:
        return torch.tensor(A / B) if out is None else out.fill_(A / B)


def true_divide_(A, B):
    logger.debug("GEMS_ASCEND TRUE_DIVIDE_")
    if isinstance(B, torch.Tensor):
        if A.shape == B.shape and A.dtype.is_floating_point:
            return _true_div_tt(A, B, out=A)
        return true_div_func(A, B, out0=A)
    else:
        if A.dtype.is_floating_point:
            return _true_div_ts(A, B, out=A)
        return true_div_func_tensor_scalar(A, B, out0=A)


# ============ trunc_divide ============
# Ascend does not support tl.extra div_rz, so we implement trunc(x/y) directly.

@triton.jit
def _trunc(x):
    return tl.where(x >= 0, tl.floor(x), tl.ceil(x))


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_func(x, y):
    return _trunc(x / y)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_func_tensor_scalar(x, y):
    return _trunc(x / y)


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_func_scalar_tensor(x, y):
    return _trunc(x / y)


def trunc_divide(A, B):
    logger.debug("GEMS_ASCEND TRUNC_DIVIDE")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return trunc_div_func(A, B)
    elif isinstance(A, torch.Tensor):
        return trunc_div_func_tensor_scalar(A, B)
    elif isinstance(B, torch.Tensor):
        return trunc_div_func_scalar_tensor(A, B)
    else:
        return torch.tensor(A / B)


def trunc_divide_(A, B):
    logger.debug("GEMS_ASCEND TRUNC_DIVIDE_")
    if isinstance(B, torch.Tensor):
        return trunc_div_func(A, B, out0=A)
    else:
        return trunc_div_func_tensor_scalar(A, B, out0=A)


# ============ floor_divide ============
# Ascend does not support tl.extra div_rn/fmod, so we reimplement floor division.

@triton.jit
def _int_floordiv(x, y):
    r = x % y
    c1 = r != 0
    c2 = (x < 0) ^ (y < 0)
    return tl.where(c1 & c2, x // y - 1, x // y)


@triton.jit
def _float_floordiv(x, y):
    # Upcast to float32 for precision
    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)

    # Compute floor(x/y) directly
    q = x_f32 / y_f32
    floor_q = tl.math.floor(q)

    # Handle zero quotient sign: -0.0 when x and y have different signs
    different_sign = (x_f32 < 0.0) ^ (y_f32 < 0.0)
    floor_q = tl.where(floor_q == 0.0, tl.where(different_sign, -0.0, 0.0), floor_q)

    return floor_q.to(x.dtype)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def floor_div_func(x, y):
    if x.type.scalar.is_int() & x.type.scalar.is_int():
        return _int_floordiv(x, y)
    else:
        return _float_floordiv(x, y)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def floor_div_func_tensor_scalar(x, y):
    if x.type.scalar.is_int() & x.type.scalar.is_int():
        return _int_floordiv(x, y)
    else:
        return _float_floordiv(x, y)


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def floor_div_func_scalar_tensor(x, y):
    if x.type.scalar.is_int() & x.type.scalar.is_int():
        return _int_floordiv(x, y)
    else:
        return _float_floordiv(x, y)


def floor_divide(A, B):
    logger.debug("GEMS_ASCEND FLOOR_DIVIDE")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return floor_div_func(A, B)
    elif isinstance(A, torch.Tensor):
        return floor_div_func_tensor_scalar(A, B)
    elif isinstance(B, torch.Tensor):
        return floor_div_func_scalar_tensor(A, B)
    else:
        return torch.tensor(A // B)


def floor_divide_(A, B):
    logger.debug("GEMS_ASCEND FLOOR_DIVIDE_")
    if isinstance(B, torch.Tensor):
        return floor_div_func(A, B, out0=A)
    else:
        return floor_div_func_tensor_scalar(A, B, out0=A)


# ============ div_mode (dispatch by rounding_mode) ============

def div_mode(A, B, rounding_mode=None):
    if rounding_mode is None:
        return true_divide(A, B)
    elif rounding_mode == "trunc":
        return trunc_divide(A, B)
    elif rounding_mode == "floor":
        return floor_divide(A, B)
    else:
        msg = f"div expected rounding_mode to be one of None, 'trunc', or 'floor' but found {rounding_mode}."
        raise ValueError(msg)


def div_mode_(A, B, rounding_mode=None):
    if rounding_mode is None:
        return true_divide_(A, B)
    elif rounding_mode == "trunc":
        return trunc_divide_(A, B)
    elif rounding_mode == "floor":
        return floor_divide_(A, B)
    else:
        msg = f"div expected rounding_mode to be one of None, 'trunc', or 'floor' but found {rounding_mode}."
        raise ValueError(msg)
