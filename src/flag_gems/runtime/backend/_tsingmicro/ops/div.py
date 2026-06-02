import os
import logging
import torch
import triton
import triton.language as tl
from flag_gems.utils import pointwise_dynamic
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.triton_lang_extension import div_rn, div_rz, fmod, trunc
from flag_gems.runtime import torch_device_fn


TOTAL_CORE_NUM = torch_device_fn.get_device_properties().multi_processor_count

logger = logging.getLogger(__name__)
_BLOCK_N = 1024 * 4  # 4096


def _true_div_out_dtype(a: torch.Tensor, b: torch.Tensor) -> torch.dtype:
    """Match ``true_div_func`` ``INT_TO_FLOAT`` promotion."""
    if a.is_floating_point() or b.is_floating_point():
        return torch.promote_types(a.dtype, b.dtype)
    return torch.float32


@triton.jit
def _div_row_broadcast_kernel_2d(
    full_ptr,
    brc_ptr,
    out_ptr,
    stride_full0,
    stride_full1,
    stride_brc0,
    stride_brc1,
    stride_out0,
    stride_out1,
    n_cols,
    num_blocks,
    FULL_IS_DIVIDEND: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tle.program_id(0)
    brc = tl.load(brc_ptr + row * stride_brc0 + 0 * stride_brc1)
    for blk in range(num_blocks):
        cols = blk * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = cols < n_cols
        v = tl.load(
            full_ptr + row * stride_full0 + cols * stride_full1,
            mask=mask,
            other=0.0,
        )
        if FULL_IS_DIVIDEND:
            out = v / brc
        else:
            out = brc / v
        tl.store(
            out_ptr + row * stride_out0 + cols * stride_out1,
            out,
            mask=mask,
        )


@triton.jit
def _div_row_broadcast_kernel_3d(
    full_ptr,
    brc_ptr,
    out_ptr,
    stride_f0,
    stride_f1,
    stride_f2,
    stride_b0,
    stride_b1,
    stride_b2,
    stride_o0,
    stride_o1,
    stride_o2,
    d0,
    d1,
    iters,
    n_cols,
    num_blocks,
    FULL_IS_DIVIDEND: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)
    num_programs = tle.num_programs(0)
    for i in range(iters):
        row = i * num_programs + pid
        if row <= d0 * d1:
            i0 = row // d1
            i1 = row % d1

            brc = tl.load(brc_ptr + i0 * stride_b0 + i1 * stride_b1 + 0 * stride_b2)

            base_f = i0 * stride_f0 + i1 * stride_f1
            base_o = i0 * stride_o0 + i1 * stride_o1

            for blk in range(num_blocks):
                cols = blk * BLOCK_N + tl.arange(0, BLOCK_N)
                mask = cols < n_cols
                v = tl.load(
                    full_ptr + base_f + cols * stride_f2,
                    mask=mask,
                    other=0.0,
                )
                if FULL_IS_DIVIDEND:
                    out = v / brc
                else:
                    out = brc / v
                tl.store(
                    out_ptr + base_o + cols * stride_o2,
                    out,
                    mask=mask,
                )


def _pick_div_broadcast_2d(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[bool, torch.Tensor, torch.Tensor, tuple[int, int]] | None:
    if a.ndim != 2 or b.ndim != 2:
        return None
    if a.device != b.device or a.dtype != b.dtype:
        return None
    if a.shape[0] != b.shape[0]:
        return None
    if a.shape[1] > 1 and b.shape[1] == 1:
        return True, a, b, tuple(a.shape)
    if a.shape[1] == 1 and b.shape[1] > 1:
        return False, b, a, tuple(b.shape)
    return None


def _pick_div_broadcast_3d(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[bool, torch.Tensor, torch.Tensor, tuple[int, int, int]] | None:
    if a.ndim != 3 or b.ndim != 3:
        return None
    if a.device != b.device or a.dtype != b.dtype:
        return None
    if a.shape[0] != b.shape[0] or a.shape[1] != b.shape[1]:
        return None
    if a.shape[2] > 1 and b.shape[2] == 1:
        return True, a, b, tuple(a.shape)
    if a.shape[2] == 1 and b.shape[2] > 1:
        return False, b, a, tuple(b.shape)
    return None


def _div_fast_row_broadcast_2d(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor | None:
    picked = _pick_div_broadcast_2d(a, b)
    if picked is None:
        return None
    a_is_dividend, full, brc, out_shape = picked
    m, n = out_shape
    if out is not None:
        if tuple(out.shape) != out_shape:
            return None
    else:
        out_dtype = _true_div_out_dtype(a, b)
        out = torch.empty(out_shape, device=full.device, dtype=out_dtype)
    block_n = _BLOCK_N
    num_blocks = triton.cdiv(n, block_n)
    _div_row_broadcast_kernel_2d[(m,)](
        full,
        brc,
        out,
        full.stride(0),
        full.stride(1),
        brc.stride(0),
        brc.stride(1),
        out.stride(0),
        out.stride(1),
        n,
        num_blocks,
        FULL_IS_DIVIDEND=a_is_dividend,
        BLOCK_N=block_n,
    )
    return out


def _div_fast_row_broadcast_3d(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor | None:
    picked = _pick_div_broadcast_3d(a, b)
    if picked is None:
        return None
    a_is_dividend, full, brc, out_shape = picked
    d0, d1, n = out_shape
    m = d0 * d1
    if out is not None:
        if tuple(out.shape) != out_shape:
            return None
    else:
        out_dtype = _true_div_out_dtype(a, b)
        out = torch.empty(out_shape, device=full.device, dtype=out_dtype)
    block_n = min(2048, triton.next_power_of_2(n))
    num_blocks = triton.cdiv(n, block_n)
    iters = triton.cdiv(d0*d1, TOTAL_CORE_NUM)
    grid = (min(d0*d1, TOTAL_CORE_NUM),)
    _div_row_broadcast_kernel_3d[grid](
        full,
        brc,
        out,
        full.stride(0),
        full.stride(1),
        full.stride(2),
        brc.stride(0),
        brc.stride(1),
        brc.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        d0,
        d1,
        iters,
        n,
        num_blocks,
        FULL_IS_DIVIDEND=a_is_dividend,
        BLOCK_N=block_n,
    )
    return out


def _div_fast_row_broadcast(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if a.ndim == 3 and b.ndim == 3:
        return _div_fast_row_broadcast_3d(a, b, out=out)
    if a.ndim == 2 and b.ndim == 2:
        return _div_fast_row_broadcast_2d(a, b, out=out)
    return None


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
    logger.debug("GEMS_TSINGMICRO TRUE_DIVIDE")
    original_precision_priority = os.environ.get("PRECISION_PRIORITY", None)
    os.environ["PRECISION_PRIORITY"] = "0"  # force to high-perf mode
    try:
        if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
            fast = _div_fast_row_broadcast(A, B)
            if fast is not None:
                return fast
            return true_div_func(A, B)
        elif isinstance(A, torch.Tensor):
            return true_div_func_tensor_scalar(A, B)
        elif isinstance(B, torch.Tensor):
            return true_div_func_scalar_tensor(A, B)
        else:
            return torch.tensor(A / B)
    finally:
        if original_precision_priority is not None:
            os.environ["PRECISION_PRIORITY"] = original_precision_priority
        else:
            os.environ.pop("PRECISION_PRIORITY", None)


def true_divide_out(A, B, out):
    logger.debug("GEMS_TSINGMICRO TRUE_DIVIDE OUT")
    original_precision_priority = os.environ.get("PRECISION_PRIORITY", None)
    os.environ["PRECISION_PRIORITY"] = "0"  # force to high-perf mode
    try:
        if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
            if _div_fast_row_broadcast(A, B, out=out) is not None:
                return out
            return true_div_func(A, B, out0=out)
        elif isinstance(A, torch.Tensor):
            return true_div_func_tensor_scalar(A, B, out0=out)
        elif isinstance(B, torch.Tensor):
            return true_div_func_scalar_tensor(A, B, out0=out)
        else:
            return torch.tensor(A / B) if out is None else out.fill_(A / B)
    finally:
        if original_precision_priority is not None:
            os.environ["PRECISION_PRIORITY"] = original_precision_priority
        else:
            os.environ.pop("PRECISION_PRIORITY", None)


def true_divide_(A, B):
    logger.debug("GEMS_TSINGMICRO TRUE_DIVIDE_")
    original_precision_priority = os.environ.get("PRECISION_PRIORITY", None)
    os.environ["PRECISION_PRIORITY"] = "0"  # force to high-perf mode
    try:
        if isinstance(B, torch.Tensor):
            if _div_fast_row_broadcast(A, B, out=A) is not None:
                return A
            return true_div_func(A, B, out0=A)
        else:
            return true_div_func_tensor_scalar(A, B, out0=A)
    finally:
        if original_precision_priority is not None:
            os.environ["PRECISION_PRIORITY"] = original_precision_priority
        else:
            os.environ.pop("PRECISION_PRIORITY", None)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_func(x, y):
    return trunc(div_rz(x, y))


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_func_tensor_scalar(x, y):
    return trunc(div_rz(x, y))


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_func_scalar_tensor(x, y):
    return trunc(div_rz(x, y))


def trunc_divide(A, B):
    logger.debug("GEMS_TSINGMICRO TRUNC_DIVIDE")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return trunc_div_func(A, B)
    elif isinstance(A, torch.Tensor):
        return trunc_div_func_tensor_scalar(A, B)
    elif isinstance(B, torch.Tensor):
        return trunc_div_func_scalar_tensor(A, B)
    else:
        return torch.tensor(A / B)


def trunc_divide_(A, B):
    logger.debug("GEMS_TSINGMICRO TRUNC_DIVIDE_")
    if isinstance(B, torch.Tensor):
        return trunc_div_func(A, B, out0=A)
    else:
        return trunc_div_func_tensor_scalar(A, B, out0=A)


@triton.jit
def _int_floordiv(x, y):
    r = x % y
    c1 = r != 0
    c2 = (x < 0) ^ (y < 0)
    return tl.where(c1 & c2, x // y - 1, x // y)


@triton.jit
def _float_floordiv(x, y):
    remainder = fmod(x, y)
    imperfect = remainder != 0.0
    different_sign = (x < 0) ^ (y < 0)

    q = div_rn(x - remainder, y)
    q = tl.where(imperfect & different_sign, q - 1, q)

    floor_q = tl.math.floor(q)
    c = q - floor_q > 0.5
    floor_q = tl.where(c, floor_q + 1.0, floor_q)

    q_is_zeros = q == 0.0
    floor_q = tl.where(q_is_zeros, tl.where(different_sign, -0.0, 0.0), floor_q)

    is_div_by_zero = y == 0.0
    float_division = x / y
    out = tl.where(is_div_by_zero, float_division, floor_q)
    return out


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
    logger.debug("GEMS_TSINGMICRO FLOOR_DIVIDE")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return floor_div_func(A, B)
    elif isinstance(A, torch.Tensor):
        return floor_div_func_tensor_scalar(A, B)
    elif isinstance(B, torch.Tensor):
        return floor_div_func_scalar_tensor(A, B)
    else:
        return torch.tensor(A // B)


def floor_divide_(A, B):
    logger.debug("GEMS_TSINGMICRO FLOOR_DIVIDE_")
    if isinstance(B, torch.Tensor):
        return floor_div_func(A, B, out0=A)
    else:
        return floor_div_func_tensor_scalar(A, B, out0=A)


def div_mode(A, B, rounding_mode=None):
    if rounding_mode is None:
        return true_divide(A, B)
    elif rounding_mode == "trunc":
        return trunc_divide(A, B)
    elif rounding_mode == "floor":
        return floor_divide(A, B)
    else:
        msg = (
            f"div expected rounding_mode to be one of None, 'trunc', or 'floor' "
            f"but found {rounding_mode}."
        )
        raise ValueError(msg)


def div_mode_(A, B, rounding_mode=None):
    if rounding_mode is None:
        return true_divide_(A, B)
    elif rounding_mode == "trunc":
        return trunc_divide_(A, B)
    elif rounding_mode == "floor":
        return floor_divide_(A, B)
    else:
        msg = (
            f"div expected rounding_mode to be one of None, 'trunc', or 'floor' "
            f"but found {rounding_mode}."
        )
        raise ValueError(msg)


@triton.jit
def _remainder(x, y):
    r = x % y
    c1 = r != 0
    c2 = (x < 0) ^ (y < 0)
    return tl.where(c1 & c2, r + y, r)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def rem_tt(x, y):
    return _remainder(x, y)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def rem_ts(x, y):
    return _remainder(x, y)


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def rem_st(x, y):
    return _remainder(x, y)


def remainder(A, B):
    logger.debug("GEMS_TSINGMICRO REMAINDER")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return rem_tt(A, B)
    elif isinstance(A, torch.Tensor):
        return rem_ts(A, B)
    elif isinstance(B, torch.Tensor):
        return rem_st(A, B)
    else:
        return torch.tensor(A % B)


def remainder_(A, B):
    logger.debug("GEMS_TSINGMICRO REMAINDER_")
    if isinstance(B, torch.Tensor):
        return rem_tt(A, B, out0=A)
    else:
        return rem_ts(A, B, out0=A)
