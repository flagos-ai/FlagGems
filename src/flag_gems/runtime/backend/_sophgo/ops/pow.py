import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic
from triton.language.extra.sophgo.libdevice import pow as _pow

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def sophgo_pow_func(x, exponent):
    x_f32 = x.to(tl.float32)
    exponent_f32 = exponent.to(tl.float32)
    abs_x = tl.abs(x_f32)
    x_safe = tl.where(abs_x > 0.0, abs_x, 1e-10)
    result = _pow(x_safe, exponent_f32)
    result = tl.where(abs_x < 1e-10, 0.0, result)
    return result


def _compute_masks_cpu(base, exponent):
    """Compute sign/NaN masks on CPU, handles both tensor and scalar base."""
    base_cpu = base if isinstance(base, float) else base.cpu()
    exp_cpu = exponent.cpu()
    is_negative = base_cpu < 0
    rounded = exp_cpu.float().round()
    is_integer = (exp_cpu.float() - rounded).abs() < 1e-6
    half = rounded * 0.5
    is_odd = (half - torch.floor(half)).abs() > 0.25
    return is_negative, is_integer, is_odd


def _apply_sign_and_nan(result, is_negative, is_integer, is_odd):
    """Apply sign correction and NaN injection from pre-computed CPU masks."""
    result_cpu = result.cpu()
    result_cpu[is_negative & is_integer & is_odd] = \
        -result_cpu[is_negative & is_integer & is_odd]
    result_cpu.masked_fill_(is_negative & ~is_integer, float('nan'))
    result.copy_(result_cpu)


def _reshape_if_needed(tensors, ndim_limit=4):
    """Flatten tensors to 1D if any has > ndim_limit dimensions."""
    orig_shapes = [t.shape for t in tensors]
    if any(t.ndim > ndim_limit for t in tensors):
        tensors = tuple(t.reshape(-1) for t in tensors)
    else:
        orig_shapes = None
    return tensors, orig_shapes


def _reshape_back(result, orig_shapes):
    if orig_shapes is not None:
        return result.reshape(orig_shapes[0])
    return result


def pow_tensor_tensor(A, exponent):
    logger.debug("SOPHGO POW_TENSOR_TENSOR")
    (A_f, ex_f), orig = _reshape_if_needed((A, exponent))
    if A_f.dtype in (torch.float16, torch.bfloat16):
        result_f32 = sophgo_pow_func(A_f.float(), ex_f.float())
        result_cpu = result_f32.cpu()
        neg, integer, odd = _compute_masks_cpu(A_f, ex_f)
        result_cpu[neg & integer & odd] = -result_cpu[neg & integer & odd]
        result_cpu[neg & ~integer] = float('nan')
        ref_cpu = torch.pow(A_f.cpu().float(), ex_f.cpu().float())
        result_cpu[torch.isinf(ref_cpu)] = float('inf')
        result = result_cpu.to(A_f.dtype).to(A_f.device)
    else:
        result = sophgo_pow_func(A_f, ex_f)
        ref_cpu = torch.pow(A_f.cpu(), ex_f.cpu())
        result_cpu = result.cpu()
        result_cpu[torch.isnan(ref_cpu)] = float('nan')
        result_cpu[torch.isinf(ref_cpu)] = float('inf')
        result.copy_(result_cpu)
    return _reshape_back(result, orig)


def pow_tensor_tensor_(A, exponent):
    logger.debug("SOPHGO POW_TENSOR_TENSOR_")
    result = pow_tensor_tensor(A, exponent)
    A.copy_(result)
    return A


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def sophgo_pow_func_tensor_scalar(x, exponent):
    x_f32 = x.to(tl.float32)
    abs_x = tl.abs(x_f32)
    x_safe = tl.where(abs_x > 0.0, abs_x, 1e-10)
    result = _pow(x_safe, exponent)
    result = tl.where(abs_x < 1e-10, 0.0, result)
    return result


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def sophgo_pow_square(x, exponent):
    x_f32 = x.to(tl.float32)
    return x_f32 * x_f32


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def sophgo_pow_cubic(x, exponent):
    x_f32 = x.to(tl.float32)
    return x_f32 * x_f32 * x_f32


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def sophgo_pow_sqrt(x, exponent):
    return tl.sqrt(x.to(tl.float32))


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def sophgo_pow_inv(x, exponent):
    return 1.0 / x.to(tl.float32)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def sophgo_pow_inv_square(x, exponent):
    inv = 1.0 / x.to(tl.float32)
    return inv * inv


PRECISE_EXP_THRESHOLD = 50


@pointwise_dynamic(is_tensor=[True, False, False], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def sophgo_pow_precise(x, int_exp, frac_exp):
    x_f32 = x.to(tl.float32)

    x2 = x_f32 * x_f32
    x4 = x2 * x2
    x8 = x4 * x4
    x16 = x8 * x8
    x32 = x16 * x16
    x64 = x32 * x32

    int_abs = -int_exp if int_exp < 0 else int_exp
    ipow = 1.0 + x_f32 * 0.0
    if int_abs & 1:  ipow = ipow * x_f32
    if int_abs & 2:  ipow = ipow * x2
    if int_abs & 4:  ipow = ipow * x4
    if int_abs & 8:  ipow = ipow * x8
    if int_abs & 16: ipow = ipow * x16
    if int_abs & 32: ipow = ipow * x32
    if int_abs & 64: ipow = ipow * x64

    if int_exp < 0:
        ipow = 1.0 / ipow

    fpart = 1.0 + x_f32 * 0.0
    if frac_exp != 0.0:
        fpart = _pow(x_f32, frac_exp)
    return ipow * fpart


def pow_tensor_scalar(A, exponent):
    logger.debug("SOPHGO POW_TENSOR_SCALAR")
    (A_f,), orig = _reshape_if_needed((A,))

    if exponent == 2.0:
        return _reshape_back(sophgo_pow_square(A_f, exponent), orig)
    if exponent == 3.0:
        return _reshape_back(sophgo_pow_cubic(A_f, exponent), orig)
    if exponent == 0.5:
        return _reshape_back(sophgo_pow_sqrt(A_f, exponent), orig)
    if exponent == -1.0:
        return _reshape_back(sophgo_pow_inv(A_f, exponent), orig)
    if exponent == -2.0:
        return _reshape_back(sophgo_pow_inv_square(A_f, exponent), orig)

    if abs(exponent) >= PRECISE_EXP_THRESHOLD:
        frac, integer = math.modf(exponent)
        int_exp = int(integer)
        result = sophgo_pow_precise(A_f, int_exp, frac)
        ref_cpu = torch.pow(A_f.cpu().float(), float(exponent))
        result_cpu = result.cpu()
        result_cpu[torch.isnan(ref_cpu)] = float('nan')
        result_cpu[torch.isinf(ref_cpu)] = float('inf')
        if A_f.dtype in (torch.float16, torch.bfloat16):
            result = result_cpu.to(A_f.dtype).to(A_f.device)
        else:
            result.copy_(result_cpu)
        return _reshape_back(result, orig)

    if A_f.dtype in (torch.float16, torch.bfloat16):
        result_f32 = sophgo_pow_func_tensor_scalar(A_f.float(), exponent)
        result_cpu = result_f32.cpu()
        A_cpu = A_f.cpu()
        exp_floor = round(exponent)
        is_integer = abs(exponent - exp_floor) < 1e-6
        if is_integer and exp_floor % 2 != 0:
            result_cpu[A_cpu < 0] = -result_cpu[A_cpu < 0]
        elif not is_integer:
            result_cpu[A_cpu < 0] = float('nan')
        ref_cpu = torch.pow(A_f.cpu().float(), float(exponent))
        result_cpu[torch.isinf(ref_cpu)] = float('inf')
        result = result_cpu.to(A_f.dtype).to(A_f.device)
    else:
        result = sophgo_pow_func_tensor_scalar(A_f, exponent)
        ref_cpu = torch.pow(A_f.cpu(), float(exponent))
        result_cpu = result.cpu()
        result_cpu[torch.isnan(ref_cpu)] = float('nan')
        result_cpu[torch.isinf(ref_cpu)] = float('inf')
        result.copy_(result_cpu)
    return _reshape_back(result, orig)


def pow_tensor_scalar_(A, exponent):
    logger.debug("SOPHGO POW_TENSOR_SCALAR_")
    result = pow_tensor_scalar(A, exponent)
    A.copy_(result)
    return A


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def sophgo_pow_func_scalar_tensor(x, exponent):
    x_f32 = x.to(tl.float32)
    exponent_f32 = exponent.to(tl.float32)
    abs_x = tl.abs(x_f32)
    x_safe = tl.where(abs_x > 0.0, abs_x, 1e-10)
    result = _pow(x_safe, exponent_f32)
    result = tl.where(abs_x < 1e-10, 0.0, result)
    return result


def pow_scalar(A, exponent):
    logger.debug("SOPHGO POW_SCALAR")
    (ex_f,), orig = _reshape_if_needed((exponent,))
    if ex_f.dtype in (torch.float16, torch.bfloat16):
        result_f32 = sophgo_pow_func_scalar_tensor(float(A), ex_f.float())
        result_cpu = result_f32.cpu()
        neg, integer, odd = _compute_masks_cpu(A, ex_f)
        result_cpu[neg & integer & odd] = -result_cpu[neg & integer & odd]
        result_cpu[neg & ~integer] = float('nan')
        ref_cpu = torch.pow(float(A), ex_f.cpu().float())
        result_cpu[torch.isinf(ref_cpu)] = float('inf')
        result = result_cpu.to(ex_f.dtype).to(ex_f.device)
    else:
        result = sophgo_pow_func_scalar_tensor(A, ex_f)
        ref_cpu = torch.pow(float(A), ex_f.cpu())
        result_cpu = result.cpu()
        result_cpu[torch.isnan(ref_cpu)] = float('nan')
        result_cpu[torch.isinf(ref_cpu)] = float('inf')
        result.copy_(result_cpu)
    return _reshape_back(result, orig)
