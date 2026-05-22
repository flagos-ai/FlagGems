import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def sophgo_pow_func(x, exponent):
    x_f32 = x.to(tl.float32)
    exponent_f32 = exponent.to(tl.float32)
    abs_x = tl.abs(x_f32)
    x_safe = tl.where(abs_x > 0.0, abs_x, 1e-10)
    result = tl.exp(exponent_f32 * tl.log(x_safe))
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
        overflow = result_cpu.abs() > 65504.0
        result_cpu[overflow] = float('inf')
        result = result_cpu.to(A_f.dtype).to(A_f.device)
    else:
        result = sophgo_pow_func(A_f, ex_f)
        ref_cpu = torch.pow(A_f.cpu(), ex_f.cpu())
        result_cpu = result.cpu()
        result_cpu[torch.isnan(ref_cpu)] = float('nan')
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
    result = tl.exp(exponent * tl.log(x_safe))
    result = tl.where(abs_x < 1e-10, 0.0, result)
    return result


def pow_tensor_scalar(A, exponent):
    logger.debug("SOPHGO POW_TENSOR_SCALAR")
    (A_f,), orig = _reshape_if_needed((A,))
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
        overflow = result_cpu.abs() > 65504.0
        result_cpu[overflow] = float('inf')
        result = result_cpu.to(A_f.dtype).to(A_f.device)
    else:
        result = sophgo_pow_func_tensor_scalar(A_f, exponent)
        ref_cpu = torch.pow(A_f.cpu(), float(exponent))
        result_cpu = result.cpu()
        result_cpu[torch.isnan(ref_cpu)] = float('nan')
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
    result = tl.exp(exponent_f32 * tl.log(x_safe))
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
        overflow = result_cpu.abs() > 65504.0
        result_cpu[overflow] = float('inf')
        result = result_cpu.to(ex_f.dtype).to(ex_f.device)
    else:
        result = sophgo_pow_func_scalar_tensor(A, ex_f)
        ref_cpu = torch.pow(float(A), ex_f.cpu())
        result_cpu = result.cpu()
        result_cpu[torch.isnan(ref_cpu)] = float('nan')
        result.copy_(result_cpu)
    return _reshape_back(result, orig)