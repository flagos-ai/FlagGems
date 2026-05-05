import logging
from enum import IntEnum

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import libentry

device = device.name
logger = logging.getLogger(__name__)


class Reduction(IntEnum):
    NONE = 0
    MEAN = 1
    SUM = 2


@libentry()
@triton.jit
def _smooth_l1_none_kernel(
    inp,
    target,
    out,
    n_elements,
    beta: tl.constexpr,
    BETA_ZERO: tl.constexpr,
    BETA_ONE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(inp + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(target + offsets, mask=mask, other=0.0).to(tl.float32)
    diff = x - y
    abs_diff = tl.abs(diff)
    if BETA_ZERO:
        loss = abs_diff
    elif BETA_ONE:
        loss = tl.where(abs_diff < 1.0, 0.5 * diff * diff, abs_diff - 0.5)
    else:
        loss = tl.where(
            abs_diff < beta,
            0.5 * diff * diff / beta,
            abs_diff - 0.5 * beta,
        )
    tl.store(out + offsets, loss, mask=mask)


@libentry()
@triton.jit
def _smooth_l1_small_reduce_kernel(
    inp,
    target,
    out,
    n_elements,
    beta: tl.constexpr,
    REDUCTION: tl.constexpr,
    BETA_ZERO: tl.constexpr,
    BETA_ONE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(inp + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(target + offsets, mask=mask, other=0.0).to(tl.float32)
    diff = x - y
    abs_diff = tl.abs(diff)
    if BETA_ZERO:
        loss = abs_diff
    elif BETA_ONE:
        loss = tl.where(abs_diff < 1.0, 0.5 * diff * diff, abs_diff - 0.5)
    else:
        loss = tl.where(
            abs_diff < beta,
            0.5 * diff * diff / beta,
            abs_diff - 0.5 * beta,
        )
    loss = tl.where(mask, loss, 0.0)
    total = tl.sum(loss, axis=0)
    if REDUCTION == 1:
        total = total / n_elements
    tl.store(out, total)


@libentry()
@triton.jit
def _smooth_l1_partial_sum_kernel(
    inp,
    target,
    mid,
    n_elements,
    beta: tl.constexpr,
    BETA_ZERO: tl.constexpr,
    BETA_ONE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(inp + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(target + offsets, mask=mask, other=0.0).to(tl.float32)
    diff = x - y
    abs_diff = tl.abs(diff)
    if BETA_ZERO:
        loss = abs_diff
    elif BETA_ONE:
        loss = tl.where(abs_diff < 1.0, 0.5 * diff * diff, abs_diff - 0.5)
    else:
        loss = tl.where(
            abs_diff < beta,
            0.5 * diff * diff / beta,
            abs_diff - 0.5 * beta,
        )
    loss = tl.where(mask, loss, 0.0)
    tl.store(mid + pid, tl.sum(loss, axis=0))


@libentry()
@triton.jit
def _smooth_l1_final_reduce_kernel(
    mid,
    out,
    mid_size,
    n_elements,
    REDUCTION: tl.constexpr,
    BLOCK_MID: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_MID)
    mask = offsets < mid_size
    vals = tl.load(mid + offsets, mask=mask, other=0.0).to(tl.float32)
    total = tl.sum(vals, axis=0)
    if REDUCTION == 1:
        total = total / n_elements
    tl.store(out, total)


@libentry()
@triton.jit
def _smooth_l1_backward_kernel(
    grad_output,
    inp,
    target,
    grad_input,
    n_elements,
    reduction_elements,
    beta: tl.constexpr,
    REDUCTION: tl.constexpr,
    BETA_ONE: tl.constexpr,
    GRAD_SCALAR: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(inp + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(target + offsets, mask=mask, other=0.0).to(tl.float32)
    diff = x - y

    if beta == 0.0:
        grad = tl.where(
            diff == 0.0,
            float("nan"),
            tl.where(diff > 0.0, 1.0, -1.0),
        )
    elif BETA_ONE:
        grad = tl.minimum(tl.maximum(diff, -1.0), 1.0)
    else:
        grad = tl.where(
            diff < -beta,
            -1.0,
            tl.where(diff > beta, 1.0, diff / beta),
        )

    if GRAD_SCALAR:
        grad_out = tl.load(grad_output).to(tl.float32)
    else:
        grad_ptrs = grad_output + offsets
        grad_out = tl.load(grad_ptrs, mask=mask, other=0.0).to(tl.float32)
    if REDUCTION == 1:
        grad_out = grad_out * (1.0 / reduction_elements)
    tl.store(grad_input + offsets, grad * grad_out, mask=mask)


@libentry()
@triton.jit
def _smooth_l1_backward_scalar_kernel(
    grad_output,
    inp,
    target,
    grad_input,
    n_elements,
    grad_scale,
    beta: tl.constexpr,
    BETA_ZERO: tl.constexpr,
    BETA_ONE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(inp + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(target + offsets, mask=mask, other=0.0).to(tl.float32)
    diff = x - y

    if BETA_ZERO:
        grad = tl.where(
            diff == 0.0,
            float("nan"),
            tl.where(diff > 0.0, 1.0, -1.0),
        )
    elif BETA_ONE:
        grad = tl.minimum(tl.maximum(diff, -1.0), 1.0)
    else:
        grad = tl.minimum(tl.maximum(diff / beta, -1.0), 1.0)

    grad_out = tl.load(grad_output).to(tl.float32) * grad_scale
    tl.store(grad_input + offsets, grad * grad_out, mask=mask)


@libentry()
@triton.jit
def _smooth_l1_backward_scalar_beta_one_kernel(
    grad_output,
    inp,
    target,
    grad_input,
    n_elements,
    grad_scale,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(inp + offsets, mask=mask, other=0.0)
    y = tl.load(target + offsets, mask=mask, other=0.0)
    grad = tl.minimum(tl.maximum(x - y, -1.0), 1.0)
    grad_out = tl.load(grad_output) * grad_scale
    tl.store(grad_input + offsets, grad * grad_out, mask=mask)


@libentry()
@triton.jit
def _smooth_l1_backward_large_scalar_mean_beta_one_kernel(
    grad_output,
    inp,
    target,
    grad_input,
    n_elements,
    reduction_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(inp + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(target + offsets, mask=mask, other=0.0).to(tl.float32)
    grad = tl.minimum(tl.maximum(x - y, -1.0), 1.0)
    grad_out = tl.load(grad_output).to(tl.float32) * (1.0 / reduction_elements)
    tl.store(grad_input + offsets, grad * grad_out, mask=mask)


def _normalize_reduction(reduction):
    if isinstance(reduction, str):
        reduction = reduction.lower()
        if reduction == "none":
            return Reduction.NONE
        if reduction == "mean":
            return Reduction.MEAN
        if reduction == "sum":
            return Reduction.SUM
    if isinstance(reduction, int) and reduction in (0, 1, 2):
        return Reduction(reduction)
    raise ValueError("reduction must be one of 'none', 'mean', or 'sum'")


def _check_beta(beta):
    beta = float(beta)
    if beta < 0:
        raise RuntimeError(  # noqa: E501
            "smooth_l1_loss does not support negative values for beta."
        )
    return beta


def _check_devices(input, target):
    if input.device.type != device or target.device.type != device:
        raise AssertionError(  # noqa: E501
            "smooth_l1_loss: input and target must be device tensors."
        )
    if input.device != target.device:
        raise AssertionError(
            "smooth_l1_loss: input and target must be on the same device."
        )


def _check_float_inputs(input, target):
    if not input.is_floating_point() or not target.is_floating_point():
        raise NotImplementedError(
            "smooth_l1_loss is implemented for floating tensors only"
        )


def _compute_dtype(input, target):
    dtype = torch.result_type(input, target)
    if not dtype.is_floating_point:
        raise NotImplementedError(
            "smooth_l1_loss is implemented for floating tensors only"
        )
    return dtype


def _broadcast_forward(input, target):
    _check_devices(input, target)
    _check_float_inputs(input, target)
    dtype = _compute_dtype(input, target)
    inp, tgt = torch.broadcast_tensors(input, target)
    if inp.dtype != dtype:
        inp = inp.to(dtype)
    if tgt.dtype != dtype:
        tgt = tgt.to(dtype)
    return inp.contiguous(), tgt.contiguous(), dtype, inp.shape


def _broadcast_backward(input, target):
    _check_devices(input, target)
    _check_float_inputs(input, target)
    dtype = _compute_dtype(input, target)
    input, target = torch.broadcast_tensors(input, target)
    if input.dtype != dtype:
        input = input.to(dtype)
    if target.dtype != dtype:
        target = target.to(dtype)
    return input.contiguous(), target.contiguous(), dtype


def _check_grad_output(grad_output, input):
    if grad_output.device.type != device:
        raise AssertionError(
            "smooth_l1_loss_backward: grad_output must be a device tensor."
        )
    if grad_output.device != input.device:
        raise AssertionError(
            "smooth_l1_loss_backward: grad_output must be on the same device."
        )


def _empty_forward(shape, dtype, device, reduction):
    if reduction == Reduction.NONE:
        return torch.empty(shape, dtype=dtype, device=device)
    if reduction == Reduction.MEAN:
        return torch.full((), float("nan"), dtype=dtype, device=device)
    return torch.zeros((), dtype=dtype, device=device)


def _copy_or_resize(out, result):
    if out.shape != result.shape:
        if out.numel() != 0:
            out.resize_(0)
        out.resize_(result.shape)
    out.copy_(result)
    return out


def _launch_none(inp, target, dtype, beta, out=None):
    n_elements = inp.numel()
    if out is None:
        out = torch.empty(inp.shape, dtype=dtype, device=inp.device)
        out_contiguous = out
    else:
        if out.device != inp.device:
            raise AssertionError(  # noqa: E501
                "smooth_l1_loss.out: out must be on the same device."
            )
        if out.shape != inp.shape:
            if out.numel() != 0:
                out.resize_(0)
            out.resize_(inp.shape)
        out_contiguous = out if out.is_contiguous() else torch.empty_like(out)

    if n_elements > 0:
        block_size = 1024
        grid = (triton.cdiv(n_elements, block_size),)
        with torch_device_fn.device(inp.device):
            _smooth_l1_none_kernel[grid](
                inp,
                target,
                out_contiguous,
                n_elements,
                beta,
                BETA_ZERO=(beta == 0.0),
                BETA_ONE=(beta == 1.0),
                BLOCK_SIZE=block_size,
                num_warps=4,
            )
    if out_contiguous is not out:
        out.copy_(out_contiguous)
    return out


def _launch_reduce(inp, target, dtype, beta, reduction, out=None):
    n_elements = inp.numel()
    if n_elements == 0:
        result = _empty_forward(inp.shape, dtype, inp.device, reduction)
        return result if out is None else _copy_or_resize(out, result)

    result = (  # noqa: E501
        torch.empty((), dtype=dtype, device=inp.device) if out is None else out
    )
    if result.device != inp.device:
        raise AssertionError(  # noqa: E501
            "smooth_l1_loss.out: out must be on the same device."
        )
    if result.dim() != 0:
        if result.numel() != 0:
            result.resize_(0)
        result.resize_(())

    if n_elements <= 8192:
        block_size = triton.next_power_of_2(n_elements)
        with torch_device_fn.device(inp.device):
            _smooth_l1_small_reduce_kernel[(1,)](
                inp,
                target,
                result,
                n_elements,
                beta,
                REDUCTION=int(reduction),
                BETA_ZERO=(beta == 0.0),
                BETA_ONE=(beta == 1.0),
                BLOCK_SIZE=block_size,
                num_warps=4,
            )
        return result

    if n_elements >= 64 * 1024 * 1024:
        block_size = 8192
    else:
        block_size = 2048 if n_elements >= 4 * 1024 * 1024 else 1024
    mid_size = triton.cdiv(n_elements, block_size)
    mid = torch.empty((mid_size,), dtype=torch.float32, device=inp.device)
    with torch_device_fn.device(inp.device):
        _smooth_l1_partial_sum_kernel[(mid_size,)](
            inp,
            target,
            mid,
            n_elements,
            beta,
            BETA_ZERO=(beta == 0.0),
            BETA_ONE=False,
            BLOCK_SIZE=block_size,
            num_warps=8 if block_size >= 2048 else 4,
        )
        if mid_size <= 65536:
            block_mid = triton.next_power_of_2(mid_size)
            _smooth_l1_final_reduce_kernel[(1,)](
                mid,
                result,
                mid_size,
                n_elements,
                REDUCTION=int(reduction),
                BLOCK_MID=block_mid,
                num_warps=8 if block_mid >= 2048 else 4,
            )
        else:
            total = mid.sum()
            if reduction == Reduction.MEAN:
                total = total / n_elements
            result.copy_(total.to(result.dtype))
    return result


def smooth_l1_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    reduction=Reduction.MEAN.value,
    beta: float = 1.0,
) -> torch.Tensor:
    logger.debug("GEMS SMOOTH_L1_LOSS")
    reduction = _normalize_reduction(reduction)
    beta = _check_beta(beta)
    inp, tgt, dtype, _ = _broadcast_forward(input, target)
    if reduction == Reduction.NONE:
        return _launch_none(inp, tgt, dtype, beta)
    return _launch_reduce(inp, tgt, dtype, beta, reduction)


def smooth_l1_loss_out(
    input: torch.Tensor,
    target: torch.Tensor,
    reduction=Reduction.MEAN.value,
    beta: float = 1.0,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    logger.debug("GEMS SMOOTH_L1_LOSS OUT")
    reduction = _normalize_reduction(reduction)
    beta = _check_beta(beta)
    inp, tgt, dtype, _ = _broadcast_forward(input, target)
    if reduction == Reduction.NONE:
        if out.dtype == dtype:
            return _launch_none(inp, tgt, dtype, beta, out=out)
        result = _launch_none(inp, tgt, dtype, beta)
        return _copy_or_resize(out, result)
    return _launch_reduce(inp, tgt, out.dtype, beta, reduction, out=out)


def smooth_l1_loss_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    target: torch.Tensor,
    reduction,
    beta: float,
) -> torch.Tensor:
    logger.debug("GEMS SMOOTH_L1_LOSS BACKWARD")
    reduction = _normalize_reduction(reduction)
    beta = _check_beta(beta)
    reduction_elements = input.numel()
    inp, tgt, _ = _broadcast_backward(input, target)
    result = torch.empty(inp.shape, dtype=input.dtype, device=input.device)
    return _smooth_l1_loss_backward_prepared(
        grad_output,
        inp,
        tgt,
        reduction,
        beta,
        reduction_elements,
        grad_input=result,
    )


def smooth_l1_loss_backward_out(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    target: torch.Tensor,
    reduction,
    beta: float,
    *,
    grad_input: torch.Tensor,
) -> torch.Tensor:
    logger.debug("GEMS SMOOTH_L1_LOSS BACKWARD OUT")
    reduction = _normalize_reduction(reduction)
    beta = _check_beta(beta)
    reduction_elements = input.numel()
    inp, tgt, _ = _broadcast_backward(input, target)
    return _smooth_l1_loss_backward_prepared(
        grad_output,
        inp,
        tgt,
        reduction,
        beta,
        reduction_elements,
        grad_input=grad_input,
    )


def _smooth_l1_loss_backward_prepared(
    grad_output,
    inp,
    tgt,
    reduction,
    beta,
    reduction_elements,
    *,
    grad_input,
):
    _check_grad_output(grad_output, inp)
    n_elements = inp.numel()
    if grad_input.device != inp.device:
        raise AssertionError(
            "smooth_l1_loss_backward: grad_input must be on the same device."
        )
    if grad_input.shape != inp.shape:
        if grad_input.numel() != 0:
            grad_input.resize_(0)
        grad_input.resize_(inp.shape)
    if grad_input.is_contiguous():
        out = grad_input
    else:
        out = torch.empty_like(grad_input)

    if n_elements == 0:
        out.zero_()
        if out is not grad_input:
            grad_input.copy_(out)
        return grad_input

    grad_scalar = grad_output.numel() == 1
    if grad_scalar:
        grad = grad_output.contiguous()
    else:
        grad = grad_output.expand(inp.shape).contiguous()
    small_scalar = grad_scalar and n_elements < 4 * 1024 * 1024
    if small_scalar:
        grad_scale = 1.0
        if reduction == Reduction.MEAN:
            grad_scale = 1.0 / reduction_elements
        if inp.dtype == torch.float16:
            block_size = 1024
            scalar_warps = 8
        elif inp.dtype == torch.float32:
            block_size = 2048
            scalar_warps = 4
        else:
            block_size = 2048
            scalar_warps = 8
        grid = (triton.cdiv(n_elements, block_size),)
        with torch_device_fn.device(inp.device):
            if beta == 1.0 and inp.dtype in (torch.float16, torch.bfloat16):
                _smooth_l1_backward_scalar_beta_one_kernel[grid](
                    grad,
                    inp,
                    tgt,
                    out,
                    n_elements,
                    grad_scale,
                    BLOCK_SIZE=block_size,
                    num_warps=scalar_warps,
                )
            else:
                _smooth_l1_backward_scalar_kernel[grid](
                    grad,
                    inp,
                    tgt,
                    out,
                    n_elements,
                    grad_scale,
                    beta,
                    BETA_ZERO=(beta == 0.0),
                    BETA_ONE=(beta == 1.0),
                    BLOCK_SIZE=block_size,
                    num_warps=scalar_warps,
                )
        if out is not grad_input:
            grad_input.copy_(out)
        return grad_input

    large_scalar_mean_beta_one = (
        grad_scalar
        and reduction == Reduction.MEAN
        and beta == 1.0
        and n_elements >= 4 * 1024 * 1024
    )
    if large_scalar_mean_beta_one:
        if inp.dtype == torch.float32:
            block_size = 512
            scalar_warps = 4
        else:
            block_size = 2048
            scalar_warps = 8
        grid = (triton.cdiv(n_elements, block_size),)
        with torch_device_fn.device(inp.device):
            _smooth_l1_backward_large_scalar_mean_beta_one_kernel[grid](
                grad,
                inp,
                tgt,
                out,
                n_elements,
                reduction_elements,
                BLOCK_SIZE=block_size,
                num_warps=scalar_warps,
            )
        if out is not grad_input:
            grad_input.copy_(out)
        return grad_input

    large_fp32 = inp.dtype == torch.float32 and n_elements >= 4 * 1024 * 1024
    large_half = (
        inp.dtype in (torch.float16, torch.bfloat16)
        and n_elements >= 4 * 1024 * 1024
    )
    if large_fp32:
        block_size = 512
    elif large_half:
        block_size = 2048
    else:
        block_size = 1024
    grid = (triton.cdiv(n_elements, block_size),)
    with torch_device_fn.device(inp.device):
        _smooth_l1_backward_kernel[grid](
            grad,
            inp,
            tgt,
            out,
            n_elements,
            reduction_elements,
            beta,
            REDUCTION=int(reduction),
            BETA_ONE=False,
            GRAD_SCALAR=grad_scalar,
            BLOCK_SIZE=block_size,
            num_warps=8 if block_size >= 2048 else 4,
        )
    if out is not grad_input:
        grad_input.copy_(out)
    return grad_input
