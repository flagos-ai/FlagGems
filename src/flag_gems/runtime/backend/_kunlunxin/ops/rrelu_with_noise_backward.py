import logging

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.runtime import torch_device_fn

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=2048,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


# ---------------------------------------------------------------------------
# Generic fallback paths (non-contiguous inputs, mixed dtypes, exotic shapes)
# ---------------------------------------------------------------------------
@pointwise_dynamic(
    is_tensor=[True, True, True, False, False, False],
    promotion_methods=[(0, 1, 2, "DEFAULT")],
    config=config_,
)
@triton.jit
def rrelu_with_noise_backward_func(grad_output, input, noise, lower, upper, training):
    grad = grad_output.to(tl.float32)
    if training:
        result = grad * noise.to(tl.float32)
    else:
        slope = (lower + upper) * 0.5
        result = grad * tl.where(input.to(tl.float32) > 0, 1.0, slope)
    return result.to(grad_output.dtype)


def _generic_fallback(grad_output, input, noise, lower, upper, training):
    return rrelu_with_noise_backward_func(
        grad_output, input, noise, lower, upper, training
    )


# Pointwise-codegen 2-tensor variant: default codegen path (same as the
# shared `mul` override), which measured faster than the custom 12-CTA
# chunk-loop config on this backend for mid/large shapes.
@pointwise_dynamic(
    is_tensor=[True, True],
    promotion_methods=[(0, 1, "DEFAULT")],
)
@triton.jit
def _pd_train_native(grad_output, noise):
    return grad_output * noise


# Same kernel under the legacy custom codegen config (kunlunAutoGrid 1-CTA
# for tiny tensors): baseline-beating only for bf16 below 16K elems, where
# the default 12-CTA grid and the flat kernel are both slower (launch floor).
@pointwise_dynamic(
    is_tensor=[True, True],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_,
)
@triton.jit
def _pd_train_native_custom(grad_output, noise):
    return grad_output * noise


# ---------------------------------------------------------------------------
# Fast paths: training mode only needs (grad_output, noise); eval mode only
# needs (grad_output, input).  Unused tensors are not loaded at all, the
# flat 1D access is fully contiguous (stride-1 block DMA) and the boundary
# mask is a compile-time constexpr so fully-covered tiles run unmasked.
# ---------------------------------------------------------------------------
@triton.jit
def _rrelu_train_kernel(
    grad_ptr,
    noise_ptr,
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if NEED_MASK:
        mask = offsets < n_elements
        grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0)
        noise = tl.load(noise_ptr + offsets, mask=mask, other=0.0)
        tl.store(out_ptr + offsets, grad * noise, mask=mask)
    else:
        grad = tl.load(grad_ptr + offsets)
        noise = tl.load(noise_ptr + offsets)
        tl.store(out_ptr + offsets, grad * noise)


@triton.jit
def _rrelu_eval_kernel(
    grad_ptr,
    input_ptr,
    out_ptr,
    n_elements,
    slope,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if NEED_MASK:
        mask = offsets < n_elements
        grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        result = grad * tl.where(x > 0, 1.0, slope)
        tl.store(out_ptr + offsets, result.to(grad_ptr.dtype.element_ty), mask=mask)
    else:
        grad = tl.load(grad_ptr + offsets).to(tl.float32)
        x = tl.load(input_ptr + offsets).to(tl.float32)
        result = grad * tl.where(x > 0, 1.0, slope)
        tl.store(out_ptr + offsets, result.to(grad_ptr.dtype.element_ty))


def _pick_block(n_elements: int):
    # Powers-of-two shapes in the benchmark/tests all divide 16384, so the
    # fast paths run unmasked; 16384 elems/CTA balances DMA burst size against
    # CTA count.  Tiny tensors take a single CTA of the exact size.
    if n_elements < 16384:
        return max(1, triton.next_power_of_2(n_elements))
    return 16384


def _launch_train(grad_output: torch.Tensor, noise: torch.Tensor, out: torch.Tensor):
    n = out.numel()
    if n == 0:
        return out
    block = _pick_block(n)
    grid = (triton.cdiv(n, block),)
    num_warps = 16 if block >= 16384 else (8 if block >= 2048 else 4)
    with torch_device_fn.device(grad_output.device):
        _rrelu_train_kernel[grid](
            grad_output,
            noise,
            out,
            n,
            BLOCK=block,
            NEED_MASK=(n % block != 0),
            num_warps=num_warps,
        )
    return out


def _launch_eval(grad_output: torch.Tensor, input: torch.Tensor, out: torch.Tensor, lower, upper):
    n = out.numel()
    if n == 0:
        return out
    block = _pick_block(n)
    grid = (triton.cdiv(n, block),)
    num_warps = 16 if block >= 16384 else (8 if block >= 2048 else 4)
    slope = (lower + upper) * 0.5
    with torch_device_fn.device(grad_output.device):
        _rrelu_eval_kernel[grid](
            grad_output,
            input,
            out,
            n,
            slope,
            BLOCK=block,
            NEED_MASK=(n % block != 0),
            num_warps=num_warps,
        )
    return out


def rrelu_with_noise_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    noise: torch.Tensor,
    lower: float,
    upper: float,
    training: bool,
    self_is_result: bool = False,
):
    logger.debug("GEMS_KUNLUNXIN RRELU_WITH_NOISE_BACKWARD")
    if training:
        if (
            grad_output.is_contiguous()
            and noise.is_contiguous()
            and grad_output.dtype == noise.dtype
        ):
            # Per-dtype / per-size dispatch (measured 2026-08-16, XPU):
            #  - bf16 : legacy 1-CTA codegen below 16K elems is faster than
            #           both the flat kernel and the default 12-CTA codegen
            #           (launch floor); default codegen from 16K up
            #  - fp16/fp32: flat kernel at <= 65536 elems (launch floor
            #           where the 12-CTA codegen grid loses), default
            #           pointwise codegen above it
            n = grad_output.numel()
            if grad_output.dtype == torch.bfloat16:
                if n < 16384:
                    return _pd_train_native_custom(grad_output, noise)
                return _pd_train_native(grad_output, noise)
            if n <= 65536:
                return _launch_train(
                    grad_output, noise, torch.empty_like(grad_output)
                )
            return _pd_train_native(grad_output, noise)
        return _generic_fallback(grad_output, input, noise, lower, upper, training)
    if (
        grad_output.is_contiguous()
        and input.is_contiguous()
        and grad_output.dtype == input.dtype
    ):
        return _launch_eval(
            grad_output, input, torch.empty_like(grad_output), lower, upper
        )
    return _generic_fallback(grad_output, input, noise, lower, upper, training)