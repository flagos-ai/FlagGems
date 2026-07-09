"""
RMSNorm forward implementation aligned with TransformerEngine's rmsnorm_fwd.

TransformerEngine rmsnorm_fwd signature:
    rmsnorm_fwd(input, weight, eps, ln_out, quantizer, otype, sm_margin, zero_centered_gamma)

This implementation supports:
    - zero_centered_gamma: If true, applies (weight + 1) instead of weight
    - Pre-allocated output tensor (ln_out)
    - Output dtype conversion (otype)
    - Returns: (output, None, rsigma) where rsigma = 1/sqrt(mean(x^2) + eps)
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def rmsnorm_fwd_kernel(
    out_ptr,  # pointer to the output
    rsigma_ptr,  # pointer to rsigma (1/sqrt(mean(x^2) + eps))
    in_ptr,  # pointer to the input
    weight_ptr,  # pointer to the weights (gamma)
    y_stride_r,  # output row stride
    y_stride_c,  # output col stride
    x_stride_r,  # input row stride
    x_stride_c,  # input col stride
    N,  # number of columns (normalized dimension)
    eps,  # epsilon for numerical stability
    zero_centered_gamma: tl.constexpr,  # if True, use (weight + 1) instead of weight
    BLOCK_SIZE: tl.constexpr,
):
    """RMSNorm forward kernel for small N (fits in single block)."""
    # Determine compute dtype
    if tl.constexpr(in_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        in_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = in_ptr.dtype.element_ty

    pid = tl.program_id(0)
    out_ptr += pid * y_stride_r
    in_ptr += pid * x_stride_r

    # Load input row
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(in_ptr + cols * x_stride_c, mask=mask, other=0.0).to(cdtype)

    # Compute RMS: sqrt(mean(x^2) + eps)
    x_sq = x * x
    var = tl.sum(x_sq, axis=0) / N
    rsigma = 1.0 / tl.sqrt(var + eps)

    # Normalize
    x_norm = x * rsigma

    # Load and apply weight
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(cdtype)
    if zero_centered_gamma:
        w = w + 1.0

    # Output
    y = (x_norm * w).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + cols * y_stride_c, y, mask=mask)

    # Store rsigma
    tl.store(rsigma_ptr + pid, rsigma)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("rms_norm_loop"),
    key=["N"],
)
@triton.jit(do_not_specialize=["eps"])
def rmsnorm_fwd_loop_kernel(
    out_ptr,  # pointer to the output
    rsigma_ptr,  # pointer to rsigma
    in_ptr,  # pointer to the input
    weight_ptr,  # pointer to the weights
    N,  # number of columns
    eps,  # epsilon
    zero_centered_gamma: tl.constexpr,
    TILE_N: tl.constexpr,
):
    """RMSNorm forward kernel for large N (requires loop)."""
    if tl.constexpr(in_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        in_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = in_ptr.dtype.element_ty

    pid = tl.program_id(0)
    in_ptr += pid * N
    out_ptr += pid * N

    # First pass: compute sum of squares
    sum_sq = tl.zeros([TILE_N], dtype=cdtype)
    for off in range(0, N, TILE_N):
        cols = off + tl.arange(0, TILE_N)
        mask = cols < N
        x = tl.load(in_ptr + cols, mask=mask, other=0.0).to(cdtype)
        sum_sq += x * x

    # Compute rsigma
    var = tl.sum(sum_sq, axis=0) / N
    rsigma = 1.0 / tl.sqrt(var + eps)
    tl.store(rsigma_ptr + pid, rsigma)

    # Second pass: normalize and apply weight
    for off in range(0, N, TILE_N):
        cols = off + tl.arange(0, TILE_N)
        mask = cols < N
        x = tl.load(in_ptr + cols, mask=mask, other=0.0).to(cdtype)
        w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(cdtype)

        if zero_centered_gamma:
            w = w + 1.0

        x_norm = x * rsigma
        y = (x_norm * w).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + cols, y, mask=mask)


def rmsnorm_fwd(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    ln_out: torch.Tensor = None,
    quantizer=None,
    otype: torch.dtype = None,
    sm_margin: int = 0,
    zero_centered_gamma: bool = False,
):
    """
    RMSNorm forward pass aligned with TransformerEngine's rmsnorm_fwd.

    Args:
        input: Input tensor of shape (..., N)
        weight: Weight tensor (gamma) of shape (N,)
        eps: Epsilon for numerical stability
        ln_out: Pre-allocated output tensor (optional)
        quantizer: FP8 quantizer (currently not supported, placeholder for API compatibility)
        otype: Output dtype (optional, defaults to input dtype)
        sm_margin: SM margin for kernel execution (currently not used, placeholder for API compatibility)
        zero_centered_gamma: If True, applies (weight + 1) instead of weight

    Returns:
        Tuple of (output, None, rsigma) where:
            - output: Normalized output tensor
            - None: Placeholder for compatibility with TransformerEngine
            - rsigma: Reciprocal of RMS, shape (M,) where M = product of batch dims

    Note:
        - quantizer parameter is currently not supported (FP8 quantization not implemented)
        - sm_margin parameter is currently ignored (Triton handles SM scheduling)
    """
    # Validate unsupported parameters
    if quantizer is not None:
        logger.warning("quantizer parameter is not yet supported, ignoring")

    if sm_margin != 0:
        logger.debug(f"sm_margin={sm_margin} is ignored in Triton implementation")
    # Handle input shape
    original_shape = input.shape
    N = original_shape[-1]

    # Flatten batch dimensions
    input_2d = input.view(-1, N)
    M = input_2d.shape[0]

    # Ensure contiguous
    input_2d = input_2d.contiguous()
    weight = weight.contiguous()

    # Determine output dtype
    if otype is None:
        otype = input.dtype

    # Allocate output tensor
    if ln_out is None:
        out = torch.empty(M, N, dtype=otype, device=input.device)
    else:
        # Use pre-allocated output
        out = ln_out.view(-1, N)
        assert out.shape == (
            M,
            N,
        ), f"ln_out shape mismatch: expected {(M, N)}, got {out.shape}"

    # Allocate rsigma tensor
    rsigma = torch.empty(M, dtype=torch.float32, device=input.device)

    # Get strides
    x_stride_r = input_2d.stride(0)
    x_stride_c = input_2d.stride(1)
    y_stride_r = out.stride(0)
    y_stride_c = out.stride(1)

    # Choose kernel based on N size
    MAX_FUSED_SIZE = 65536 // input.element_size()

    with torch_device_fn.device(input.device):
        if N <= MAX_FUSED_SIZE:
            # Use single-block kernel
            BLOCK_SIZE = triton.next_power_of_2(N)
            rmsnorm_fwd_kernel[(M,)](
                out,
                rsigma,
                input_2d,
                weight,
                y_stride_r,
                y_stride_c,
                x_stride_r,
                x_stride_c,
                N,
                eps,
                zero_centered_gamma,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        else:
            # Use loop kernel for large N
            rmsnorm_fwd_loop_kernel[(M,)](
                out,
                rsigma,
                input_2d,
                weight,
                N,
                eps,
                zero_centered_gamma,
            )

    # Restore original shape for output
    out = out.view(*original_shape[:-1], N)
    if ln_out is not None:
        out = ln_out  # Return the same tensor object

    return out, None, rsigma
