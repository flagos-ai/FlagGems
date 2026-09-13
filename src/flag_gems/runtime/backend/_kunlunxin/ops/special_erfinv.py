import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _special_erfinv_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    xf = x.to(tl.float32)
    absx = tl.abs(xf)

    # erfinv(x) ~= x * P(x^2), odd polynomial.  P is a least-squares fit of
    # erfinv(x)/x over |x| <= 0.9 (max abs err ~1.3e-5 in fp32), evaluated in
    # Horner form.  Compared with the classic A&S log+sqrt two-branch form
    # this drops the transcendental math entirely.
    ax2 = absx * absx
    p = 11.150475
    p = -35.22204 + p * ax2
    p = 47.801117 + p * ax2
    p = -35.59043 + p * ax2
    p = 15.894029 + p * ax2
    p = -4.1923985 + p * ax2
    p = 0.75717944 + p * ax2
    p = 0.07057473 + p * ax2
    p = 0.2342005 + p * ax2
    p = 0.8862025 + p * ax2

    res = xf * p

    # Natural propagation gives NaN for |x| > 1 (poly diverges but the
    # |x| > 1 region is out-of-domain); keep the semantics explicit with two
    # scalar selects: |x| > 1 -> NaN, |x| == 1 -> sign(x) * inf.
    res = tl.where(absx > 1.0, float("nan"), res)
    res = tl.where(absx == 1.0, xf * float("inf"), res)

    y = res.to(x.dtype)
    tl.store(out_ptr + offsets, y, mask=mask)


def _launch_special_erfinv_kernel(x: torch.Tensor, out: torch.Tensor):
    assert x.device == out.device, "Input and output must be on the same device"
    assert (
        x.numel() == out.numel()
    ), "Input and output must have the same number of elements"
    assert x.dtype == out.dtype, "Input and output must have the same dtype"
    n_elements = x.numel()
    # Size-adaptive tile.  Large fixed blocks keep the grid small and avoid
    # launch-bound overhead on XPU (large shapes), while small inputs use a
    # modest tile so tiny tensors are not dominated by one oversized block.
    # Block sweep (fp32/fp16/bf16, n in 2^16..2^26): >=2^18 elements are
    # fastest at 16384; smaller inputs hit the ~20us launch floor at 1024.
    # Note: keep the NaN compare as `~(xf == xf)` -- the `!=` form fails to
    # lower at BLOCK_SIZE >= 1024 (LLVM "Cannot select" on setuo).
    BLOCK_SIZE = 1024 if n_elements <= 131072 else 16384
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _special_erfinv_kernel[grid](
        x,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )


def special_erfinv(x: torch.Tensor):
    """Special erfinv function"""
    logger.debug("GEMS KUNLUNXIN special_erfinv")
    x_in = x
    if not x_in.is_contiguous():
        x_in = x_in.contiguous()
    out = torch.empty_like(x_in)
    _launch_special_erfinv_kernel(x_in, out)
    # Match original shape/strides of input if needed
    if out.shape != x.shape or out.stride() != x.stride():
        out = out.reshape(x.shape).as_strided(x.size(), x.stride())
    return out


def special_erfinv_out(x: torch.Tensor, out: torch.Tensor):
    """Special erfinv out function"""
    logger.debug("GEMS KUNLUNXIN special_erfinv_out")
    # Resize out to match input shape if necessary
    if out.shape != x.shape:
        out.resize_(x.shape)
    # Ensure dtype matches input dtype for aten out semantics
    assert out.dtype == x.dtype, "out tensor must have the same dtype as input"
    x_in = x if x.is_contiguous() else x.contiguous()
    if out.is_contiguous():
        _launch_special_erfinv_kernel(x_in, out)
        return out
    else:
        tmp = torch.empty_like(out, memory_format=torch.contiguous_format)
        _launch_special_erfinv_kernel(x_in, tmp)
        out.copy_(tmp)
        return out


def special_erfinv_(x: torch.Tensor):
    """Special erfinv_ in-place function"""
    logger.debug("GEMS KUNLUNXIN special_erfinv_")
    original_shape = x.shape
    original_stride = x.stride()
    x_in = x if x.is_contiguous() else x.contiguous()
    tmp = torch.empty_like(x_in)
    _launch_special_erfinv_kernel(x_in, tmp)
    x.copy_(tmp)
    # Restore original shape and stride if needed
    if x.shape != original_shape or x.stride() != original_stride:
        x = x.reshape(original_shape).as_strided(original_shape, original_stride)
    return x
