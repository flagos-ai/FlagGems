import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def special_modified_bessel_k1_kernel(
    x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)
    ax = tl.abs(x_f32)

    # For x = 0, K1 diverges, return 0 (or large value)
    # Small region: |x| <= 3.75
    # Polynomial approximation from Cephes
    y = x_f32 / 3.75
    y2 = y * y

    # Polynomial for small |x|
    p = 0.00032411
    p = 0.00301532 + y2 * p
    p = 0.02658733 + y2 * p
    p = 0.15084934 + y2 * p
    p = 0.51498869 + y2 * p
    p = 0.87890594 + y2 * p
    p = 0.5 + y2 * p

    # Small x: K1(x) ~ 1/x for very small x
    # Using the polynomial approximation
    ans_small = x_f32 * p

    # Large region: |x| > 3.75
    # Use asymptotic expansion: K1(x) ~ sqrt(pi/(2x)) * exp(-x) * poly(1/x)
    t = 3.75 / tl.maximum(ax, 1e-20)
    q = -0.00420059
    q = 0.01787654 + t * q
    q = -0.02895312 + t * q
    q = 0.02282967 + t * q
    q = -0.01031555 + t * q
    q = 0.00163801 + t * q
    q = -0.00362018 + t * q
    q = -0.03988024 + t * q
    q = 0.39894228 + t * q

    # Prefactor: sqrt(pi/(2x)) * exp(-x)
    pref = tl.exp(-ax) / tl.sqrt(tl.maximum(ax, 1e-20) * 1.5707963267948966)
    ans_large = pref * q

    is_small = ax <= 3.75
    ans = tl.where(is_small, ans_small, ans_large)

    # Handle x = 0: K1(0) = inf
    ans = tl.where(ax < 1e-30, float("inf"), ans)

    # Cast back to input dtype and store
    tl.store(out_ptr + offsets, ans.to(x.dtype), mask=mask)


def _launch_special_modified_bessel_k1(x: torch.Tensor, out: torch.Tensor):
    assert (
        x.numel() == out.numel()
    ), "Input and output must have the same number of elements"
    assert x.dtype == out.dtype, "Input and output must have the same dtype"

    n_elements = x.numel()
    if n_elements == 0:
        return

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(x.device):
        special_modified_bessel_k1_kernel[grid](
            x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE
        )


def special_modified_bessel_k1(self: torch.Tensor):
    logger.debug("GEMS_ILUVATAR SPECIAL_MODIFIED_BESSEL_K1")
    x = self
    x_c = x.contiguous()
    out = torch.empty_like(x_c)
    _launch_special_modified_bessel_k1(x_c, out)
    if x.layout == torch.strided and x.is_contiguous():
        return out
    else:
        return out.view_as(x)


def special_modified_bessel_k1_out(self: torch.Tensor, out: torch.Tensor):
    logger.debug("GEMS_ILUVATAR SPECIAL_MODIFIED_BESSEL_K1_OUT")
    x = self
    if out.dtype != x.dtype:
        raise TypeError("out dtype must match input dtype")
    if out.device != x.device:
        raise TypeError("out device must match input device")

    x_c = x.contiguous()
    out_c = out.contiguous()
    _launch_special_modified_bessel_k1(x_c, out_c)
    if out_c.data_ptr() != out.data_ptr():
        out.copy_(out_c)
    return out
