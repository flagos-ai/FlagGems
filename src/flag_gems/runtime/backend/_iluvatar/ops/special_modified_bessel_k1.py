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
    # Modified Bessel function of the second kind, order 1: K_1(x)
    # Implementation using series + asymptotic with linear blend.
    # Series: accurate for x < 0.5
    # Asymptotic: accurate for x > 1.5
    # Linear blend in between.

    gamma = 0.577215664901532860606512090082402431042159335
    sqrt_pi_over_2 = 1.2533141373155002512078826424055  # sqrt(pi/2)

    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)

    # Handle non-positive values
    is_non_positive = x_f32 <= 0.0
    x_safe = tl.where(x_f32 <= 0.0, 0.1, x_f32)

    # Compute log(x/2) and related terms
    log_half = tl.log(x_safe / 2.0)
    ln_term = log_half + gamma

    # Series expansion for very small x (x < 0.5)
    x2 = x_safe * x_safe
    x4 = x2 * x2
    x6 = x4 * x2
    x8 = x4 * x4
    x10 = x8 * x2

    term1 = 1.0 / x_safe
    term2 = x_safe / 2.0 * (ln_term - 0.5)
    term3 = x2 * x_safe / 16.0 * (ln_term - 5.0 / 6.0)
    term4 = x4 * x_safe / 128.0 * (ln_term - 23.0 / 24.0)
    term5 = x6 * x_safe / 2304.0 * (ln_term - 235.0 / 276.0)
    term6 = x8 * x_safe / 41472.0 * (ln_term - 1469.0 / 1560.0)
    term7 = x10 * x_safe / 74304.0 * (ln_term - 7519.0 / 7080.0)

    series_result = term1 + term2 + term3 + term4 + term5 + term6 + term7

    # Asymptotic expansion for larger x (x > 1.5)
    x_inv = 1.0 / x_safe
    x_inv2 = x_inv * x_inv
    x_inv3 = x_inv2 * x_inv

    asymp_correction = (
        1.0 + 3.0 / 8.0 * x_inv - 15.0 / 128.0 * x_inv2 + 105.0 / 1024.0 * x_inv3
    )
    asymp_result = sqrt_pi_over_2 / tl.sqrt(x_safe) * tl.exp(-x_safe) * asymp_correction

    # Linear blend from series to asymptotic
    # At x <= 0.5: blend = 1.0 (series only)
    # At x >= 1.5: blend = 0.0 (asymptotic only)
    # Between 0.5 and 1.5: linear interpolation
    blend = tl.where(
        x_safe <= 0.5, 1.0, tl.where(x_safe >= 1.5, 0.0, 1.0 - (x_safe - 0.5) / 1.0)
    )

    result = series_result * blend + asymp_result * (1.0 - blend)

    # Handle non-positive values: compute fp32 NaN without fp64 constant
    # to avoid IXRTC ERROR_UNSUPPORTED_CAST
    zero = x_f32 - x_f32
    result = tl.where(is_non_positive, zero / zero, result)

    # Cast back to input dtype and store
    tl.store(out_ptr + offsets, result.to(x.dtype), mask=mask)


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
