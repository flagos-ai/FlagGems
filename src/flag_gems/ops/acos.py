import logging

import torch
import triton
import triton.language as tl


logger = logging.getLogger(__name__)


"""
Compute element-wise acos using asin:
    acos(x) = HALF_PI - asin(x)
Compute element-wise asin using 15th order Taylor's expansion near 0:
    asin(x) = x + 1/6*x^3 + 3/40*x^5 + 5/112*x^7 + ... + 13!!/(14!!*15)*x^15
So, the Taylor's expansion is accurate enough if and only if x is near 0.
For values of x near -1 or 1, a transformation to values near 0 is needed:
    asin(x) = | arcsin(sqrt(1-x^2)) - HALF_PI,                                  x belongs to (-1, -2^(-0.5))
              | x + 1/6*x^3 + 3/40*x^5 + 5/112*x^7 + ... + 13!!/(14!!*15)*x^15, x belongs to (-2^(-0.5), 2^(-0.5))
              | HALF_PI - arcsin(sqrt(1-x^2)),                                  x belongs to (2^(-0.5), 1)
I couldn't find a convenient way to write constant tensor in triton language as coefficients.
Hence, I use the recursive formula of coefficients:
    C(0) = 1,
    C(i) = C(i-1) * (2*i - 1)**2 / (2*i * (2*i + 1))
"""
@triton.jit()
def acos_kernel(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
    HALF_PI = 1.5707963267948966192313216916398
    TYLOR_COUNT = 8  # Number of items in Taylor's expansion
    coef = 1.0  # 1st coefficient of Taylor's expansion of asin(x)
    pid = tl.program_id(0)
    block_start = BLOCK_SIZE * pid
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    x_square = x * x
    tylor_mask = x_square < 0.5
    positive_mask = x > 0
    x = tl.where(tylor_mask, x, tl.sqrt(1 - x_square))
    x_square = x * x
    half_pi = tl.zeros_like(x) + HALF_PI
    y = tl.zeros_like(x)
    for i in range(TYLOR_COUNT):
        if i > 0:
            coef *= (2 * i - 1) * (2 * i - 1) / (2 * i * (2 * i + 1))
        y += x * tl.cast(coef, x.dtype)
        x *= x_square
    y_far = tl.where(positive_mask, half_pi - y, y - half_pi)  # asin(x), far from 0
    y = tl.where(tylor_mask, y, y_far)  # asin(x)
    y = half_pi - y  # acos(x)
    tl.store(y_ptr + offsets, y, mask=mask)


def acos(x):
    logger.debug("GEMS ACOS FORWARD")
    n = x.numel()
    BLOCK_SIZE = max(
        min(n // 40, 8192), 1024
    )  # empirical value for Atlas A2/A3 series.
    grid = [(n + BLOCK_SIZE - 1) // BLOCK_SIZE]
    y = torch.empty_like(x)

    acos_kernel[grid](x, y, n, BLOCK_SIZE)
    return y
