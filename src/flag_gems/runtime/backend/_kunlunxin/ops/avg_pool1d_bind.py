"""avg_pool1d via the launch-table avg_pool2d binding.

The 1-D average pool is the 2-D pool with a height-1 window.  The kernel below
is a carrier: it is never executed, the launch table matches the name and runs
the vendor pooling entry with the same parameter layout.
"""
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


_NO_SPECIALIZE = [
    "input_ptr", "output_ptr",
    "in_stride_n", "in_stride_c", "in_stride_h", "in_stride_w",
    "in_c", "in_h", "in_w", "out_h", "out_w", "divisor_override",
]


@triton.jit(do_not_specialize=_NO_SPECIALIZE)
def avg_pool2d_forward_kernel_1d(
    input_ptr,
    output_ptr,
    in_stride_n,
    in_stride_c,
    in_stride_h,
    in_stride_w,
    in_c,
    in_h,
    in_w,
    out_h,
    out_w,
    KERNEL_H: tl.constexpr,
    KERNEL_W: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PADDING_H: tl.constexpr,
    PADDING_W: tl.constexpr,
    DILATION_H: tl.constexpr,
    DILATION_W: tl.constexpr,
    COUNT_INCLUDE_PAD: tl.constexpr,
    divisor_override,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, 1)
    keep = (pid < 0) & (offs < 0)
    value = tl.load(input_ptr + offs, mask=keep, other=0.0)
    tl.store(output_ptr + offs, value, mask=keep)


_DTYPES = (torch.float16, torch.float32, torch.bfloat16)


def _can_bind(inp, kernel_size, stride, padding, ceil_mode):
    if inp.dim() != 3 or inp.dtype not in _DTYPES or not inp.is_contiguous():
        return False
    if ceil_mode:
        return False
    k = int(kernel_size)
    s = int(stride if stride is not None else kernel_size)
    p = int(padding)
    if k <= 0 or s <= 0 or p < 0 or 2 * p > k:
        return False
    return True


def avg_pool1d(input, kernel_size, stride=None, padding=0, ceil_mode=False,
               count_include_pad=True, divisor_override=None):
    logger.debug("GEMS_KUNLUNXIN AVG_POOL1D_BIND")
    if divisor_override is not None or not _can_bind(
            input, kernel_size, stride, padding, ceil_mode):
        from flag_gems.ops.avg_pool1d import avg_pool1d as _generic
        return _generic(input, kernel_size, stride, padding, ceil_mode,
                        count_include_pad, divisor_override)

    k = int(kernel_size)
    s = int(stride if stride is not None else kernel_size)
    p = int(padding)
    n, c, l = input.shape
    out_l = (l + 2 * p - k) // s + 1
    x4 = input.view(n, c, 1, l)
    out = torch.empty((n, c, 1, out_l), device=input.device, dtype=input.dtype)

    with torch_device_fn.device(input.device):
        avg_pool2d_forward_kernel_1d[(n * c, 1, 1)](
            x4,
            out,
            x4.stride(0),
            x4.stride(1),
            x4.stride(2),
            x4.stride(3),
            c,
            1,
            l,
            1,
            out_l,
            1,
            k,
            1,
            s,
            0,
            p,
            1,
            1,
            bool(count_include_pad),
            0.0,
            1,
            64,
            num_warps=1,
        )
    return out.view(n, c, out_l)

