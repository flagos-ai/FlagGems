import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(f'flag_gems.runtime._ascend.ops.{__name__.split(".")[-1]}')


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("complex"), key=["N2"])
@triton.jit
def complex_kernel_flat(
    real_ptr,
    imag_ptr,
    out_ptr,
    N2,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < N2

    src_idx = idx >> 1
    is_imag = (idx % 2) != 0

    r_val = tl.load(real_ptr + src_idx, mask=(src_idx < (N2 >> 1)), other=0.0)
    i_val = tl.load(imag_ptr + src_idx, mask=(src_idx < (N2 >> 1)), other=0.0)

    res = tl.where(is_imag, i_val, r_val)
    tl.store(out_ptr + idx, res, mask=mask)


def complex(real, imag):
    requested_out_dtype = (
        torch.complex64 if real.dtype == torch.float32 else torch.complex128
    )

    if real.dtype == torch.float64:
        real = real.to(torch.float32)
        imag = imag.to(torch.float32)

    base_dtype = torch.float32
    kernel_out_dtype = torch.complex64

    orig_shape = real.shape
    real_flat = real.reshape(-1).contiguous()
    imag_flat = imag.reshape(-1).contiguous()

    N = real_flat.numel()
    N2 = 2 * N

    out_flat = torch.empty(N, dtype=kernel_out_dtype, device=real.device)
    out_view = out_flat.view(base_dtype)

    def grid(meta):
        return (triton.cdiv(N2, meta["BLOCK_SIZE"]),)

    with torch_device_fn.device(real.device):
        complex_kernel_flat[grid](
            real_flat,
            imag_flat,
            out_view,
            N2,
        )

    res = out_flat.reshape(orig_shape)

    if res.dtype != requested_out_dtype:
        res = res.to(requested_out_dtype)

    return res
