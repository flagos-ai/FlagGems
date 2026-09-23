import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

BLOCK = 1024
BIG_BLOCK = 16384
TINY_NUMEL = 65536


@libentry()
@triton.jit
def diff_row_kernel(
    in_ptr,
    out_ptr,
    NCOMP,
    BLOCK: tl.constexpr,
    CAST16: tl.constexpr,
    RNE_BF16: tl.constexpr,
):
    pid_row = tle.program_id(0)
    pid_chunk = tle.program_id(1)
    in_base = pid_row.to(tl.int64) * (NCOMP + 1)
    out_base = pid_row.to(tl.int64) * NCOMP
    offs = pid_chunk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < NCOMP
    a = tl.load(in_ptr + in_base + offs, mask=mask)
    b = tl.load(in_ptr + in_base + offs + 1, mask=mask)
    if CAST16:
        d = (b.to(tl.int32) - a.to(tl.int32)).to(a.dtype)
    elif RNE_BF16:
        t = b.to(tl.float32) - a.to(tl.float32)
        tbits = t.to(tl.uint32, bitcast=True)
        tbits = (tbits + 0x7FFF + ((tbits >> 16) & 1)) & 0xFFFF0000
        d = tbits.to(tl.float32, bitcast=True).to(tl.bfloat16)
    else:
        d = b - a
    tl.store(out_ptr + out_base + offs, d, mask=mask)


def diff(input, n=1, dim=-1, prepend=None, append=None) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN DIFF")

    if prepend is not None:
        input = torch.cat([prepend, input], dim=dim)
    if append is not None:
        input = torch.cat([input, append], dim=dim)

    if n <= 0:
        return input

    shape = list(input.shape)
    dim = dim % input.ndim
    reduce_len = shape[dim]

    if n >= reduce_len:
        empty_tensor = torch.tensor([], dtype=input.dtype, device=input.device)
        return torch.reshape(empty_tensor, shape[:dim] + [0] + shape[(dim + 1) :])

    if (
        n == 1 or (n == 2 and input.dtype != torch.bfloat16)
    ) and input.numel() > TINY_NUMEL:
        out = input
        for _ in range(n):
            idx_hi = [slice(None)] * out.ndim
            idx_hi[dim] = slice(1, None)
            idx_lo = [slice(None)] * out.ndim
            idx_lo[dim] = slice(0, -1)
            out = out[tuple(idx_hi)] - out[tuple(idx_lo)]
        return out

    input = dim_compress(input, dim)
    N = reduce_len
    M = input.numel() // N
    block = BIG_BLOCK if N >= 4096 else BLOCK

    def _launch(src, dst, n_comp):
        grid = (M, triton.cdiv(n_comp, block))
        with torch_device_fn.device(src.device):
            diff_row_kernel[grid](
                src,
                dst,
                n_comp,
                BLOCK=block,
                CAST16=bool(src.dtype == torch.int16),
                RNE_BF16=bool(src.dtype == torch.bfloat16),
                buffer_size_limit=2048,
            )

    out_shape = list(input.shape)
    out_shape[-1] = N - n
    output = torch.empty(out_shape, device=input.device, dtype=input.dtype)

    if n == 1:
        _launch(input, output, N - 1)
        return torch.moveaxis(output, -1, dim)

    scratch_a_shape = list(input.shape)
    scratch_a_shape[-1] = N - 1
    scratch_a = torch.empty(scratch_a_shape, device=input.device, dtype=input.dtype)
    if n >= 3:
        scratch_b_shape = list(input.shape)
        scratch_b_shape[-1] = N - 2
        scratch_b = torch.empty(scratch_b_shape, device=input.device, dtype=input.dtype)

    _launch(input, scratch_a, N - 1)
    src = scratch_a

    for k in range(1, n):
        if k == n - 1:
            dst = output
        elif k % 2 == 1:
            dst = scratch_b
        else:
            dst = scratch_a
        _launch(src, dst, N - k - 1)
        src = dst

    return torch.moveaxis(output, -1, dim)
