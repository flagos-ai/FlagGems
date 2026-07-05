import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle
from ..ops.copy import copy


@libentry()
@triton.jit
def contiguous_copy_2d_kernel(
    inp_ptr,
    out_ptr,
    M,
    N,
    inp_stride0,
    inp_stride1,
    out_stride0,
    out_stride1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tle.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols = tle.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    mask = (rows < M) & (cols < N)
    vals = tl.load(
        inp_ptr + rows * inp_stride0 + cols * inp_stride1,
        mask=mask,
        other=0.0,
    )
    tl.store(
        out_ptr + rows * out_stride0 + cols * out_stride1,
        vals,
        mask=mask,
    )


def contiguous(inp, memory_format=torch.contiguous_format):
    assert memory_format == torch.contiguous_format
    logging.debug("GEMS CONTIGUOUS")
    if inp.is_contiguous(memory_format=memory_format):
        return inp
    if inp.ndim == 2:
        out = torch.empty_like(inp, memory_format=memory_format)
        M, N = inp.shape
        grid = (triton.cdiv(M, 32), triton.cdiv(N, 128))
        with torch_device_fn.device(inp.device):
            contiguous_copy_2d_kernel[grid](
                inp,
                out,
                M,
                N,
                inp.stride(0),
                inp.stride(1),
                out.stride(0),
                out.stride(1),
                BLOCK_M=32,
                BLOCK_N=128,
            )
        return out
    out = torch.empty_like(inp, memory_format=memory_format)
    return copy(inp, out0=out)
