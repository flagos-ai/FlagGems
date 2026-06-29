import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)
NUM_SIPS = 24


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def _alias_copy_kernel(src_ptr, dst_ptr, N_total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    for block_id in tl.range(pid, (N_total + BLOCK - 1) // BLOCK, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        vals = tl.load(src_ptr + off, mask=mask)
        tl.store(dst_ptr + off, vals, mask=mask)


def _launch_copy(src, dst, N):
    BLOCK = 8192
    grid = min((N + BLOCK - 1) // BLOCK, NUM_SIPS * 2)
    with torch_device_fn.device(src.device):
        _alias_copy_kernel[(grid,)](src, dst, N, BLOCK=BLOCK, num_warps=4)


def alias_copy(x: torch.Tensor):
    logger.debug("GEMS_ENFLAME ALIAS_COPY")
    out = torch.empty_like(x)
    n_elements = out.numel()
    if n_elements == 0:
        return out
    src = x.contiguous() if not x.is_contiguous() else x
    if not out.is_contiguous():
        out = out.contiguous()
    if src.dtype != out.dtype:
        raise RuntimeError("alias_copy: dtype mismatch between input and output.")
    _launch_copy(src, out, n_elements)
    return out


def alias_copy_out(x: torch.Tensor, out: torch.Tensor):
    logger.debug("GEMS_ENFLAME ALIAS_COPY_OUT")
    if x.dtype != out.dtype:
        raise RuntimeError("alias_copy_out: dtype of input and output must match.")
    if x.numel() != out.numel():
        raise RuntimeError(
            "alias_copy_out: input and output must have the same number of elements."
        )
    if x.device != out.device:
        raise RuntimeError(
            "alias_copy_out: input and output must be on the same device."
        )
    if not out.is_contiguous():
        raise RuntimeError("alias_copy_out: output tensor must be contiguous.")
    src = x.contiguous() if not x.is_contiguous() else x
    n_elements = out.numel()
    if n_elements == 0:
        return out
    _launch_copy(src, out, n_elements)
    return out
