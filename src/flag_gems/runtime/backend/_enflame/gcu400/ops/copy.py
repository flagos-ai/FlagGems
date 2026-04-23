import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

NUM_SIPS = 24

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)

_SMALL_THRESHOLD = 8192


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def copy_flat_kernel(
    src_ptr,
    dst_ptr,
    N_total,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)

    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        x = tl.load(src_ptr + off, mask=mask)
        tl.store(dst_ptr + off, x, mask=mask)


def _choose_block(N_total, itemsize):
    if itemsize <= 1:
        if N_total <= 65536:
            return 4096
        return 8192
    if N_total <= 65536:
        return 8192
    if N_total <= 524288:
        return 32768
    return 65536


def _run_copy_kernel(src, dst, N_total):
    BLOCK = _choose_block(N_total, dst.element_size())
    NUM_BLOCKS = triton.cdiv(N_total, BLOCK)
    grid_size = min(NUM_BLOCKS, NUM_SIPS * 2)

    with torch_device_fn.device(src.device):
        copy_flat_kernel[(grid_size,)](
            src, dst, N_total,
            BLOCK=BLOCK,
            num_warps=8,
        )


def _redispatch(dst, src, non_blocking):
    return torch.ops.aten.copy_.default.redispatch(
        _FALLBACK_KEYSET, dst, src, non_blocking
    )


def copy(
    template: torch.Tensor, src: torch.Tensor, *, non_blocking=False
):
    logger.debug("GEMS COPY (functional)")
    out = torch.empty_strided(
        template.size(), template.stride(),
        dtype=template.dtype, device=template.device,
    )
    copy_(out, src, non_blocking=bool(non_blocking))
    return out


def copy_(dst: torch.Tensor, src, non_blocking: bool = False):
    if not isinstance(src, torch.Tensor):
        return _redispatch(dst, src, non_blocking)

    N = dst.numel()

    if N <= _SMALL_THRESHOLD:
        return _redispatch(dst, src, non_blocking)

    if (
        N <= 2**31 - 1
        and dst.element_size() >= 2
        and dst.is_contiguous()
        and src.is_contiguous()
        and dst.dtype == src.dtype
        and dst.shape == src.shape
        and dst.device == src.device
        and dst.layout == torch.strided
        and not dst.is_quantized
        and not src.is_complex()
        and not torch._C._is_alias_of(dst, src)
    ):
        _run_copy_kernel(src, dst, N)
        return dst

    return _redispatch(dst, src, non_blocking)
