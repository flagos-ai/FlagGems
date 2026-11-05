import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger("flag_gems.ops.channel_shuffle")

_MAX_BLOCK = 4096


@libentry()
@triton.jit
def channel_shuffle_kernel(
    in_ptr,
    out_ptr,
    s,
    cg,
    g,
    BLOCK: tl.constexpr,
    I64: tl.constexpr,
):
    """Each program copies one contiguous row of ``s`` elements: out[n,i,j,:] = x[n,j,i,:]."""
    sb = tl.program_id(0)
    i = tl.program_id(1)
    nj = tl.program_id(2)

    j = nj % g
    n = nj // g

    # Channel indices are < C and fit in int32; it is the multiplication by the
    # per-channel element count ``s`` that can exceed int32 on tensors with
    # >= 2**31 elements.  The promotion to int64 must therefore happen *before*
    # that product -- casting the product afterwards would only sign-extend a
    # value that has already wrapped around.
    if I64:
        src_c = (n.to(tl.int64) * g + j) * cg + i
        dst_c = (n.to(tl.int64) * cg + i) * g + j
        offs = sb.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    else:
        src_c = (n * g + j) * cg + i
        dst_c = (n * cg + i) * g + j
        offs = sb * BLOCK + tl.arange(0, BLOCK)
    src_base = src_c * s
    dst_base = dst_c * s

    mask = offs < s
    val = tl.load(in_ptr + src_base + offs, mask=mask)
    tl.store(out_ptr + dst_base + offs, val, mask=mask)


def channel_shuffle(input: torch.Tensor, groups: int) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN CHANNEL_SHUFFLE")

    x = input
    if x.ndim < 3:
        raise ValueError(
            f"Input must have at least 3 dimensions (C, H, W), got {x.ndim}"
        )

    n, c = x.shape[0], x.shape[1]
    g = int(groups)
    assert g > 0, "groups must be > 0"
    assert c % g == 0, f"C ({c}) must be divisible by groups ({g})"

    if not x.is_contiguous():
        x = x.contiguous()

    out = torch.empty_like(x)
    if x.numel() == 0:
        return out

    cg = c // g
    s = x.numel() // (n * c)
    block = min(triton.next_power_of_2(s), _MAX_BLOCK)
    grid = (triton.cdiv(s, block), cg, n * g)
    with torch_device_fn.device(x.device):
        channel_shuffle_kernel[grid](
            x,
            out,
            s,
            cg,
            g,
            BLOCK=block,
            I64=x.numel() >= 2**31,
        )
    return out
