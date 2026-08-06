# Copyright 2026 FlagOS Contributors
#
# Kunlunxin (XPU) override of histc.
#
# Root cause: the generic implementation counts via masked `tl.atomic_add`
# (histc_kernel_simple).  On XPU the masked atomic-add path silently drops a
# fraction (~1%) of the updates, so every bin comes out systematically low
# (e.g. res 320 vs ref 327 for the (32,64,16) case) -> 20 fp32 cases fail.
#
# Fix: avoid atomic-add entirely.  Launch one program per bin; each program
# streams the whole input in BLOCK chunks and sums how many elements fall in
# its bin using a plain `tl.sum` reduction.  bins <= 100 in the test suite so
# the extra passes over the data are cheap and, crucially, exact.
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def histc_bin_kernel(
    inp_ptr,
    out_ptr,
    n_elements,
    bins,
    min_val,
    max_val,
    BLOCK_SIZE: tl.constexpr,
):
    b = ext.program_id(0)
    inv_scale = bins / (max_val - min_val)
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    for start in range(0, n_elements, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        v = tl.load(inp_ptr + offs, mask=mask, other=float("nan")).to(tl.float32)
        idx = tl.floor((v - min_val) * inv_scale).to(tl.int32)
        # elements exactly at max_val belong to the last bin
        idx = tl.where(v == max_val, bins - 1, idx)
        in_range = (v >= min_val) & (v <= max_val)
        hit = mask & in_range & (idx == b)
        acc += hit.to(tl.int32)
    total = tl.sum(acc)
    tl.store(out_ptr + b, total.to(tl.float32))


def histc(inp, bins=100, min=0, max=0):
    logger.debug("GEMS_KUNLUNXIN HISTC")

    inp = inp.contiguous()

    min_val = float(min)
    max_val = float(max)

    if min_val == 0 and max_val == 0:
        min_val = float(inp.min().item())
        max_val = float(inp.max().item())

    if min_val == max_val:
        out = torch.zeros(bins, dtype=inp.dtype, device=inp.device)
        count = ((inp == min_val) & ~torch.isnan(inp)).sum().item()
        out[0] = count
        return out

    out = torch.zeros(bins, dtype=inp.dtype, device=inp.device)

    n_elements = inp.numel()
    if n_elements == 0:
        return out

    BLOCK_SIZE = 1024
    grid = (bins,)

    with torch_device_fn.device(inp.device):
        histc_bin_kernel[grid](
            inp,
            out,
            n_elements,
            bins,
            min_val,
            max_val,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return out
