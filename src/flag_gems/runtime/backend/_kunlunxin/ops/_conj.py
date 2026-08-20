# Kunlunxin (XPU) override of `_conj`.
#
# `_conj(A)` returns the complex conjugate: out = (re, -im).
#
# The generic path (`flag_gems/ops/_conj.py`) launches a kernel that indexes the
# real/imag parts with `base = offsets * 2` and loads `base` (real) and
# `base + 1` (imag) as two SEPARATE stride-2 passes over the interleaved complex
# storage. On XPU a stride-2 gather is discrete access -> the block DMA engine
# degrades to per-element loads and bandwidth collapses. IR baseline
# `harness/perf_ir_4/ir-conj-dev5.log`: gems speedup 0.000-0.21
# ([4096,4096] 52ms, [1024,65536] 202ms), avg 0.0417.
#
# Fix: for a *contiguous* complex tensor the raw storage is one contiguous real
# stream [re0, im0, re1, im1, ...]. A single 1D contiguous kernel copies it while
# negating the odd (imaginary) lanes -> ONE stride-1 block DMA in and out (no
# strided access). This mirrors the already-landed `resolve_conj` fix. The
# `i % 2` test only selects which loaded VALUE to negate; it never appears in an
# address, so the load/store addresses stay affine (stride 1). Non-contiguous
# inputs fall back to the correct generic materialize path.
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _conj_flat_kernel(fin, fout, n2, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n2
    x = tl.load(fin + i, mask=m)
    out = tl.where((i % 2) == 1, -x, x)
    tl.store(fout + i, out, mask=m)


def _conj(input: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN CONJ")
    if not input.is_complex():
        raise RuntimeError("_conj only supports complex tensors")

    src = input if input.is_contiguous() else input.contiguous()
    out = torch.empty_like(src)

    # view_as_real exposes the interleaved real storage as a trailing-dim-2 real
    # tensor (float32 for complex64, float16 for complex32). Flatten to a
    # contiguous 1D stream so the kernel does pure stride-1 block DMA.
    fin = torch.view_as_real(src).reshape(-1)
    fout = torch.view_as_real(out).reshape(-1)
    n2 = fin.numel()

    BLOCK = 8192
    grid = (triton.cdiv(n2, BLOCK),)
    with torch_device_fn.device(src.device):
        _conj_flat_kernel[grid](fin, fout, n2, BLOCK=BLOCK, num_warps=8)

    return out
