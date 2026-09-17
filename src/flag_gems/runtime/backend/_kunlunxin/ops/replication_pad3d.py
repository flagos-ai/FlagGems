import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


# Flat 1D kernel over the ENTIRE output (all N,C,D,H,W at once).
#
# ROOT CAUSE of the old slowness (generic ops/replication_pad3d.py): it used
# `@libtuner(key=["H_out","W_out"])` to supply BLOCK_H/BLOCK_W at launch and a
# 3D grid of one 2D (H,W) tile per (n,c,d) plane. On KunlunXin XPU the launch
# path autotune triggers per-key recompiles, and the per-plane base pointer
# `... + iz*stride_xd` combined with `pid * runtime_stride` addressing keeps
# every plane on the discrete path. Baseline shapes ran at speedup 0.006-0.074.
#
# Fix: flatten the whole output into one linear index `o = pid*BLOCK + arange`
# and store to `o` directly (base = pid * constexpr BLOCK -> provably stride-1
# block DMA). A single `o < total_out` mask handles the tail. The replication
# clamp on the load side is a data-dependent gather (unavoidable), but the store
# is fully contiguous and the kernel compiles once (no libtuner recompile).
@triton.jit
def replication_pad3d_kernel(
    x_ptr,
    out_ptr,
    D_in,
    H_in,
    W_in,
    D_out,
    H_out,
    W_out,
    pad_l,
    pad_t,
    pad_f,
    DHW_in,
    HW_in,
    DHW_out,
    HW_out,
    total_out,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    mask = o < total_out

    # Decode flat output index -> (nc, d_out, h_out, w_out)
    nc = o // DHW_out
    rem = o % DHW_out
    d_out = rem // HW_out
    rem2 = rem % HW_out
    h_out = rem2 // W_out
    w_out = rem2 % W_out

    # Replication clamp (edge padding): clamp each axis into the valid input range.
    iz = d_out.to(tl.int32) - pad_f
    iz = tl.where(iz < 0, 0, iz)
    iz = tl.where(iz > D_in - 1, D_in - 1, iz)

    iy = h_out.to(tl.int32) - pad_t
    iy = tl.where(iy < 0, 0, iy)
    iy = tl.where(iy > H_in - 1, H_in - 1, iy)

    ix = w_out.to(tl.int32) - pad_l
    ix = tl.where(ix < 0, 0, ix)
    ix = tl.where(ix > W_in - 1, W_in - 1, ix)

    in_offs = nc * DHW_in + iz * HW_in + iy * W_in + ix
    vals = tl.load(x_ptr + in_offs, mask=mask)
    tl.store(out_ptr + o, vals, mask=mask)


def replication_pad3d(x, padding):
    logger.debug("GEMS_KUNLUNXIN REPLICATION_PAD3D")
    if isinstance(padding, int):
        pad_l = pad_r = pad_t = pad_b = pad_f = pad_ba = padding
    else:
        pad_l, pad_r, pad_t, pad_b, pad_f, pad_ba = padding

    is_4d = x.ndim == 4
    if is_4d:
        x = x.unsqueeze(0)

    x = x.contiguous()
    N, C, D_in, H_in, W_in = x.shape
    D_out, H_out, W_out = (
        D_in + pad_f + pad_ba,
        H_in + pad_t + pad_b,
        W_in + pad_l + pad_r,
    )

    out = torch.empty((N, C, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    HW_in = H_in * W_in
    DHW_in = D_in * HW_in
    HW_out = H_out * W_out
    DHW_out = D_out * HW_out
    total_out = N * C * DHW_out

    BLOCK = 1024
    grid = (triton.cdiv(total_out, BLOCK),)
    with torch_device_fn.device(x.device):
        replication_pad3d_kernel[grid](
            x,
            out,
            D_in,
            H_in,
            W_in,
            D_out,
            H_out,
            W_out,
            pad_l,
            pad_t,
            pad_f,
            DHW_in,
            HW_in,
            DHW_out,
            HW_out,
            total_out,
            BLOCK=BLOCK,
        )

    return out.squeeze(0) if is_4d else out
