"""Kunlunxin (XPU) override of ``aten::replication_pad3d_backward``.

Performance / correctness notes (2026-09-10, device XPU 0):
- The previous implementation scattered ``grad_output`` into the input with
  ``tl.atomic_add`` over a flat (batch, output_voxels) grid. On the XPU
  backend ``tl.atomic_add`` is both pathologically slow (the benchmark
  measured 21-52 ms for ~0.5M elements, ~50x-300x slower than the torch
  reference) and, per the ``replication_pad2d_backward`` vendor notes,
  silently drops updates (non-deterministic lost contributions). This
  implementation is therefore ATOMIC-FREE.
- ``replication_pad3d_backward`` is a many-to-one map: output position
  ``z = clamp(d_out - pad_front, 0, D-1)`` etc.  The backward is a gather
  instead of a scatter: each input voxel ``(d, h, w)`` receives the sum of
  ``grad_output`` over the *box* of output positions mapping to it.  The
  group for input index ``i`` (size ``n``, output size ``m = n + p``) is a
  contiguous range ``[lo, lo + cnt)`` where
      i == 0     -> lo = 0,                          cnt = (n==1 ? m : p+1)
      i == n-1   -> lo = p + n - 1,                  cnt = m - lo  (0 if lo>=m)
      otherwise  -> lo = p + i,                      cnt = 1 if 0<=lo<m else 0
  (lo is clamped to ``[0, m-1]``; ``cnt`` may be 0 under cropping pads).
  These formulas handle negative padding (crop) as well.
- One program handles ``BLOCK_DHW`` input voxels of one (N, C) batch:
  grid = (cdiv(D*H*W, BLOCK_DHW), N*C).  Every ``grad_input`` cell is
  written by exactly ONE program, from a disjoint, complete partition of
  ``grad_output``.
- All shape dims are ``tl.constexpr`` so the per-lane decodes become shifts
  and all address arithmetic stays int32 (the previous flat kernel used
  runtime dims and paid int64 div/mod + long address chains).
- The group loops are DYNAMIC (``range(0, maxg_x)`` with a runtime bound) and
  the kernel is launched with ``isCloseUnrollControl=True``: the fully
  unrolled static form blows the XPU compiler's ``uni_sram`` budget for
  large pad groups (e.g. five-dimensional ``m`` when an input dim is 1:
  a 5x5x5 static unroll raises "Failed to tune buffer size").  With a
  dynamic loop every lane still executes ``maxg`` predicated iterations,
  so the interior (group size 1) fast case is unchanged while the code
  stays compact for any padding.
- Loads are unconditional and in-bounds: the group start ``lo`` is clamped
  to ``[0, m-1]`` and the loop offset ``min(r, max(cnt-1, 0))`` keeps every
  address inside the tensor even for lanes masked out by the tail; per-
  contribution validity is re-applied with a value-level ``tl.where`` (the
  XPU backend's masked-load ``other`` handling is unreliable).  Only the
  final store is masked.
- fp32 register accumulation and a single explicit ``.to(OUT_DTYPE)`` cast
  on the store, matching the reference opmath (no fp32 intermediate buffer,
  no extra cast pass).
"""

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _replication_pad3d_backward_kernel(
    grad_output_ptr,
    grad_input_ptr,
    pad_left,
    pad_right,
    pad_top,
    pad_bottom,
    pad_front,
    pad_back,
    maxg_d,
    maxg_h,
    maxg_w,
    OUT_DTYPE: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    OD: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    BLOCK_DHW: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    bc = tl.program_id(1).to(tl.int32)

    offs = pid * BLOCK_DHW + tl.arange(0, BLOCK_DHW)
    mask = offs < D * H * W

    hw = H * W
    d = offs // hw
    h = (offs // W) % H
    w = offs % W

    xw_raw = tl.where(w == 0, 0, tl.where(w == W - 1, pad_left + W - 1, pad_left + w))
    xw = tl.minimum(tl.maximum(xw_raw, 0), OW - 1)
    cnt_w = tl.where(
        w == 0,
        tl.where(W == 1, OW, tl.maximum(pad_left + 1, 0)),
        tl.where(
            w == W - 1,
            tl.where(xw_raw >= OW, 0, OW - tl.maximum(xw_raw, 0)),
            tl.where((xw_raw >= 0) & (xw_raw < OW), 1, 0),
        ),
    )
    yh_raw = tl.where(h == 0, 0, tl.where(h == H - 1, pad_top + H - 1, pad_top + h))
    yh = tl.minimum(tl.maximum(yh_raw, 0), OH - 1)
    cnt_h = tl.where(
        h == 0,
        tl.where(H == 1, OH, tl.maximum(pad_top + 1, 0)),
        tl.where(
            h == H - 1,
            tl.where(yh_raw >= OH, 0, OH - tl.maximum(yh_raw, 0)),
            tl.where((yh_raw >= 0) & (yh_raw < OH), 1, 0),
        ),
    )
    zd_raw = tl.where(d == 0, 0, tl.where(d == D - 1, pad_front + D - 1, pad_front + d))
    zd = tl.minimum(tl.maximum(zd_raw, 0), OD - 1)
    cnt_d = tl.where(
        d == 0,
        tl.where(D == 1, OD, tl.maximum(pad_front + 1, 0)),
        tl.where(
            d == D - 1,
            tl.where(zd_raw >= OD, 0, OD - tl.maximum(zd_raw, 0)),
            tl.where((zd_raw >= 0) & (zd_raw < OD), 1, 0),
        ),
    )

    gb = bc * (OD * OH * OW)
    ob = bc * (D * H * W)

    acc = tl.zeros((BLOCK_DHW,), dtype=tl.float32)
    for r in range(0, maxg_d):
        rr = tl.minimum(r, tl.maximum(cnt_d - 1, 0))
        row_base = gb + (zd + rr) * (OH * OW)
        sel_d = r < cnt_d
        for c in range(0, maxg_h):
            cc = tl.minimum(c, tl.maximum(cnt_h - 1, 0))
            col_base = row_base + (yh + cc) * OW
            sel_dh = sel_d & (c < cnt_h)
            for s in range(0, maxg_w):
                ss = tl.minimum(s, tl.maximum(cnt_w - 1, 0))
                v = tl.load(grad_output_ptr + col_base + xw + ss).to(tl.float32)
                acc += tl.where(sel_dh & (s < cnt_w), v, 0.0)

    dst = ob + d * hw + h * W + w
    tl.store(grad_input_ptr + dst, acc.to(OUT_DTYPE), mask=mask)


def replication_pad3d_backward(
    grad_output: torch.Tensor, self: torch.Tensor, padding
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN REPLICATION_PAD3D_BACKWARD")

    if not isinstance(padding, (list, tuple)) or len(padding) != 6:
        raise ValueError("padding must contain six values")
    if self.dim() < 3:
        raise ValueError("self must have at least three dimensions")
    if grad_output.device != self.device or grad_output.dtype != self.dtype:
        raise ValueError("grad_output and self must have the same device and dtype")
    if self.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("replication_pad3d_backward supports floating point dtypes")

    pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back = map(int, padding)
    x = self.contiguous()
    go = grad_output.contiguous()

    D = int(x.shape[-3])
    H = int(x.shape[-2])
    W = int(x.shape[-1])
    OD = D + pad_front + pad_back
    OH = H + pad_top + pad_bottom
    OW = W + pad_left + pad_right
    if OD <= 0 or OH <= 0 or OW <= 0:
        raise ValueError("padding results in a non-positive output dimension")
    if tuple(go.shape[-3:]) != (OD, OH, OW):
        raise ValueError(
            "grad_output spatial shape "
            f"{tuple(go.shape[-3:])} does not match {(OD, OH, OW)}"
        )
    if tuple(go.shape[:-3]) != tuple(x.shape[:-3]):
        raise ValueError("grad_output and self must have matching leading dimensions")

    if (
        pad_left == 0
        and pad_right == 0
        and pad_top == 0
        and pad_bottom == 0
        and pad_front == 0
        and pad_back == 0
    ):
        return go.reshape(x.shape)

    if D * H * W == 0:
        return torch.zeros_like(x)

    nzc = math.prod(x.shape[:-3]) if x.dim() > 3 else 1

    out = torch.empty_like(x)
    if self.dtype == torch.float16:
        out_dtype = tl.float16
    elif self.dtype == torch.bfloat16:
        out_dtype = tl.bfloat16
    else:
        out_dtype = tl.float32

    maxg_w = max(pad_left + 1, pad_right + 1, 1)
    if W == 1:
        maxg_w = max(maxg_w, OW)
    maxg_h = max(pad_top + 1, pad_bottom + 1, 1)
    if H == 1:
        maxg_h = max(maxg_h, OH)
    maxg_d = max(pad_front + 1, pad_back + 1, 1)
    if D == 1:
        maxg_d = max(maxg_d, OD)

    block = 256 if D * H * W <= 4096 else 512
    grid = (triton.cdiv(D * H * W, block), nzc)
    with torch_device_fn.device(x.device):
        _replication_pad3d_backward_kernel[grid](
            go,
            out,
            pad_left,
            pad_right,
            pad_top,
            pad_bottom,
            pad_front,
            pad_back,
            maxg_d,
            maxg_h,
            maxg_w,
            OUT_DTYPE=out_dtype,
            D=D,
            H=H,
            W=W,
            OD=OD,
            OH=OH,
            OW=OW,
            BLOCK_DHW=block,
            isCloseUnrollControl=True,
        )
    return out
