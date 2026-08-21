import logging
from contextlib import nullcontext

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

TILE_BYTES = 64 * 1024
TILE_FP32_CAP = 64 * 1024


@triton.jit
def _backward_gather_kernel(
    gop,
    gip,
    CO,
    OH,
    OW,
    H,
    W,
    PL: tl.constexpr,
    PR: tl.constexpr,
    PT: tl.constexpr,
    PB: tl.constexpr,
    R: tl.constexpr,
    WN: tl.constexpr,
):
    pid = ext.program_id(0)
    n_row_blocks = (H + R - 1) // R
    nc = CO + pid // n_row_blocks
    ih0 = (pid % n_row_blocks) * R
    c_out = nc * OH * OW
    c_in = nc * H * W

    rr = ih0 + tl.arange(0, R)
    iw = tl.arange(0, WN)
    col_mask = iw < W
    valid = (rr < H)[:, None] & col_mask[None, :]

    # interior tile
    acc = tl.load(
        gop + c_out + (rr + PT)[:, None] * OW + (PL + iw)[None, :],
        mask=valid,
        other=0.0,
    )
    tl.store(gip + c_in + rr[:, None] * W + iw[None, :], acc, mask=valid)

    # boundary row fix-ups: rows 0 and H-1 also sum the pad rows
    if ih0 == 0:
        rs = tl.arange(0, PT + 1)
        row = tl.sum(
            tl.load(
                gop + c_out + rs[:, None] * OW + (PL + iw)[None, :],
                mask=col_mask[None, :],
                other=0.0,
            ).to(tl.float32),
            axis=0,
        )
        if PB > 0:
            if H == 1:
                rs2 = OH - PB + tl.arange(0, PB)
                row += tl.sum(
                    tl.load(
                        gop + c_out + rs2[:, None] * OW + (PL + iw)[None, :],
                        mask=col_mask[None, :],
                        other=0.0,
                    ).to(tl.float32),
                    axis=0,
                )
        tl.store(gip + c_in + iw, row, mask=col_mask)
    if (ih0 + R > H - 1) and (H > 1):
        ih_last = H - 1
        rs = ih_last + PT + tl.arange(0, PB + 1)
        row = tl.sum(
            tl.load(
                gop + c_out + rs[:, None] * OW + (PL + iw)[None, :],
                mask=col_mask[None, :],
                other=0.0,
            ).to(tl.float32),
            axis=0,
        )
        tl.store(gip + c_in + ih_last * W + iw, row, mask=col_mask)

    # boundary column fix-ups: main col + pad cols; rows covered by the row
    # fix-ups or the corner kernel are excluded from the overwrite
    if PT == 0:
        if PB == 0:
            col_row_mask = rr < H
        else:
            col_row_mask = rr < (H - 1)
    else:
        if PB == 0:
            col_row_mask = rr > 0
        else:
            col_row_mask = (rr > 0) & (rr < (H - 1))
    if PL > 0:
        cs = tl.arange(0, PL + 1)
        col_sum = tl.sum(
            tl.load(
                gop + c_out + (rr + PT)[:, None] * OW + cs[None, :],
                mask=(rr < H)[:, None],
                other=0.0,
            ).to(tl.float32),
            axis=1,
            keep_dims=True,
        )
        if PR > 0:
            if W == 1:
                # both edges collapse onto column 0: fold the right pad cols in
                cs2 = W + PL + tl.arange(0, PR)
                col_sum += tl.sum(
                    tl.load(
                        gop + c_out + (rr + PT)[:, None] * OW + cs2[None, :],
                        mask=(rr < H)[:, None],
                        other=0.0,
                    ).to(tl.float32),
                    axis=1,
                    keep_dims=True,
                )
        tl.store(gip + c_in + rr[:, None] * W, col_sum, mask=col_row_mask[:, None])
    if PR > 0:
        if W > 1 or PL == 0:
            cs_r = W - 1 + PL + tl.arange(0, PR + 1)
            col_sum = tl.sum(
                tl.load(
                    gop + c_out + (rr + PT)[:, None] * OW + cs_r[None, :],
                    mask=(rr < H)[:, None],
                    other=0.0,
                ).to(tl.float32),
                axis=1,
                keep_dims=True,
            )
            tl.store(
                gip + c_in + rr[:, None] * W + (W - 1),
                col_sum,
                mask=col_row_mask[:, None],
            )


@triton.jit
def _corner_kernel(
    gop,
    gip,
    OH,
    OW,
    H,
    W,
    PL: tl.constexpr,
    PR: tl.constexpr,
    PT: tl.constexpr,
    PB: tl.constexpr,
):
    # patches the four corner elements; all-scalar, kept separate from the
    # wide-vector main kernel (see the note above)
    nc = ext.program_id(0)
    c_out = nc * OH * OW
    c_in = nc * H * W

    # top-left (0, 0)
    if ((PL > 0) or (PT > 0)) or (((H == 1) and (PB > 0)) or ((W == 1) and (PR > 0))):
        v = tl.load(gop + c_out + PT * OW + PL).to(tl.float32)
        for c in range(PL):
            v += tl.load(gop + c_out + PT * OW + c)
        for r in range(PT):
            v += tl.load(gop + c_out + r * OW + PL)
            for c in range(PL):
                v += tl.load(gop + c_out + r * OW + c)
        if H == 1:
            for r in range(PB):
                v += tl.load(gop + c_out + (OH - PB + r) * OW + PL)
                for c in range(PL):
                    v += tl.load(gop + c_out + (OH - PB + r) * OW + c)
        if W == 1:
            for c in range(PR):
                v += tl.load(gop + c_out + PT * OW + (W + PL + c))
                for r in range(PT):
                    v += tl.load(gop + c_out + r * OW + (W + PL + c))
            if H == 1:
                for c in range(PR):
                    for r in range(PB):
                        v += tl.load(gop + c_out + (OH - PB + r) * OW + (W + PL + c))
        tl.store(gip + c_in, v)

    # top-right (0, W-1); folded into top-left when W == 1
    if (W > 1) and (((PR > 0) or (PT > 0)) or ((H == 1) and (PB > 0))):
        v = tl.load(gop + c_out + PT * OW + (W - 1 + PL)).to(tl.float32)
        for c in range(PR):
            v += tl.load(gop + c_out + PT * OW + (W + PL + c))
        for r in range(PT):
            v += tl.load(gop + c_out + r * OW + (W - 1 + PL))
            for c in range(PR):
                v += tl.load(gop + c_out + r * OW + (W + PL + c))
        if H == 1:
            for r in range(PB):
                v += tl.load(gop + c_out + (OH - PB + r) * OW + (W - 1 + PL))
                for c in range(PR):
                    v += tl.load(gop + c_out + (OH - PB + r) * OW + (W + PL + c))
        tl.store(gip + c_in + (W - 1), v)

    # bottom-left (H-1, 0); folded into top-left when H == 1
    if H > 1 and PB > 0:
        v = tl.load(gop + c_out + (H - 1 + PT) * OW + PL).to(tl.float32)
        for c in range(PL):
            v += tl.load(gop + c_out + (H - 1 + PT) * OW + c)
        for r in range(PB):
            v += tl.load(gop + c_out + (OH - PB + r) * OW + PL)
            for c in range(PL):
                v += tl.load(gop + c_out + (OH - PB + r) * OW + c)
        if W == 1:
            for c in range(PR):
                v += tl.load(gop + c_out + (H - 1 + PT) * OW + (W + PL + c))
                for r in range(PB):
                    v += tl.load(gop + c_out + (OH - PB + r) * OW + (W + PL + c))
        tl.store(gip + c_in + (H - 1) * W, v)

    # bottom-right (H-1, W-1); folded into the above when H == 1 or W == 1
    if ((H > 1) and (W > 1)) and ((PB > 0) and (PR > 0)):
        v = tl.load(gop + c_out + (H - 1 + PT) * OW + (W - 1 + PL)).to(tl.float32)
        for c in range(PR):
            v += tl.load(gop + c_out + (H - 1 + PT) * OW + (W + PL + c))
        for r in range(PB):
            v += tl.load(gop + c_out + (OH - PB + r) * OW + (W - 1 + PL))
            for c in range(PR):
                v += tl.load(gop + c_out + (OH - PB + r) * OW + (W + PL + c))
        tl.store(gip + c_in + (H - 1) * W + (W - 1), v)


def _has_edges(pl, pr, pt, pb):
    return pl > 0 or pr > 0 or pt > 0 or pb > 0


_plan_cache = {}


def _replication_pad2d_backward_impl(
    grad_output: torch.Tensor, self: torch.Tensor, padding, *, out: torch.Tensor = None
) -> torch.Tensor:
    if isinstance(padding, torch.Tensor):
        padding = tuple(padding.tolist())
    if isinstance(padding, int):
        pad_left = pad_right = pad_top = pad_bottom = padding
    elif isinstance(padding, (tuple, list)):
        if len(padding) != 4:
            raise ValueError(
                "padding must be a sequence of 4 integers: "
                "(pad_left, pad_right, pad_top, pad_bottom)"
            )
        pad_left, pad_right, pad_top, pad_bottom = map(int, padding)
    else:
        raise TypeError(f"Unexpected padding type: {type(padding)}")

    if pad_left < 0 or pad_right < 0 or pad_top < 0 or pad_bottom < 0:
        raise ValueError("Padding values must be non-negative")

    is_3d = self.ndim == 3
    if is_3d:
        C, H, W = self.shape
        N = 1
        grad_output_4d = grad_output.contiguous().view(1, C, *grad_output.shape[-2:])
    elif self.ndim == 4:
        N, C, H, W = self.shape
        grad_output_4d = grad_output.contiguous()
    else:
        raise ValueError("replication_pad2d_backward expects 3D or 4D input")

    OH = H + pad_top + pad_bottom
    OW = W + pad_left + pad_right

    if not _has_edges(pad_left, pad_right, pad_top, pad_bottom):
        result = grad_output_4d.to(self.dtype)
        if is_3d:
            result = result.view(C, H, W)
        if out is not None:
            out.copy_(result)
            return out
        return result

    device = self.device
    dev_idx = (
        device.index if device.index is not None else torch_device_fn.current_device()
    )
    if out is None:
        out = torch.empty_strided(
            (N, C, H, W),
            (C * H * W, H * W, W, 1),
            device=device,
            dtype=self.dtype,
        )

    itemsize = grad_output_4d.element_size()
    # tiling plan depends only on (H, W, itemsize, padding, dtypes)
    _plan_key = (
        H,
        W,
        itemsize,
        (pad_left, pad_right, pad_top, pad_bottom),
        grad_output_4d.dtype,
        out.dtype,
    )
    plan = _plan_cache.get(_plan_key)
    if plan is None:
        w_next_pow2 = triton.next_power_of_2(W)
        rows_per_block = max(1, min(H, TILE_BYTES // (W * itemsize)))
        rows_per_block = min(rows_per_block, max(1, TILE_FP32_CAP // (w_next_pow2 * 4)))
        plan = (w_next_pow2, rows_per_block)
        _plan_cache[_plan_key] = plan
    w_next_pow2, rows_per_block = plan
    grid = (N * C * triton.cdiv(H, rows_per_block),)
    if torch_device_fn.current_device() != dev_idx:
        ctx = torch_device_fn.device(dev_idx)
    else:
        ctx = nullcontext()  # skip the guard on the single-device path
    with ctx:
        _backward_gather_kernel[grid](
            grad_output_4d,
            out,
            0,
            OH,
            OW,
            H,
            W,
            pad_left,
            pad_right,
            pad_top,
            pad_bottom,
            rows_per_block,
            w_next_pow2,
        )
        if (pad_top > 0 or pad_bottom > 0) and (pad_left > 0 or pad_right > 0):
            _corner_kernel[(N * C,)](
                grad_output_4d,
                out,
                OH,
                OW,
                H,
                W,
                pad_left,
                pad_right,
                pad_top,
                pad_bottom,
            )

    if is_3d:
        out = out.view(C, H, W)
    return out


def replication_pad2d_backward(grad_output, self, padding):
    logger.debug("GEMS_ASCEND REPLICATION_PAD2D_BACKWARD")
    return _replication_pad2d_backward_impl(grad_output, self, padding, out=None)


def replication_pad2d_backward_grad_input(grad_output, self, padding, *, grad_input):
    logger.debug("GEMS_ASCEND REPLICATION_PAD2D_BACKWARD_GRAD_INPUT")
    return _replication_pad2d_backward_impl(grad_output, self, padding, out=grad_input)
