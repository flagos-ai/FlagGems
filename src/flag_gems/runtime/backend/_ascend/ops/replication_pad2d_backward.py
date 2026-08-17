import logging
import os
from contextlib import nullcontext

import torch
import triton
import triton.language as tl
from torch_npu._C import _npu_getCurrentRawStream

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
        row = tl.load(gop + c_out + PT * OW + (PL + iw), mask=col_mask, other=0.0).to(
            tl.float32
        )
        for r in range(PT):
            row += tl.load(gop + c_out + r * OW + (PL + iw), mask=col_mask, other=0.0)
        if H == 1:
            for r in range(PB):
                row += tl.load(
                    gop + c_out + (OH - PB + r) * OW + (PL + iw),
                    mask=col_mask,
                    other=0.0,
                )
        tl.store(gip + c_in + iw, row, mask=col_mask)
    if (ih0 + R > H - 1) and (H > 1):
        ih_last = H - 1
        row = tl.load(
            gop + c_out + (ih_last + PT) * OW + (PL + iw), mask=col_mask, other=0.0
        ).to(tl.float32)
        for r in range(PB):
            row += tl.load(
                gop + c_out + (OH - PB + r) * OW + (PL + iw),
                mask=col_mask,
                other=0.0,
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
        col_sum = tl.load(
            gop + c_out + (rr + PT)[:, None] * OW + PL,
            mask=(rr < H)[:, None],
            other=0.0,
        ).to(tl.float32)
        for c in range(PL):
            col_sum += tl.load(
                gop + c_out + (rr + PT)[:, None] * OW + c,
                mask=(rr < H)[:, None],
                other=0.0,
            ).to(tl.float32)
        if W == 1:
            # both edges collapse onto column 0: fold the right pad cols in
            for c in range(PR):
                col_sum += tl.load(
                    gop + c_out + (rr + PT)[:, None] * OW + (W + PL + c),
                    mask=(rr < H)[:, None],
                    other=0.0,
                ).to(tl.float32)
        tl.store(gip + c_in + rr[:, None] * W, col_sum, mask=col_row_mask[:, None])
    if PR > 0:
        if W > 1:
            col_sum = tl.load(
                gop + c_out + (rr + PT)[:, None] * OW + (W - 1 + PL),
                mask=(rr < H)[:, None],
                other=0.0,
            ).to(tl.float32)
            for c in range(PR):
                col_sum += tl.load(
                    gop + c_out + (rr + PT)[:, None] * OW + (W + PL + c),
                    mask=(rr < H)[:, None],
                    other=0.0,
                ).to(tl.float32)
            tl.store(
                gip + c_in + rr[:, None] * W + (W - 1),
                col_sum,
                mask=col_row_mask[:, None],
            )
        elif PL == 0:
            # W == 1 with no left pad: column 0 is the right edge
            col_sum = tl.load(
                gop + c_out + (rr + PT)[:, None] * OW + (W - 1 + PL),
                mask=(rr < H)[:, None],
                other=0.0,
            ).to(tl.float32)
            for c in range(PR):
                col_sum += tl.load(
                    gop + c_out + (rr + PT)[:, None] * OW + (W + PL + c),
                    mask=(rr < H)[:, None],
                    other=0.0,
                ).to(tl.float32)
            tl.store(gip + c_in + rr[:, None] * W, col_sum, mask=col_row_mask[:, None])


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


_fast_launch_cache = {}
_plan_cache = {}
_BINDER_MEMO_SIZE = 4
_binder_memo = {}


def _arg_sig(runtime_args):
    sig = []
    for a in runtime_args:
        if isinstance(a, torch.Tensor):
            sig.append((a.dtype, a.data_ptr()))
        else:
            sig.append((type(a).__name__, a))
    return tuple(sig)


def _fast_launch(jit_fn, grid, constexpr_args, runtime_args, device=None):
    if device is None:
        device = torch_device_fn.current_device()
    all_args = runtime_args + tuple(constexpr_args)
    # The fast path relies on triton-fork internals (binder, compiled-kernel
    # cache); fall back to the plain python launch where they differ.
    if not hasattr(jit_fn, "binder"):
        jit_fn[grid](*all_args)
        return
    try:
        debug_val = os.environ.get("TRITON_DEBUG", "0") == "1"
        if jit_fn.binder is None:
            jit_fn[grid](*all_args)
        memo_key = (debug_val, constexpr_args, _arg_sig(runtime_args))
        slot = _binder_memo.get(id(jit_fn))
        if slot is None:
            slot = {}
            _binder_memo[id(jit_fn)] = slot
        jit_key = slot.get(memo_key)
        if jit_key is None:
            (
                _,
                sig_and_spec,
                constexpr_vals,
                _,
                excess_kwargs,
            ) = jit_fn.binder(*all_args, debug=debug_val)
            jit_key = "".join(sig_and_spec) + str((constexpr_vals, excess_kwargs))
            if len(slot) >= _BINDER_MEMO_SIZE:
                slot.pop(next(iter(slot)))  # evict the oldest entry
            slot[memo_key] = jit_key
        entry = _fast_launch_cache.get(jit_key)
        if entry is None:
            # one normal launch populates the jit's compiled-kernel cache
            jit_fn[grid](*all_args)
            compiled = jit_fn.cache[device][jit_key]
            run = compiled.run  # triggers _init_handles: loads the binary once
            entry = (run, compiled.function, compiled.packed_metadata)
            _fast_launch_cache[jit_key] = entry
        run, fn, pm = entry
        stream = _npu_getCurrentRawStream(device)
        run(grid[0], 1, 1, stream, fn, pm, None, None, None, *runtime_args)
    except Exception:
        jit_fn[grid](*all_args)


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
        _fast_launch(
            _backward_gather_kernel,
            grid,
            (pad_left, pad_right, pad_top, pad_bottom, rows_per_block, w_next_pow2),
            (grad_output_4d, out, 0, OH, OW, H, W),
            device=dev_idx,
        )
        if (pad_top > 0 or pad_bottom > 0) and (pad_left > 0 or pad_right > 0):
            _fast_launch(
                _corner_kernel,
                (N * C,),
                (pad_left, pad_right, pad_top, pad_bottom),
                (grad_output_4d, out, OH, OW, H, W),
                device=dev_idx,
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
