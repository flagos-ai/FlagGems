import logging
import math

import torch
import triton
import triton.experimental.tle as tle
import triton.language as tl

from flag_gems.runtime import torch_device_fn

from ..utils.tle_copy import (
    _DSA_BUF_DTYPE,
    _DSA_MOVE_DTYPE,
    _tle_dsa_row_copy_kernel,
    tle_copy,
    tle_dma_available,
)

logger = logging.getLogger(__name__)

# Grid used for the hand-written reflected-edge payload. Saturates the DMA
# engines from ~32 clusters onward on XPU3 for the giant-shape edge work.
_EDGE_GRID = 32


def _tle_interior_copy(x: torch.Tensor, out: torch.Tensor, pad_left: int, W_in: int):
    """Bulk-copy the interior in[b, :] -> out[b, pad_left : pad_left+W_in) on the
    tle.dsa SDNN path, with a tile tuned for this contiguous-src/strided-row-dst
    shape.

    `tle_copy`'s generic `_dsa_tile` picks the widest row that fits the on-chip
    budget (2-byte dtypes -> 2048-col x 8-row). On this giant reflection-pad
    shape a narrower row with more rows per tile (1024-col x 16-row) moves the
    same bytes ~15% faster (fp16/bf16 31.5us -> 27.7us on [32,64,2048]); 4-byte
    already sits at its optimum (1024-col x 8-row), so only the tile shape moves.
    Returns True on success, False to let the caller keep its own fallback.
    """
    if not tle_dma_available():
        return False
    es = x.element_size()
    move = _DSA_MOVE_DTYPE.get(es)
    if move is None:
        return False
    W_out = out.shape[-1]
    B = int(math.prod(x.shape[:-1])) if x.dim() > 1 else 1
    try:
        src_v = x.reshape(B, W_in).view(move)
        dst_v = out.view(B, W_out).view(move).narrow(1, pad_left, W_in)
    except RuntimeError:
        return False
    styp = _DSA_BUF_DTYPE[move]
    rows_block, cols_block = (16, 1024) if es == 2 else (8, 1024)
    cols_block = min(cols_block, triton.next_power_of_2(W_in))
    row_blocks = triton.cdiv(B, rows_block)
    grid = (row_blocks, triton.cdiv(W_in, cols_block))
    _tle_dsa_row_copy_kernel[grid](
        src_v,
        dst_v,
        B,
        W_in,
        src_v.stride(0),
        dst_v.stride(0),
        1,
        row_blocks,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        rows_block,
        cols_block,
        False,
        styp,
        styp,
        False,
        False,
        is_sdnn=True,
        num_stages=2,
    )
    return True


@tle.raw.dialect("xpu3", file="pad_edges3.xpu")
def pad_edges3(out, inp, B, W_in, W_out, pad_left, pad_right, es, pid, npid): ...


@triton.jit(
    do_not_specialize=[
        "B",
        "W_in",
        "W_out",
        "pad_left",
        "pad_right",
        "es",
        "npid",
    ]
)
def pad1d_raw_edges_kernel(
    out_i8, in_i8, B, W_in, W_out, pad_left, pad_right, es, npid
):
    pid = tl.program_id(axis=0)
    tle.raw.call(
        pad_edges3,
        (out_i8, in_i8, B, W_in, W_out, pad_left, pad_right, es, pid, npid),
    )


@triton.jit
def reflection_pad1d_kernel(
    in_ptr,
    out_ptr,
    W_in,
    pad_left,
    W_out,
    total_out,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr = True,
):
    pid = tl.program_id(axis=0)
    o = pid * BLOCK + tl.arange(0, BLOCK)

    b = o // W_out
    w_idx = o - b * W_out

    x = w_idx.to(tl.int32) - pad_left
    pW = 2 * (W_in - 1)
    t = tl.abs(x)
    iw = tl.where(t < W_in, t, pW - t)

    in_offs = b * W_in + iw
    if NEED_MASK:
        mask = o < total_out
        vals = tl.load(in_ptr + in_offs, mask=mask)
        tl.store(out_ptr + o, vals, mask=mask)
    else:
        vals = tl.load(in_ptr + in_offs)
        tl.store(out_ptr + o, vals)


@triton.jit
def copy_tensor_kernel(in_ptr, out_ptr, total, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    mask = o < total
    vals = tl.load(in_ptr + o, mask=mask)
    tl.store(out_ptr + o, vals, mask=mask)


@triton.jit
def pad1d_side_kernel(
    in_ptr,
    out_ptr,
    W_in,
    pad_left,
    W_out,
    total_side,
    PAD: tl.constexpr,
    SIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    idx_c = tl.minimum(idx, total_side - 1)
    m = idx < total_side
    b = idx_c // PAD
    j = idx_c - b * PAD
    if SIDE == 0:
        src = pad_left - j
        dst = b * W_out + j
    else:
        src = W_in - 2 - j
        dst = b * W_out + W_in + pad_left + j
    v = tl.load(in_ptr + b * W_in + src)
    tl.store(out_ptr + dst, v, mask=m)


@triton.jit
def pad1d_both_sides_kernel(
    in_ptr,
    out_ptr,
    W_in,
    pad_left,
    pad_right,
    W_out,
    pad_sum,
    total_pad,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    m = idx < total_pad
    idx_c = tl.minimum(idx, total_pad - 1)
    b = idx_c // pad_sum
    r = idx_c - b * pad_sum
    is_left = r < pad_left
    j_right = r - pad_left
    src = tl.where(is_left, pad_left - r, W_in - 2 - j_right)
    dst = tl.where(
        is_left,
        b * W_out + r,
        b * W_out + W_in + pad_left + j_right,
    )
    v = tl.load(in_ptr + b * W_in + src)
    tl.store(out_ptr + dst, v, mask=m)


def _launch_reflection_pad1d(input: torch.Tensor, padding, out: torch.Tensor = None):
    if not isinstance(padding, (list, tuple)) or len(padding) != 2:
        raise ValueError(
            "padding must be a sequence of length 2: (pad_left, pad_right)"
        )
    pad_left, pad_right = int(padding[0]), int(padding[1])
    if pad_left < 0 or pad_right < 0:
        raise ValueError("padding values must be >= 0")
    if input.dim() < 1:
        raise ValueError("input must have at least 1 dimension")

    x = input.contiguous()
    W_in = int(x.shape[-1])
    if W_in <= 0:
        raise ValueError("last dimension (width) must be > 0")

    W_out = W_in + pad_left + pad_right
    leading_shape = x.shape[:-1]
    B = int(math.prod(leading_shape)) if len(leading_shape) > 0 else 1

    if out is None:
        out = torch.empty((*leading_shape, W_out), device=x.device, dtype=x.dtype)
    else:
        expected_shape = (*leading_shape, W_out)
        if tuple(out.shape) != expected_shape:
            raise ValueError(
                f"out tensor has shape {tuple(out.shape)}, expected {expected_shape}"
            )
        if out.dtype != x.dtype:
            raise ValueError(
                f"out dtype {out.dtype} does not match input dtype {x.dtype}"
            )
        if out.device != x.device:
            raise ValueError("out must be on the same device as input")
        out = out.contiguous()

    BLOCK = 1024

    if pad_left == 0 and pad_right == 0:
        total = B * W_in
        grid = (triton.cdiv(total, BLOCK),)
        with torch_device_fn.device(x.device):
            copy_tensor_kernel[grid](x, out, total, BLOCK=BLOCK)
        return out

    if W_in < 2:
        raise ValueError(
            "input width must be at least 2 for reflection padding when padding > 0"
        )
    if pad_left >= W_in or pad_right >= W_in:
        raise ValueError(
            "padding values must be less than the input width for reflection padding"
        )

    total_out = B * W_out
    if total_out >= 262144:
        with torch_device_fn.device(x.device):
            # Interior bulk copy via on-chip DMA (tle.dsa). No ATen fallback.
            # `mid` is a pure strided view (no data movement, no kernel launch).
            mid = out.narrow(-1, pad_left, W_in)
            if _tle_interior_copy(x, out, pad_left, W_in) or tle_copy(x, mid):
                pad_sum = pad_left + pad_right
                if pad_sum > 0:
                    es = x.element_size()
                    npid = min(_EDGE_GRID, B) if B > 0 else 1
                    in_i8 = x.view(torch.int8)
                    out_i8 = out.view(torch.int8)
                    pad1d_raw_edges_kernel[(npid,)](
                        out_i8,
                        in_i8,
                        B,
                        W_in,
                        W_out,
                        pad_left,
                        pad_right,
                        es,
                        npid,
                    )
                return out

    # Tiny shapes are launch-bound. On XPU3 the 2-byte (fp16/bf16) load/store
    # path carries a fixed per-kernel penalty at BLOCK=256 that the 4-byte
    # (fp32) path does not; widening to BLOCK=512 lets the codegen emit wider
    # aligned vector ops and erases it (measured fp16/bf16 (3,33)/(2,4,64):
    # ~0.77-0.91x -> ~0.99-1.12x, fp32 unchanged ~1.07x). Mid shapes keep 1024.
    BLOCK = 512 if total_out <= 1024 else 1024
    need_mask = (total_out % BLOCK) != 0
    grid = (triton.cdiv(total_out, BLOCK),)
    with torch_device_fn.device(x.device):
        reflection_pad1d_kernel[grid](
            x,
            out,
            W_in,
            pad_left,
            W_out,
            total_out,
            BLOCK=BLOCK,
            NEED_MASK=need_mask,
        )
    return out


def reflection_pad1d(input: torch.Tensor, padding):
    logger.debug("GEMS_KUNLUNXIN REFLECTION_PAD1D")
    return _launch_reflection_pad1d(input, padding, out=None)


def reflection_pad1d_out(input: torch.Tensor, padding, out: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN REFLECTION_PAD1D_OUT")
    return _launch_reflection_pad1d(input, padding, out=out)
