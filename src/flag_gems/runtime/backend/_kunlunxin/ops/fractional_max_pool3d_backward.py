import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _fractional_max_pool3d_backward_scatter_kernel(
    grad_output_ptr,
    indices_ptr,
    grad_input_ptr,
    n_out,
    OUT_PER_NC: tl.constexpr,
    IN_PLANE: tl.constexpr,
    EVEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Non-atomic scatter fast path (Kunlunxin/XPU3).

    One lane per output position.  Valid whenever ``alpha >= k`` in every dim
    (``in - k >= k * (out - 1)``, or ``out == 1``): the ``k``-wide windows are
    pairwise disjoint, so every input position belongs to at most one window,
    each output's argmax index is distinct, and plain stores can never race --
    exact and deterministic, O(n_out).

    The reference forward stores ``indices`` as the **plane-local** flat offset
    ``d*IN_H*IN_W + h*IN_W + w`` (int64), so the destination input element is
    ``nc * IN_PLANE + idx``.  This mirrors the proven 2d
    ``_fractional_max_pool2d_backward_scatter_kernel``; ``tl.atomic_add``
    scatter is banned on this backend (it loses updates, seed-dependent).
    """
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        idx = tl.load(indices_ptr + offsets).to(tl.int32)
        val = tl.load(grad_output_ptr + offsets).to(tl.float32)
        nc = offsets // OUT_PER_NC
        tl.store(
            grad_input_ptr + nc * IN_PLANE + idx,
            val.to(grad_input_ptr.dtype.element_ty),
        )
    else:
        m = offsets < n_out
        idx = tl.load(indices_ptr + offsets, mask=m, other=0).to(tl.int32)
        val = tl.load(grad_output_ptr + offsets, mask=m, other=0.0).to(tl.float32)
        nc = offsets // OUT_PER_NC
        tl.store(
            grad_input_ptr + nc * IN_PLANE + idx,
            val.to(grad_input_ptr.dtype.element_ty),
            mask=m,
        )


@libentry()
@triton.jit
def _fractional_max_pool3d_backward_gather_kernel(
    grad_output_ptr,
    indices_ptr,
    grad_input_ptr,
    n_elems,
    in_d,
    in_h,
    in_w,
    out_d,
    out_h,
    out_w,
    inv_alpha_d,
    inv_alpha_h,
    inv_alpha_w,
    k_d: tl.constexpr,
    k_h: tl.constexpr,
    k_w: tl.constexpr,
    MAX_D: tl.constexpr,
    MAX_H: tl.constexpr,
    MAX_W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Gather-based fractional max pool 3d backward (Kunlunxin/XPU3).

    General fallback (used when ``alpha < k`` in some dim, i.e. overlapping
    windows where the scatter fast path is not race-free).  One lane per input
    position (flat 1-D over n, c, d, h, w).  For each input position the
    candidate outputs whose window can contain it are scanned (bounded box from
    ``inv_alpha`` with a +-2 rounding margin, exactly as the proven 2d gather
    kernel) and their gradients are accumulated wherever the stored argmax
    equals this lane's plane-local flat position.  Deterministic, race-free.
    The scan ranges are rolled (``range``, not ``tl.static_range``): the XPU
    unroll-control pass fails when the whole per-program body is unrolled.
    """
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elems
    safe = tl.where(mask, offsets, 0)

    in_plane = in_d * in_h * in_w
    in_hw = in_h * in_w
    nc = safe // in_plane
    rem = safe % in_plane
    d = rem // in_hw
    rem2 = rem % in_hw
    h = rem2 // in_w
    w = rem2 % in_w
    my_flat = rem  # plane-local flat index (d*IN_H*IN_W + h*IN_W + w)

    od_lo = tl.maximum(
        0, (tl.maximum(d - k_d, 0).to(tl.float32) * inv_alpha_d).to(tl.int32) - 2
    )
    od_hi = tl.minimum(
        out_d - 1, ((d + 1).to(tl.float32) * inv_alpha_d).to(tl.int32) + 2
    )
    oh_lo = tl.maximum(
        0, (tl.maximum(h - k_h, 0).to(tl.float32) * inv_alpha_h).to(tl.int32) - 2
    )
    oh_hi = tl.minimum(
        out_h - 1, ((h + 1).to(tl.float32) * inv_alpha_h).to(tl.int32) + 2
    )
    ow_lo = tl.maximum(
        0, (tl.maximum(w - k_w, 0).to(tl.float32) * inv_alpha_w).to(tl.int32) - 2
    )
    ow_hi = tl.minimum(
        out_w - 1, ((w + 1).to(tl.float32) * inv_alpha_w).to(tl.int32) + 2
    )

    out_per_nc = out_d * out_h * out_w
    gop = grad_output_ptr + nc * out_per_nc
    iop = indices_ptr + nc * out_per_nc

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for a_d in range(0, MAX_D):
        o_d = od_lo + a_d
        d_ok = (o_d <= od_hi) & (o_d < out_d)
        c_d = tl.minimum(o_d, out_d - 1)
        for a_h in range(0, MAX_H):
            o_h = oh_lo + a_h
            h_ok = (o_h <= oh_hi) & (o_h < out_h)
            c_h = tl.minimum(o_h, out_h - 1)
            for a_w in range(0, MAX_W):
                o_w = ow_lo + a_w
                w_ok = (o_w <= ow_hi) & (o_w < out_w)
                c_w = tl.minimum(o_w, out_w - 1)
                o_off = (c_d * out_h + c_h) * out_w + c_w
                idx = tl.load(iop + o_off).to(tl.int32)
                val = tl.load(gop + o_off).to(tl.float32)
                active = mask & d_ok & h_ok & w_ok
                acc += tl.where(active & (idx == my_flat), val, 0.0)

    tl.store(
        grad_input_ptr + offsets,
        acc.to(grad_input_ptr.dtype.element_ty),
        mask=mask,
    )


def _parse_size(value):
    if isinstance(value, (int, float)):
        return value, value, value
    return value[0], value[1], value[2]


def _pick_block(numel):
    # Widest tile that divides ``numel`` -> unmasked EVEN path (masked stores
    # block block-DMA lowering on XPU3).  A 2048-lane tile only pays off once
    # there is enough work to amortize it (measured: 200704 elems 331->217 us,
    # but 8192 elems 10.2->11.9 us), so gate it on ``numel``.
    ladder = (2048, 1024, 512, 256, 128) if numel >= 65536 else (1024, 512, 256, 128)
    for cand in ladder:
        if numel % cand == 0:
            return cand, True
    return 256, False


def fractional_max_pool3d_backward(
    grad_output,
    input,
    kernel_size,
    output_size,
    indices,
):
    logger.debug("GEMS_KUNLUNXIN FRACTIONAL_MAX_POOL3D_BACKWARD")
    input = input.contiguous()
    grad_output = grad_output.contiguous()
    indices = indices.contiguous()
    n, c, in_d, in_h, in_w = input.shape
    k_d, k_h, k_w = _parse_size(kernel_size)
    out_d, out_h, out_w = _parse_size(output_size)
    if input.numel() == 0 or grad_output.numel() == 0:
        return torch.zeros_like(input)

    in_plane = in_d * in_h * in_w
    out_per_nc = out_d * out_h * out_w
    n_out = grad_output.numel()

    scatter = (
        (out_d == 1 or in_d - k_d >= k_d * (out_d - 1))
        and (out_h == 1 or in_h - k_h >= k_h * (out_h - 1))
        and (out_w == 1 or in_w - k_w >= k_w * (out_w - 1))
    )

    with torch_device_fn.device(input.device):
        if scatter:
            grad_input = torch.zeros_like(input)
            block, even = _pick_block(n_out)
            _fractional_max_pool3d_backward_scatter_kernel[
                (triton.cdiv(n_out, block),)
            ](
                grad_output,
                indices,
                grad_input,
                n_out,
                out_per_nc,
                in_plane,
                EVEN=even,
                BLOCK=block,
                num_warps=max(2, block // 256),
                isCloseVectorization=True,
                buffer_size_limit=2048,
            )
        else:
            inv_alpha_d = (out_d - 1) / (in_d - k_d) if in_d > k_d else 0.0
            inv_alpha_h = (out_h - 1) / (in_h - k_h) if in_h > k_h else 0.0
            inv_alpha_w = (out_w - 1) / (in_w - k_w) if in_w > k_w else 0.0
            max_d = max(
                1, min(out_d, int(math.ceil((k_d + 1) * inv_alpha_d)) + 6)
            )
            max_h = max(
                1, min(out_h, int(math.ceil((k_h + 1) * inv_alpha_h)) + 6)
            )
            max_w = max(
                1, min(out_w, int(math.ceil((k_w + 1) * inv_alpha_w)) + 6)
            )
            n_elems = input.numel()
            grad_input = torch.empty_like(input)
            _fractional_max_pool3d_backward_gather_kernel[
                (triton.cdiv(n_elems, 128),)
            ](
                grad_output,
                indices,
                grad_input,
                n_elems,
                in_d,
                in_h,
                in_w,
                out_d,
                out_h,
                out_w,
                inv_alpha_d,
                inv_alpha_h,
                inv_alpha_w,
                k_d,
                k_h,
                k_w,
                MAX_D=max_d,
                MAX_H=max_h,
                MAX_W=max_w,
                BLOCK=128,
                num_warps=2,
                isCloseVectorization=True,
                buffer_size_limit=2048,
            )
    return grad_input


# Dedicated tests/benchmarks may call ``flag_gems.ops.<op>`` directly, which
# bypasses ``SpecOpRegistrar`` (it only rewrites the top-level ``flag_gems``
# namespace).  Publish the backend impl there too
# (wiki/patterns/xpu3-dedicated-tests-call-flag-gems-ops.md).
import sys as _sys

_generic_ops_module = _sys.modules.get("flag_gems.ops")
if _generic_ops_module is not None:
    setattr(
        _generic_ops_module,
        "fractional_max_pool3d_backward",
        fractional_max_pool3d_backward,
    )
