import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _fractional_max_pool3d_lattice_kernel(
    input_ptr,
    output_ptr,
    indices_ptr,
    numel,
    # Constexpr geometry: the flat div/mod decode becomes shifts/masks and the
    # tap addresses become affine in the lane index.
    IN_PLANE: tl.constexpr,
    IN_H: tl.constexpr,
    IN_W: tl.constexpr,
    OUT_D: tl.constexpr,
    OUT_H: tl.constexpr,
    OUT_W: tl.constexpr,
    # Integer window lattice: start(o) = A * o + OFF (see host dispatch).
    A_D: tl.constexpr,
    A_H: tl.constexpr,
    A_W: tl.constexpr,
    OFF_D: tl.constexpr,
    OFF_H: tl.constexpr,
    OFF_W: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    EVEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Integer-alpha fast path: one flat tile of output positions.

    When ``(in - k) % (out - 1) == 0`` in every dim, ``alpha`` is an exact
    integer ``A`` and the reference start formula collapses to
    ``start(o) = A * o`` for *every* sample value:
    ``trunc((o + s) * A) - trunc(s * A) == o * A`` because ``o * A`` is an
    integer and ``s * A >= 0``.  The last output's clamp ``in - k`` equals
    ``A * (out - 1)`` as well.  The sampled windows are then a fixed lattice,
    so no per-lane random-sample math is needed and every tap access is affine
    in the output position -- which is what lets the backend use wide DMA tiles
    instead of per-lane scalar gathers (~10x over the generic kernel).

    ``OFF_*`` covers ``out == 1``: there ``alpha == 0`` while the clamped start
    is ``in - k`` (the only output is the last one).
    """
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        safe = offs
    else:
        m = offs < numel
        safe = tl.where(m, offs, 0)

    o_w = safe % OUT_W
    t = safe // OUT_W
    o_h = t % OUT_H
    t = t // OUT_H
    o_d = t % OUT_D
    nc = t // OUT_D

    # Contiguous input: plane base is nc * IN_D * IN_H * IN_W.
    plane = nc * IN_PLANE
    d_start = o_d * A_D + OFF_D
    h_start = o_h * A_H + OFF_H
    w_start = o_w * A_W + OFF_W

    dtype = input_ptr.type.element_ty
    max_val = tl.full((BLOCK,), -float("inf"), dtype)
    max_idx = tl.zeros((BLOCK,), dtype=tl.int32)

    for kd in tl.static_range(KD):
        d_off = (d_start + kd) * (IN_H * IN_W)
        for kh in tl.static_range(KH):
            dh = d_off + (h_start + kh) * IN_W
            for kw in tl.static_range(KW):
                if EVEN:
                    val = tl.load(input_ptr + plane + dh + w_start + kw)
                else:
                    val = tl.load(
                        input_ptr + plane + dh + w_start + kw,
                        mask=m,
                        other=-float("inf"),
                    )
                flat = dh + w_start + kw
                # first-maximum in (kd, kh, kw) order, like the reference scan
                upd = val > max_val
                max_val = tl.where(upd, val, max_val)
                max_idx = tl.where(upd, flat, max_idx)

    if EVEN:
        tl.store(output_ptr + offs, max_val)
        tl.store(indices_ptr + offs, max_idx.to(tl.int64))
    else:
        tl.store(output_ptr + offs, max_val, mask=m)
        tl.store(indices_ptr + offs, max_idx.to(tl.int64), mask=m)


@libentry()
@triton.jit
def _fractional_max_pool3d_forward_kernel(
    input_ptr,
    output_ptr,
    indices_ptr,
    random_samples_ptr,
    numel,
    in_d: tl.constexpr,
    in_h: tl.constexpr,
    in_w: tl.constexpr,
    out_d: tl.constexpr,
    out_h: tl.constexpr,
    out_w: tl.constexpr,
    kernel_d: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    alpha_d,
    alpha_h,
    alpha_w,
    STATIC_TAPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """General path (non-integer alpha): sample-dependent window starts.

    Formula (same as ATen, evaluated in fp32 -- the fp64 correctness reference
    keeps ``_random_samples`` unrounded, so fp32 reproduces its indices on the
    whole matrix; see ``harness/perf_ir/probe_fmp3d_formulas.py``)::

        alpha        = (in - k) / (out - 1)          # 0 when out == 1
        sample_index = trunc(sample * alpha)
        start(o)     = trunc((o + sample) * alpha) - sample_index
        start(out - 1) = in - k                      # clamped
    """
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    safe = tl.where(mask, offs, 0)

    # Output is contiguous: the same flat offset addresses out and indices.
    o_w = safe % out_w
    tmp = safe // out_w
    o_h = tmp % out_h
    tmp = tmp // out_h
    o_d = tmp % out_d
    nc = tmp // out_d

    dtype = input_ptr.type.element_ty
    f32 = tl.float32

    s_d = tl.load(random_samples_ptr + nc * 3).to(f32)
    s_h = tl.load(random_samples_ptr + nc * 3 + 1).to(f32)
    s_w = tl.load(random_samples_ptr + nc * 3 + 2).to(f32)

    base_d = (s_d * alpha_d).to(tl.int32)
    base_h = (s_h * alpha_h).to(tl.int32)
    base_w = (s_w * alpha_w).to(tl.int32)

    st_d = ((o_d.to(f32) + s_d) * alpha_d).to(tl.int32) - base_d
    st_h = ((o_h.to(f32) + s_h) * alpha_h).to(tl.int32) - base_h
    st_w = ((o_w.to(f32) + s_w) * alpha_w).to(tl.int32) - base_w
    st_d = tl.where(o_d == out_d - 1, in_d - kernel_d, st_d)
    st_h = tl.where(o_h == out_h - 1, in_h - kernel_h, st_h)
    st_w = tl.where(o_w == out_w - 1, in_w - kernel_w, st_w)

    in_hw: tl.constexpr = in_h * in_w
    plane = nc * (in_d * in_hw)

    max_val = tl.full((BLOCK,), -float("inf"), f32)
    max_idx = tl.zeros((BLOCK,), dtype=tl.int32)

    if STATIC_TAPS:
        for kd in tl.static_range(kernel_d):
            d_off = (st_d + kd) * in_hw
            for kh in tl.static_range(kernel_h):
                dh = d_off + (st_h + kh) * in_w
                for kw in tl.static_range(kernel_w):
                    flat = dh + st_w + kw
                    val = tl.load(input_ptr + plane + flat).to(f32)
                    upd = val > max_val
                    max_val = tl.where(upd, val, max_val)
                    max_idx = tl.where(upd, flat, max_idx)
    else:
        # Runtime-loop fallback for large windows: must stay at <=128 lanes
        # (wiki/patterns/xpu3-loop-wide-tile-unrollcontrol-crash).
        for kd in range(kernel_d):
            d_off = (st_d + kd) * in_hw
            for kh in range(kernel_h):
                dh = d_off + (st_h + kh) * in_w
                for kw in range(kernel_w):
                    flat = dh + st_w + kw
                    val = tl.load(input_ptr + plane + flat).to(f32)
                    upd = val > max_val
                    max_val = tl.where(upd, val, max_val)
                    max_idx = tl.where(upd, flat, max_idx)

    tl.store(output_ptr + offs, max_val.to(dtype), mask=mask)
    tl.store(indices_ptr + offs, max_idx.to(tl.int64), mask=mask)


def _parse_size(value):
    if isinstance(value, (int, float)):
        return value, value, value
    return value[0], value[1], value[2]


def _lattice_stride(in_size, out_size, k):
    """Return ``(A, OFF)`` for the integer-alpha lattice, or ``None``.

    ``start(o) = A * o + OFF``; ``OFF`` is non-zero only for ``out == 1``
    (``alpha == 0`` but the clamped start is ``in - k``).
    """
    if out_size == 1:
        return 0, in_size - k
    num, den = in_size - k, out_size - 1
    if num % den != 0:
        return None
    return num // den, 0


def fractional_max_pool3d(
    input,
    kernel_size,
    output_size=None,
    output_ratio=None,
    return_indices: bool = True,
    _random_samples=None,
):
    logger.debug("GEMS_KUNLUNXIN FRACTIONAL_MAX_POOL3D")
    if isinstance(output_ratio, torch.Tensor) and _random_samples is None:
        _random_samples = output_ratio
        output_ratio = None
    assert input.dim() == 5, f"Expected 5D input, got {input.dim()}D"
    input = input.contiguous()
    batch_size, channels, in_d, in_h, in_w = input.shape
    kernel_d, kernel_h, kernel_w = _parse_size(kernel_size)
    if output_size is not None:
        out_d, out_h, out_w = _parse_size(output_size)
    elif output_ratio is not None:
        ratio_d, ratio_h, ratio_w = _parse_size(output_ratio)
        out_d = int(in_d * ratio_d)
        out_h = int(in_h * ratio_h)
        out_w = int(in_w * ratio_w)
    else:
        raise ValueError("Either output_size or output_ratio must be specified")

    assert out_d + kernel_d - 1 <= in_d
    assert out_h + kernel_h - 1 <= in_h
    assert out_w + kernel_w - 1 <= in_w

    if _random_samples is None:
        # Reference keeps the samples unrounded (fp64 upcast), so draw fp32.
        _random_samples = torch.rand(
            batch_size, channels, 3, device=input.device, dtype=torch.float32
        )
    else:
        assert _random_samples.shape == (batch_size, channels, 3), (
            f"_random_samples must have shape ({batch_size}, {channels}, 3), "
            f"got {_random_samples.shape}"
        )
        _random_samples = _random_samples.contiguous()
    random_samples = _random_samples.reshape(batch_size * channels, 3)

    output = torch.empty(
        (batch_size, channels, out_d, out_h, out_w),
        device=input.device,
        dtype=input.dtype,
    )
    indices = torch.empty(
        (batch_size, channels, out_d, out_h, out_w),
        device=input.device,
        dtype=torch.int64,
    )
    numel = output.numel()
    if numel == 0:
        return (output, indices) if return_indices else output

    strides = (
        _lattice_stride(in_d, out_d, kernel_d),
        _lattice_stride(in_h, out_h, kernel_h),
        _lattice_stride(in_w, out_w, kernel_w),
    )
    taps = kernel_d * kernel_h * kernel_w
    with torch_device_fn.device(input.device):
        if all(s is not None for s in strides):
            (a_d, off_d), (a_h, off_h), (a_w, off_w) = strides
            # Widest tile that divides numel: the unmasked path is the fast one
            # (masked loads/stores block block-DMA lowering on XPU3).
            block = 0
            for cand in (1024, 512, 256):
                if numel % cand == 0:
                    block = cand
                    break
            even = block != 0
            if not even:
                block = 256
            grid = (triton.cdiv(numel, block),)
            _fractional_max_pool3d_lattice_kernel[grid](
                input,
                output,
                indices,
                numel,
                in_d * in_h * in_w,
                in_h,
                in_w,
                out_d,
                out_h,
                out_w,
                a_d,
                a_h,
                a_w,
                off_d,
                off_h,
                off_w,
                kernel_d,
                kernel_h,
                kernel_w,
                EVEN=even,
                BLOCK=block,
                num_warps=max(2, block // 256),
                isCloseVectorization=True,
                buffer_size_limit=2048,
            )
        else:
            alpha_d = (in_d - kernel_d) / (out_d - 1) if out_d > 1 else 0.0
            alpha_h = (in_h - kernel_h) / (out_h - 1) if out_h > 1 else 0.0
            alpha_w = (in_w - kernel_w) / (out_w - 1) if out_w > 1 else 0.0
            static_taps = taps <= 8
            if not static_taps:
                block = 128
            elif numel >= 8192:
                block = 1024
            elif numel >= 1024:
                block = 512
            else:
                block = 256
            grid = (triton.cdiv(numel, block),)
            _fractional_max_pool3d_forward_kernel[grid](
                input,
                output,
                indices,
                random_samples,
                numel,
                in_d,
                in_h,
                in_w,
                out_d,
                out_h,
                out_w,
                kernel_d,
                kernel_h,
                kernel_w,
                alpha_d,
                alpha_h,
                alpha_w,
                STATIC_TAPS=static_taps,
                BLOCK=block,
                num_warps=4,
                isCloseVectorization=True,
                buffer_size_limit=2048,
            )
    if return_indices:
        return output, indices
    return output
