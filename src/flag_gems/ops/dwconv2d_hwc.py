import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.device_info import get_device_capability
from flag_gems.utils.triton_version_utils import HAS_TLE

logger = logging.getLogger(__name__)

if HAS_TLE:
    import triton.experimental.tle.language as tle
else:
    tle = None

# TLE `extract_tile` lowers to efficient sub-tile slicing only on Hopper+.
_TLE_MIN_CAPABILITY = 9

# Default tiling. TILE_C is deliberately large: the kernel is channels-last, so
# the C axis is contiguous and wants a wide tile.
_DEFAULT_TILE_OH = 4
_DEFAULT_TILE_OW = 4
_DEFAULT_TILE_C = 64


def _next_pow2(x: int) -> int:
    return 1 if x <= 1 else 2 ** math.ceil(math.log2(x))


# ---------------------------------------------------------------------------
# Baseline kernel (no TLE)
#
# Reloads the input patch from global memory once per (kh, kw) tap. Every tap
# re-reads the same rows with a one-pixel shift, so the halo is fetched KH*KW
# times.
# ---------------------------------------------------------------------------


@triton.jit
def _dwconv2d_hwc_kernel(
    inp_ptr,  # (H, W, C)
    wgt_ptr,  # (KH, KW, C)
    out_ptr,  # (OH, OW, C)
    H,
    W,
    C,
    OH,
    OW,
    KH: tl.constexpr,
    KW: tl.constexpr,
    TILE_OH: tl.constexpr,
    TILE_OW: tl.constexpr,
    TILE_C: tl.constexpr,
):
    pid_oh = tl.program_id(0)
    pid_ow = tl.program_id(1)
    pid_c = tl.program_id(2)

    oh0 = pid_oh * TILE_OH
    ow0 = pid_ow * TILE_OW
    c0 = pid_c * TILE_C

    offs_c = c0 + tl.arange(0, TILE_C)
    offs_oh = oh0 + tl.arange(0, TILE_OH)
    offs_ow = ow0 + tl.arange(0, TILE_OW)
    c_mask = offs_c < C

    acc = tl.zeros((TILE_OH, TILE_OW, TILE_C), dtype=tl.float32)

    for kh in tl.static_range(KH):
        for kw in tl.static_range(KW):
            w_ptrs = wgt_ptr + (kh * KW + kw) * C + offs_c
            w = tl.load(w_ptrs, mask=c_mask, other=0.0)

            ih = offs_oh + kh
            iw = offs_ow + kw
            ih_ok = ih < H
            iw_ok = iw < W

            inp_ptrs = (
                inp_ptr
                + ih[:, None, None] * (W * C)
                + iw[None, :, None] * C
                + offs_c[None, None, :]
            )
            mask = ih_ok[:, None, None] & iw_ok[None, :, None] & c_mask[None, None, :]
            x = tl.load(inp_ptrs, mask=mask, other=0.0)

            acc += x * w[None, None, :]

    out_ptrs = (
        out_ptr
        + offs_oh[:, None, None] * (OW * C)
        + offs_ow[None, :, None] * C
        + offs_c[None, None, :]
    )
    out_mask = (
        (offs_oh[:, None, None] < OH)
        & (offs_ow[None, :, None] < OW)
        & c_mask[None, None, :]
    )
    tl.store(out_ptrs, acc, mask=out_mask)


# ---------------------------------------------------------------------------
# TLE kernel (Triton >= 3.6, Hopper+)
#
# Same math, but the halo region (TILE_OH + KH - 1, TILE_OW + KW - 1, TILE_C) is
# loaded ONCE, and each (kh, kw) tap is sliced out of it with `tle.extract_tile`
# instead of issuing another global load. Global traffic drops from KH*KW halo
# reads to one.
# ---------------------------------------------------------------------------

if HAS_TLE:

    @triton.jit
    def _dwconv2d_hwc_kernel_tle(
        inp_ptr,
        wgt_ptr,
        out_ptr,
        H,
        W,
        C,
        OH,
        OW,
        KH: tl.constexpr,
        KW: tl.constexpr,
        TILE_OH: tl.constexpr,
        TILE_OW: tl.constexpr,
        TILE_C: tl.constexpr,
        HALO_H: tl.constexpr,  # TILE_OH + KH - 1
        HALO_W: tl.constexpr,  # TILE_OW + KW - 1
        HALO_H_P2: tl.constexpr,  # next_pow2(HALO_H) -- tl.arange needs pow2
        HALO_W_P2: tl.constexpr,
    ):
        pid_oh = tl.program_id(0)
        pid_ow = tl.program_id(1)
        pid_c = tl.program_id(2)

        oh0 = pid_oh * TILE_OH
        ow0 = pid_ow * TILE_OW
        c0 = pid_c * TILE_C

        offs_c = c0 + tl.arange(0, TILE_C)
        c_mask = offs_c < C

        # Halo tile: covers every input pixel any tap in this output tile reads.
        # Rounded up to a power of two for tl.arange, then masked back down.
        offs_ih = oh0 + tl.arange(0, HALO_H_P2)
        offs_iw = ow0 + tl.arange(0, HALO_W_P2)

        ih_ok = (offs_ih < oh0 + HALO_H) & (offs_ih < H)
        iw_ok = (offs_iw < ow0 + HALO_W) & (offs_iw < W)

        halo_ptrs = (
            inp_ptr
            + offs_ih[:, None, None] * (W * C)
            + offs_iw[None, :, None] * C
            + offs_c[None, None, :]
        )
        halo_mask = ih_ok[:, None, None] & iw_ok[None, :, None] & c_mask[None, None, :]
        halo = tl.load(halo_ptrs, mask=halo_mask, other=0.0)

        acc = tl.zeros((TILE_OH, TILE_OW, TILE_C), dtype=tl.float32)

        for kh in tl.static_range(KH):
            for kw in tl.static_range(KW):
                w_ptrs = wgt_ptr + (kh * KW + kw) * C + offs_c
                w = tl.load(w_ptrs, mask=c_mask, other=0.0)

                # The (kh, kw) tap reads the halo shifted by (kh, kw).
                patch = tle.extract_tile(
                    halo,
                    index=[kh, kw, 0],
                    tile_shape=[TILE_OH, TILE_OW, TILE_C],
                    strides=[1, 1, 1],
                )
                acc += patch * w[None, None, :]

        offs_oh = oh0 + tl.arange(0, TILE_OH)
        offs_ow = ow0 + tl.arange(0, TILE_OW)
        out_ptrs = (
            out_ptr
            + offs_oh[:, None, None] * (OW * C)
            + offs_ow[None, :, None] * C
            + offs_c[None, None, :]
        )
        out_mask = (
            (offs_oh[:, None, None] < OH)
            & (offs_ow[None, :, None] < OW)
            & c_mask[None, None, :]
        )
        tl.store(out_ptrs, acc, mask=out_mask)

else:
    _dwconv2d_hwc_kernel_tle = None


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def _tle_available(x: torch.Tensor) -> bool:
    if not HAS_TLE:
        return False
    if x.device.type != "cuda":
        return False
    return get_device_capability()[0] >= _TLE_MIN_CAPABILITY


def dwconv2d_hwc(
    input: torch.Tensor,
    weight: torch.Tensor,
    tile_oh: int = _DEFAULT_TILE_OH,
    tile_ow: int = _DEFAULT_TILE_OW,
    tile_c: int = _DEFAULT_TILE_C,
) -> torch.Tensor:
    """Channels-last depthwise 2D convolution, valid padding, unit stride.

    This is NOT a general conv2d and is not an aten override. It is a narrow,
    channels-last (HWC) depthwise kernel: one filter per channel, no batch
    dimension, no bias, stride == dilation == 1, padding == 0. The equivalent
    eager computation is::

        x = input.permute(2, 0, 1).unsqueeze(0)                  # (1, C, H, W)
        w = weight.reshape(KH * KW, C).T.reshape(C, 1, KH, KW)   # (C, 1, KH, KW)
        y = F.conv2d(x, w, groups=C)
        out = y.squeeze(0).permute(1, 2, 0)                      # (OH, OW, C)

    For general NCHW convolution with bias, stride, padding, groups, and a
    backward pass, use ``flag_gems.ops.conv2d`` instead.

    Args:
        input: (H, W, C) channels-last activations.
        weight: (KH, KW, C) one filter per channel.
        tile_oh, tile_ow, tile_c: output tile sizes.

    Returns:
        (OH, OW, C) with OH = H - KH + 1, OW = W - KW + 1, same dtype as input.
    """
    logger.debug("GEMS DWCONV2D_HWC")

    assert input.ndim == 3, f"input must be (H, W, C), got shape {tuple(input.shape)}"
    assert weight.ndim == 3, f"weight must be (KH, KW, C), got {tuple(weight.shape)}"
    assert (
        input.shape[2] == weight.shape[2]
    ), f"channel mismatch: input C={input.shape[2]}, weight C={weight.shape[2]}"
    assert (
        input.dtype == weight.dtype
    ), f"dtype mismatch: input {input.dtype}, weight {weight.dtype}"

    input = input.contiguous()
    weight = weight.contiguous()

    H, W, C = input.shape
    KH, KW, _ = weight.shape

    assert KH <= H and KW <= W, f"kernel {KH}x{KW} larger than input {H}x{W}"

    OH = H - KH + 1
    OW = W - KW + 1

    out = torch.empty((OH, OW, C), device=input.device, dtype=input.dtype)

    grid = (
        triton.cdiv(OH, tile_oh),
        triton.cdiv(OW, tile_ow),
        triton.cdiv(C, tile_c),
    )

    with torch_device_fn.device(input.device):
        if _tle_available(input):
            halo_h = tile_oh + KH - 1
            halo_w = tile_ow + KW - 1
            _dwconv2d_hwc_kernel_tle[grid](
                input,
                weight,
                out,
                H,
                W,
                C,
                OH,
                OW,
                KH=KH,
                KW=KW,
                TILE_OH=tile_oh,
                TILE_OW=tile_ow,
                TILE_C=tile_c,
                HALO_H=halo_h,
                HALO_W=halo_w,
                HALO_H_P2=_next_pow2(halo_h),
                HALO_W_P2=_next_pow2(halo_w),
            )
        else:
            _dwconv2d_hwc_kernel[grid](
                input,
                weight,
                out,
                H,
                W,
                C,
                OH,
                OW,
                KH=KH,
                KW=KW,
                TILE_OH=tile_oh,
                TILE_OW=tile_ow,
                TILE_C=tile_c,
            )

    return out
