"""Accuracy tests for dwconv2d_hwc (channels-last depthwise conv2d)."""

import sys

import pytest
import torch
import torch.nn.functional as F

import flag_gems
import flag_gems.ops.dwconv2d_hwc
from flag_gems.ops.dwconv2d_hwc import dwconv2d_hwc
from flag_gems.utils.triton_version_utils import HAS_TLE

from .accuracy_utils import QUICK_MODE

dwconv2d_hwc_mod = sys.modules["flag_gems.ops.dwconv2d_hwc"]
# import flag_gems.ops.dwconv2d_hwc as dwconv2d_hwc_mod
# from flag_gems.ops.dwconv2d_hwc import dwconv2d_hwc


device = flag_gems.device


# Per-dtype tolerance. The kernel accumulates in fp32 and casts back, so it
# differs from the eager reference by ~1 ULP of the output dtype. fp16 has 10
# mantissa bits (1 ULP at magnitude ~8 is 0.0078); bf16 has only 7 (1 ULP at
# the same magnitude is 0.0625), so it needs a much looser bound.
_TOL = {
    torch.float32: dict(rtol=1e-3, atol=1e-3),
    torch.float16: dict(rtol=1e-2, atol=1e-2),
    torch.bfloat16: dict(rtol=5e-2, atol=5e-2),
}


def _assert_close(res, ref):
    assert res.dtype == ref.dtype
    torch.testing.assert_close(res.float(), ref.float(), **_TOL[res.dtype])


# ---------------------------------------------------------------------------
# PyTorch reference
# ---------------------------------------------------------------------------


def _ref_dwconv2d_hwc(inp, wgt):
    """Eager reference: HWC -> NCHW -> F.conv2d(groups=C) -> HWC."""
    C = inp.shape[2]
    KH, KW, _ = wgt.shape

    x = inp.permute(2, 0, 1).unsqueeze(0).float()  # (1, C, H, W)
    w = wgt.reshape(KH * KW, C).T.reshape(C, 1, KH, KW).float()  # (C, 1, KH, KW)
    y = F.conv2d(x, w, groups=C, padding=0)
    return y.squeeze(0).permute(1, 2, 0).to(inp.dtype)  # (OH, OW, C)


def _make_input(H, W, C, KH, KW, dtype, seed=0):
    torch.manual_seed(seed)
    inp = torch.randn(H, W, C, device=device, dtype=dtype)
    wgt = torch.randn(KH, KW, C, device=device, dtype=dtype)
    return inp, wgt


# ---------------------------------------------------------------------------
# Test matrices
# ---------------------------------------------------------------------------

DW_HW_LIST = [112, 128, 256, 512] if not QUICK_MODE else [128]
DW_C_LIST = [64, 128] if not QUICK_MODE else [64]
DW_K_LIST = [3, 5, 7] if not QUICK_MODE else [5]
DW_DTYPE_LIST = [torch.float32, torch.float16]


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.dwconv2d_hwc
@pytest.mark.parametrize("HW", DW_HW_LIST)
@pytest.mark.parametrize("C", DW_C_LIST)
@pytest.mark.parametrize("K", DW_K_LIST)
@pytest.mark.parametrize("dtype", DW_DTYPE_LIST)
def test_accuracy_dwconv2d_hwc(HW, C, K, dtype):
    inp, wgt = _make_input(HW, HW, C, K, K, dtype)

    ref = _ref_dwconv2d_hwc(inp, wgt)
    res = dwconv2d_hwc(inp, wgt)

    assert res.shape == (HW - K + 1, HW - K + 1, C)
    _assert_close(res, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.dwconv2d_hwc
@pytest.mark.parametrize("KH,KW", [(3, 5), (5, 3), (1, 7), (7, 1)])
def test_accuracy_dwconv2d_hwc_asymmetric_kernel(KH, KW):
    """KH != KW exercises the halo dims independently on each axis."""
    dtype = torch.float32
    inp, wgt = _make_input(128, 128, 64, KH, KW, dtype)

    ref = _ref_dwconv2d_hwc(inp, wgt)
    res = dwconv2d_hwc(inp, wgt)

    assert res.shape == (128 - KH + 1, 128 - KW + 1, 64)
    _assert_close(res, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.dwconv2d_hwc
@pytest.mark.parametrize("H,W", [(113, 127), (65, 129), (100, 100)])
def test_accuracy_dwconv2d_hwc_non_square(H, W):
    """Non-square, non-power-of-two spatial dims -- exercises the output masks."""
    dtype = torch.float32
    inp, wgt = _make_input(H, W, 64, 5, 5, dtype)

    ref = _ref_dwconv2d_hwc(inp, wgt)
    res = dwconv2d_hwc(inp, wgt)

    assert res.shape == (H - 5 + 1, W - 5 + 1, 64)
    _assert_close(res, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.dwconv2d_hwc
@pytest.mark.parametrize("C", [1, 3, 33, 65, 100])
def test_accuracy_dwconv2d_hwc_ragged_channels(C):
    """C not a multiple of TILE_C -- exercises the channel mask."""
    dtype = torch.float32
    inp, wgt = _make_input(64, 64, C, 3, 3, dtype)

    ref = _ref_dwconv2d_hwc(inp, wgt)
    res = dwconv2d_hwc(inp, wgt)

    assert res.shape == (62, 62, C)
    _assert_close(res, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.dwconv2d_hwc
@pytest.mark.parametrize(
    "tile_oh,tile_ow,tile_c", [(2, 2, 32), (4, 4, 64), (8, 8, 128)]
)
def test_accuracy_dwconv2d_hwc_tiling(tile_oh, tile_ow, tile_c):
    """Result must not depend on the tile sizes."""
    dtype = torch.float32
    inp, wgt = _make_input(128, 128, 64, 5, 5, dtype)

    ref = _ref_dwconv2d_hwc(inp, wgt)
    res = dwconv2d_hwc(inp, wgt, tile_oh=tile_oh, tile_ow=tile_ow, tile_c=tile_c)

    _assert_close(res, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.dwconv2d_hwc
def test_dwconv2d_hwc_kernel_equals_input_size():
    """KH == H, KW == W collapses the output to 1x1."""
    dtype = torch.float32
    inp, wgt = _make_input(7, 7, 32, 7, 7, dtype)

    ref = _ref_dwconv2d_hwc(inp, wgt)
    res = dwconv2d_hwc(inp, wgt)

    assert res.shape == (1, 1, 32)
    _assert_close(res, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.dwconv2d_hwc
def test_dwconv2d_hwc_non_contiguous_input():
    """A permuted (non-contiguous) input must still give the right answer."""
    dtype = torch.float32
    torch.manual_seed(0)
    # build it as CHW then permute to HWC -- non-contiguous by construction
    chw = torch.randn(64, 128, 128, device=device, dtype=dtype)
    inp = chw.permute(1, 2, 0)
    assert not inp.is_contiguous()
    wgt = torch.randn(5, 5, 64, device=device, dtype=dtype)

    ref = _ref_dwconv2d_hwc(inp, wgt)
    res = dwconv2d_hwc(inp, wgt)

    _assert_close(res, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.dwconv2d_hwc
def test_dwconv2d_hwc_does_not_clobber_input():
    dtype = torch.float32
    inp, wgt = _make_input(64, 64, 64, 3, 3, dtype)
    inp_orig = inp.clone()
    wgt_orig = wgt.clone()

    dwconv2d_hwc(inp, wgt)

    torch.testing.assert_close(inp, inp_orig, rtol=0, atol=0)
    torch.testing.assert_close(wgt, wgt_orig, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# TLE-vs-baseline parity
#
# The tests above validate whichever path the dispatcher picked. These force the
# other path so both kernels are covered on TLE-capable hardware.
# ---------------------------------------------------------------------------


def _has_tle_hw():
    if not (HAS_TLE and torch.cuda.is_available()):
        return False
    return torch.cuda.get_device_capability()[0] >= 9


@pytest.mark.skipif(
    not _has_tle_hw(), reason="requires Triton>=3.6 with TLE on Hopper+"
)
@pytest.mark.dwconv2d_hwc
@pytest.mark.parametrize("K", [3, 5, 7])
@pytest.mark.parametrize("dtype", DW_DTYPE_LIST)
def test_dwconv2d_hwc_tle_matches_baseline(K, dtype, monkeypatch):
    inp, wgt = _make_input(128, 128, 64, K, K, dtype)

    # TLE path (default on this HW)
    tle_out = dwconv2d_hwc(inp, wgt)

    # Force the baseline kernel
    # monkeypatch.setattr("flag_gems.ops.dwconv2d_hwc._tle_available", lambda _x: False)
    monkeypatch.setattr(dwconv2d_hwc_mod, "_tle_available", lambda _x: False)
    base_out = dwconv2d_hwc(inp, wgt)

    _assert_close(tle_out, base_out)
