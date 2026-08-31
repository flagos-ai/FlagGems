"""Performance benchmarks for dwconv2d_hwc.

Unlike causal_conv1d this op IS stateless and DOES have a torch reference, so we
benchmark three providers side by side: baseline Triton, TLE, and torch.

Note the torch column is not a like-for-like comparison: torch works in NCHW, so
its timing includes the two permutes needed to get in and out of HWC. It is here
as a sanity floor, not as the thing to beat.
"""

import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import triton

import flag_gems
import flag_gems.ops.dwconv2d_hwc  # noqa: F401
from flag_gems.utils.triton_version_utils import HAS_TLE

dw = sys.modules["flag_gems.ops.dwconv2d_hwc"]
device = flag_gems.device

WARMUP = 25
REP = 100
RUNS = 5  # repeats of do_bench, for the spread


def _has_tle_hw():
    if not (HAS_TLE and torch.cuda.is_available()):
        return False
    return torch.cuda.get_device_capability()[0] >= 9


def _make_input(H, W, C, KH, KW, dtype, seed=0):
    torch.manual_seed(seed)
    inp = torch.randn(H, W, C, device=device, dtype=dtype)
    wgt = torch.randn(KH, KW, C, device=device, dtype=dtype)
    return inp, wgt


def _torch_dwconv(inp, wgt):
    C = inp.shape[2]
    KH, KW, _ = wgt.shape
    x = inp.permute(2, 0, 1).unsqueeze(0)
    w = wgt.reshape(KH * KW, C).T.reshape(C, 1, KH, KW)
    y = F.conv2d(x, w, groups=C, padding=0)
    return y.squeeze(0).permute(1, 2, 0)


def _run_with_tle(enabled, fn):
    """Run fn with the TLE dispatcher forced on or off."""
    orig = dw._tle_available
    try:
        if not enabled:
            dw._tle_available = lambda _x: False
        return fn()
    finally:
        dw._tle_available = orig


def _bench(fn):
    """do_bench a few times; return the median-of-medians and the p90."""
    p50s = []
    for _ in range(RUNS):
        ms = triton.testing.do_bench(fn, warmup=WARMUP, rep=REP, quantiles=[0.5])
        p50s.append(ms if isinstance(ms, float) else ms[0])
    p50s = np.array(p50s)
    return float(np.median(p50s)), float(np.percentile(p50s, 90))


def _gbps(H, W, C, KH, KW, dtype, ms):
    """Achieved bandwidth. Depthwise conv is memory bound -- this is the metric
    that matters, not TFLOPS."""
    itemsize = torch.tensor([], dtype=dtype).element_size()
    OH, OW = H - KH + 1, W - KW + 1
    # ideal traffic: read the input once, write the output once
    nbytes = (H * W * C + OH * OW * C) * itemsize
    return nbytes / (ms * 1e-3) / 1e9


def _tflops(H, W, C, KH, KW, ms):
    OH, OW = H - KH + 1, W - KW + 1
    flops = 2 * OH * OW * C * KH * KW  # one FMA per (pixel, channel, tap)
    return flops / (ms * 1e-3) / 1e12


# ---------------------------------------------------------------------------
# Spatial sweep
# ---------------------------------------------------------------------------

HW_VALS = [112, 128, 256, 512]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.dwconv2d_hwc
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_perf_dwconv2d_hwc(dtype):
    C, K = 64, 5
    tle_on = _has_tle_hw()

    print(
        f"\n=== dwconv2d_hwc | C={C} K={K}x{K} {dtype} | "
        f"TLE={'yes' if tle_on else 'no (baseline only)'} ==="
    )
    header = (
        f"{'H=W':<8} {'base ms':<10} {'base GB/s':<11} {'base TF':<9} "
        f"{'torch ms':<10}"
    )
    if tle_on:
        header += f" {'tle ms':<10} {'tle GB/s':<11} {'speedup':<9} {'match':<6}"
    print(header)

    for HW in HW_VALS:
        inp, wgt = _make_input(HW, HW, C, K, K, dtype)

        base_ms, _ = _run_with_tle(
            False, lambda: _bench(lambda: dw.dwconv2d_hwc(inp, wgt))
        )
        torch_ms, _ = _bench(lambda: _torch_dwconv(inp, wgt))

        row = (
            f"{HW:<8} {base_ms:<10.4f} "
            f"{_gbps(HW, HW, C, K, K, dtype, base_ms):<11.1f} "
            f"{_tflops(HW, HW, C, K, K, base_ms):<9.2f} "
            f"{torch_ms:<10.4f}"
        )

        if tle_on:
            tle_ms, _ = _bench(lambda: dw.dwconv2d_hwc(inp, wgt))

            # correctness spot-check alongside the timing
            out_tle = dw.dwconv2d_hwc(inp, wgt)
            out_base = _run_with_tle(False, lambda: dw.dwconv2d_hwc(inp, wgt))
            atol = 1e-2 if dtype == torch.float16 else 1e-3
            try:
                torch.testing.assert_close(
                    out_tle.float(), out_base.float(), rtol=atol, atol=atol
                )
                match = "pass"
            except AssertionError:
                match = "FAIL"

            speedup = base_ms / tle_ms if tle_ms > 0 else float("nan")
            row += (
                f" {tle_ms:<10.4f} "
                f"{_gbps(HW, HW, C, K, K, dtype, tle_ms):<11.1f} "
                f"{speedup:<9.2f} {match:<6}"
            )

        print(row)


# ---------------------------------------------------------------------------
# Kernel-size sweep
#
# This is the one that tells you whether extract_tile is doing its job. The
# baseline re-reads the halo once per tap, so its traffic grows as KH*KW; the TLE
# version loads the halo once regardless. Speedup should therefore INCREASE with
# K. If it stays flat, extract_tile is not buying what it should.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_tle_hw(), reason="requires TLE on Hopper+")
@pytest.mark.dwconv2d_hwc
def test_perf_dwconv2d_hwc_kernel_sweep():
    dtype = torch.float32
    HW, C = 256, 64

    print(f"\n=== dwconv2d_hwc kernel sweep | H=W={HW} C={C} {dtype} ===")
    print(f"{'K':<6} {'base ms':<10} {'tle ms':<10} {'speedup':<9} {'taps':<6}")

    for K in (3, 5, 7, 9):
        inp, wgt = _make_input(HW, HW, C, K, K, dtype)
        base_ms, _ = _run_with_tle(
            False, lambda: _bench(lambda: dw.dwconv2d_hwc(inp, wgt))
        )
        tle_ms, _ = _bench(lambda: dw.dwconv2d_hwc(inp, wgt))
        print(
            f"{K:<6} {base_ms:<10.4f} {tle_ms:<10.4f} "
            f"{base_ms / tle_ms:<9.2f} {K * K:<6}"
        )


# ---------------------------------------------------------------------------
# Tile-size sweep -- the halo is (TILE_OH + K - 1) x (TILE_OW + K - 1), so small
# tiles pay a large halo-to-body ratio. Useful for picking the defaults.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.dwconv2d_hwc
def test_perf_dwconv2d_hwc_tile_sweep():
    dtype = torch.float32
    HW, C, K = 256, 64, 5
    inp, wgt = _make_input(HW, HW, C, K, K, dtype)

    print(f"\n=== dwconv2d_hwc tile sweep | H=W={HW} C={C} K={K}x{K} {dtype} ===")
    print(f"{'tile':<14} {'ms':<10} {'GB/s':<11} {'halo/body':<10}")

    for toh, tow, tc in [
        (2, 2, 64),
        (4, 4, 32),
        (4, 4, 64),
        (4, 4, 128),
        (8, 8, 64),
        (16, 16, 64),
    ]:
        ms, _ = _bench(lambda: dw.dwconv2d_hwc(inp, wgt, toh, tow, tc))
        halo = (toh + K - 1) * (tow + K - 1)
        body = toh * tow
        print(
            f"{f'{toh}x{tow}x{tc}':<14} {ms:<10.4f} "
            f"{_gbps(HW, HW, C, K, K, dtype, ms):<11.1f} "
            f"{halo / body:<10.2f}"
        )
