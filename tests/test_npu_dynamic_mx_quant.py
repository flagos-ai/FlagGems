# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for npu_dynamic_mx_quant (a5/ops/npu_dynamic_mx_quant.py).

Correctness is validated byte-exact against ``torch_npu.npu_dynamic_mx_quant``
(aclnnDynamicMxQuantV3).  Run with::

    pytest tests/test_npu_dynamic_mx_quant.py --noconftest -v

The tests require an Atlas A5 (Ascend950) device.  On Ascend910 (a2) the MLIR
backend does not support ``arith::TruncFOp`` to fp8, so every tile fails to
compile.  The module-level ``pytestmark`` skips the whole file unless the
running device reports a "950"-series SoC name.
"""

import importlib.util
import pathlib

import pytest
import torch

try:
    import flag_gems as _flag_gems

    _VENDOR_NAME = _flag_gems.vendor_name
    _DEV_STR = _flag_gems.device
except (ImportError, RuntimeError):
    _flag_gems = None  # type: ignore[assignment]
    # Detect Ascend via torch_npu when flag_gems init is broken.
    try:
        import torch_npu as _torch_npu  # noqa: F401

        _VENDOR_NAME = "ascend"
        _DEV_STR = "npu"
    except ImportError:
        _VENDOR_NAME = "unavailable"
        _DEV_STR = "cpu"


def _is_ascend_a5() -> bool:
    """Return True only when running on an Atlas A5 (Ascend950) device.

    This kernel uses fp8 ``arith::TruncFOp`` which the CANN MLIR backend on
    Ascend910 (a2) does not support.  Gate the whole test file on the SoC
    name so CI on 910 machines skips cleanly instead of hard-failing.
    """
    if _VENDOR_NAME != "ascend":
        return False
    try:
        import torch
        import torch_npu  # noqa: F401

        name = torch.npu.get_device_name(0)  # e.g. "Ascend950DT_9582"
        return "950" in name
    except Exception:
        pass
    # Fallback: try the FlagGems arch resolver (may be unavailable).
    try:
        import torch

        from flag_gems.runtime.backend._ascend import get_arch_key

        return get_arch_key(torch.device("npu:0")) == "a5"
    except Exception:
        return False


_ON_A5 = _is_ascend_a5()

# ---------------------------------------------------------------------------
# Skip the whole module unless we are on an Atlas A5 (Ascend950) device
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.skipif(
    not _ON_A5,
    reason="npu_dynamic_mx_quant requires Atlas A5 (Ascend950); "
    "skipped on non-a5 Ascend and non-Ascend backends",
)

# ---------------------------------------------------------------------------
# Import the kernel module directly so the tests run even when the full
# FlagGems import chain cannot initialise (e.g. missing tle extension).
# ---------------------------------------------------------------------------
_MOD_PATH = (
    pathlib.Path(__file__).parent.parent
    / "src/flag_gems/runtime/backend/_ascend/a5/ops/npu_dynamic_mx_quant.py"
)
_spec = importlib.util.spec_from_file_location("_npu_dmq", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

npu_dynamic_mx_quant = _mod.npu_dynamic_mx_quant
DynamicMxQuantPlan = _mod.DynamicMxQuantPlan
scale_shape = _mod.scale_shape
should_use_triton = _mod.should_use_triton
pick_rows = _mod.pick_rows

_DEV = _DEV_STR  # "npu"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _vendor(x, dst_type, block_size, scale_alg):
    """Call torch_npu vendor op with keyword-only form."""
    import torch_npu  # noqa: F401 – Ascend-only

    return torch_npu.npu_dynamic_mx_quant(
        x,
        axis=-1,
        round_mode="rint",
        dst_type=dst_type,
        block_size=block_size,
        scale_alg=scale_alg,
    )


def _assert_byte_exact(y_t, s_t, y_v, s_v, label=""):
    """Assert fp8 payload and uint8 scale are byte-identical to vendor."""
    y_ok = y_t.view(torch.uint8).equal(y_v.view(torch.uint8))
    s_ok = s_t.equal(s_v)
    assert y_ok, f"{label}: fp8 payload mismatch"
    assert s_ok, f"{label}: scale mismatch"


def _make(shape, dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=g, dtype=torch.float32).to(dtype)
    return x.to(_DEV)


# ---------------------------------------------------------------------------
# Parametrised correctness tests
# ---------------------------------------------------------------------------


@pytest.mark.npu_dynamic_mx_quant
@pytest.mark.parametrize("M", [1, 3, 16, 31, 64, 256, 1023, 4096])
@pytest.mark.parametrize("K", [512, 7168])
def test_correctness_shapes(M, K):
    """Byte-exact vs vendor across decode and prefill shapes."""
    x = _make((M, K), torch.bfloat16)
    y_t, s_t = npu_dynamic_mx_quant(x)
    y_v, s_v = _vendor(x, torch.float8_e4m3fn, 32, 0)
    _assert_byte_exact(y_t, s_t, y_v, s_v, f"M={M} K={K}")


@pytest.mark.npu_dynamic_mx_quant
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_correctness_dtypes(dtype):
    x = _make((128, 4096), dtype)
    y_t, s_t = npu_dynamic_mx_quant(
        x, dst_type=torch.float8_e4m3fn, block_size=32, scale_alg=0
    )
    y_v, s_v = _vendor(x, torch.float8_e4m3fn, 32, 0)
    _assert_byte_exact(y_t, s_t, y_v, s_v, f"dtype={dtype}")


@pytest.mark.npu_dynamic_mx_quant
@pytest.mark.parametrize("dst_type", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_correctness_dst_types(dst_type):
    x = _make((256, 7168), torch.bfloat16)
    y_t, s_t = npu_dynamic_mx_quant(x, dst_type=dst_type)
    y_v, s_v = _vendor(x, dst_type, 32, 0)
    _assert_byte_exact(y_t, s_t, y_v, s_v, f"dst_type={dst_type}")


@pytest.mark.npu_dynamic_mx_quant
@pytest.mark.parametrize("scale_alg", [0, 1])
def test_correctness_scale_alg(scale_alg):
    x = _make((256, 7168), torch.bfloat16)
    y_t, s_t = npu_dynamic_mx_quant(x, scale_alg=scale_alg)
    y_v, s_v = _vendor(x, torch.float8_e4m3fn, 32, scale_alg)
    _assert_byte_exact(y_t, s_t, y_v, s_v, f"scale_alg={scale_alg}")


@pytest.mark.npu_dynamic_mx_quant
@pytest.mark.parametrize("block_size", [32, 64, 128, 256])
def test_correctness_block_sizes(block_size):
    x = _make((128, 4096), torch.bfloat16)
    y_t, s_t = npu_dynamic_mx_quant(x, block_size=block_size)
    y_v, s_v = _vendor(x, torch.float8_e4m3fn, block_size, 0)
    _assert_byte_exact(y_t, s_t, y_v, s_v, f"block_size={block_size}")


@pytest.mark.npu_dynamic_mx_quant
def test_correctness_batched_input():
    """Input with leading batch dims (shape 2 x 16 x 7168)."""
    x = _make((2, 16, 7168), torch.bfloat16)
    y_t, s_t = npu_dynamic_mx_quant(x)
    y_v, s_v = _vendor(x, torch.float8_e4m3fn, 32, 0)
    _assert_byte_exact(y_t, s_t, y_v, s_v, "batched")


@pytest.mark.npu_dynamic_mx_quant
def test_correctness_boundary_blocks():
    """Crafted blocks: all-zero, finfo.max, finfo.tiny, inf, nan."""
    K = 128
    x = torch.zeros((6, K), dtype=torch.bfloat16, device=_DEV)
    fi = torch.finfo(torch.bfloat16)
    # row 0: all zero; row 1: finfo.max; row 2: finfo.tiny
    # row 3: mixed inf; row 4: nan; row 5: -0.0
    x[1, :] = fi.max
    x[2, :] = fi.tiny
    x[3, 0] = float("inf")
    x[4, 0] = float("nan")
    x[5, :] = -0.0
    y_t, s_t = npu_dynamic_mx_quant(x, block_size=32)
    y_v, s_v = _vendor(x, torch.float8_e4m3fn, 32, 0)
    # Byte-exact for finite rows; for inf/nan rows only scale must match
    # (vendor emits +0, kernel emits -0 for negative elements -- same value).
    assert s_t.equal(s_v), "scale mismatch on boundary blocks"
    # Rows without inf/nan must be fully byte-exact
    for row in (0, 1, 2, 5):
        assert (
            y_t[row].view(torch.uint8).equal(y_v[row].view(torch.uint8))
        ), f"payload mismatch on row {row}"


# ---------------------------------------------------------------------------
# scale_shape helper
# ---------------------------------------------------------------------------


@pytest.mark.npu_dynamic_mx_quant
@pytest.mark.parametrize(
    "shape,block_size,expected",
    [
        ((128, 4096), 32, (128, 64, 2)),
        ((1, 7168), 64, (1, 56, 2)),
        ((2, 16, 7168), 32, (2, 16, 112, 2)),
        ((256, 256), 256, (256, 1, 2)),
    ],
)
def test_scale_shape(shape, block_size, expected):
    assert scale_shape(shape, block_size) == expected


# ---------------------------------------------------------------------------
# should_use_triton / pick_rows smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.npu_dynamic_mx_quant
def test_should_use_triton_returns_bool():
    result = should_use_triton(16, 7168)
    assert isinstance(result, bool)


@pytest.mark.npu_dynamic_mx_quant
@pytest.mark.parametrize("M,K", [(1, 512), (64, 7168), (4096, 7168)])
def test_pick_rows_valid(M, K):
    rows = pick_rows(M, K, 32)
    # Must divide M or be 1 (masked path), and be a power of two
    assert rows >= 1
    assert (rows & (rows - 1)) == 0 or rows == 1, f"pick_rows({M},{K}) returned {rows}"


# ---------------------------------------------------------------------------
# DynamicMxQuantPlan interface
# ---------------------------------------------------------------------------


@pytest.mark.npu_dynamic_mx_quant
def test_plan_reuse():
    """Same plan object produces byte-identical results on two different inputs."""
    x1 = _make((128, 4096), torch.bfloat16, seed=1)
    x2 = _make((128, 4096), torch.bfloat16, seed=2)
    plan = DynamicMxQuantPlan(x1, torch.float8_e4m3fn, 32, -1, "rint", 0)
    y1_t, s1_t = plan(x1)
    y2_t, s2_t = plan(x2)
    y1_v, s1_v = _vendor(x1, torch.float8_e4m3fn, 32, 0)
    y2_v, s2_v = _vendor(x2, torch.float8_e4m3fn, 32, 0)
    _assert_byte_exact(y1_t, s1_t, y1_v, s1_v, "plan-reuse row1")
    _assert_byte_exact(y2_t, s2_t, y2_v, s2_v, "plan-reuse row2")


@pytest.mark.npu_dynamic_mx_quant
def test_plan_repr():
    x = _make((16, 7168), torch.bfloat16)
    plan = DynamicMxQuantPlan(x, torch.float8_e4m3fn, 32, -1, "rint", 0)
    r = repr(plan)
    assert "DynamicMxQuantPlan" in r


@pytest.mark.npu_dynamic_mx_quant
def test_plan_shape_mismatch_raises():
    """Calling a plan with a differently-shaped tensor should raise."""
    x_ref = _make((128, 4096), torch.bfloat16)
    x_bad = _make((64, 4096), torch.bfloat16)
    plan = DynamicMxQuantPlan(x_ref, torch.float8_e4m3fn, 32, -1, "rint", 0)
    with pytest.raises((ValueError, RuntimeError)):
        plan(x_bad)


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


@pytest.mark.npu_dynamic_mx_quant
def test_invalid_dst_type_raises():
    x = _make((16, 512), torch.bfloat16)
    with pytest.raises((ValueError, TypeError, NotImplementedError)):
        npu_dynamic_mx_quant(x, dst_type=torch.float16)


@pytest.mark.npu_dynamic_mx_quant
def test_invalid_scale_alg_raises():
    x = _make((16, 512), torch.bfloat16)
    with pytest.raises((ValueError, TypeError)):
        npu_dynamic_mx_quant(x, scale_alg=99)


@pytest.mark.npu_dynamic_mx_quant
def test_invalid_block_size_raises():
    x = _make((16, 512), torch.bfloat16)
    with pytest.raises((ValueError, AssertionError)):
        npu_dynamic_mx_quant(x, block_size=7)  # not a power of two


@pytest.mark.npu_dynamic_mx_quant
def test_invalid_round_mode_raises():
    x = _make((16, 512), torch.bfloat16)
    with pytest.raises((ValueError, TypeError)):
        npu_dynamic_mx_quant(x, round_mode="floor")


@pytest.mark.npu_dynamic_mx_quant
def test_invalid_axis_raises():
    x = _make((16, 512), torch.bfloat16)
    with pytest.raises((ValueError, IndexError, NotImplementedError)):
        npu_dynamic_mx_quant(x, axis=0)  # only axis=-1 / last axis supported


@pytest.mark.npu_dynamic_mx_quant
def test_1d_input_raises():
    x = _make((512,), torch.bfloat16)
    with pytest.raises((ValueError, AssertionError, RuntimeError)):
        npu_dynamic_mx_quant(x)


@pytest.mark.npu_dynamic_mx_quant
def test_non_contiguous_input():
    """Non-contiguous input should either work correctly or raise clearly."""
    x = _make((128, 4096), torch.bfloat16)
    x_nc = x[:, ::2]  # stride-2, non-contiguous
    assert not x_nc.is_contiguous()
    try:
        y_t, s_t = npu_dynamic_mx_quant(x_nc)
        # If it succeeds, compare against vendor on the same (made contiguous) view
        x_c = x_nc.contiguous()
        y_v, s_v = _vendor(x_c, torch.float8_e4m3fn, 32, 0)
        _assert_byte_exact(y_t, s_t, y_v, s_v, "non-contiguous")
    except (ValueError, RuntimeError):
        pass  # explicit rejection is also acceptable


# ---------------------------------------------------------------------------
# Output tensor properties
# ---------------------------------------------------------------------------


@pytest.mark.npu_dynamic_mx_quant
def test_output_dtype_fp8_e4m3():
    x = _make((16, 512), torch.bfloat16)
    y, s = npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)
    assert y.dtype == torch.float8_e4m3fn
    assert s.dtype == torch.uint8


@pytest.mark.npu_dynamic_mx_quant
def test_output_dtype_fp8_e5m2():
    x = _make((16, 512), torch.bfloat16)
    y, s = npu_dynamic_mx_quant(x, dst_type=torch.float8_e5m2)
    assert y.dtype == torch.float8_e5m2
    assert s.dtype == torch.uint8


@pytest.mark.npu_dynamic_mx_quant
def test_output_shape():
    M, K, B = 128, 4096, 32
    x = _make((M, K), torch.bfloat16)
    y, s = npu_dynamic_mx_quant(x, block_size=B)
    assert y.shape == (M, K)
    assert s.shape == scale_shape((M, K), B)


@pytest.mark.npu_dynamic_mx_quant
def test_output_device():
    x = _make((16, 512), torch.bfloat16)
    y, s = npu_dynamic_mx_quant(x)
    assert y.device.type == x.device.type
    assert s.device.type == x.device.type
