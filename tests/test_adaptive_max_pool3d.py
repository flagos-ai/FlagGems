"""
Test cases for adaptive_max_pool3d — full correctness coverage.

Covers all 9 kernel dispatch paths, known model shapes, edge cases,
special values (NaN, ties, zeros/negatives), empty tensors, and
integer output_size broadcasting.

Kernel dispatch paths (from adaptive_max_pool3d in operator source):
  Path A — Identity:       in_d==out_d && in_h==out_h && in_w==out_w
  Path B — out_d=1 fast:   out_d==1 && in_d>1 → D-reduce then 2D pooling
  Path C — Global max.dim:  (1,1,1) && spatial >= 4096
  Path D — Global blk_red:  (1,1,1) && 64 <= spatial < 4096
  Path E — Global→1D:       (1,1,1) && spatial < 64
  Path F — Large window:    total<=4096 && win>=1024 && blk_est<156
  Path G — 1D kernel:       (total<=4096 && !prefer_2d) ||
                             (total>4096 && win<=2048)
  Path H — 2D fast:         !1D && out_h<=16 && out_w<=16
  Path I — 2D regular:      !1D && (out_h>16 || out_w>16)
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES

# ============================================================================
# Curated (shape, output_size) pairs — each exercises a specific code path
# or real-world model shape.
# ============================================================================

# To keep parametrization manageable, shapes and output_sizes are grouped
# by category.  For each group we test only the compatible combinations
# (output_size ≤ input_spatial element-wise).


def _compatible(shape, out_size):
    """Return True if output_size is element-wise <= input spatial dims."""
    return all(o <= i for o, i in zip(out_size, shape[2:]))


# ============================================================================
# SHAPE GROUPS
# ============================================================================

# --- Path A: Identity ---
IDENTITY_SHAPES = [
    ((2, 256, 2, 14, 14), (2, 14, 14), "Path A: exact identity"),
    ((2, 128, 8, 16, 16), (8, 16, 16), "Path A: power-of-2 identity"),
    ((2, 256, 4, 7, 7), (4, 7, 7), "Path A: odd spatial identity"),
    ((4, 64, 8, 32, 32), (8, 32, 32), "Path A: large identity"),
]

# --- Path B: out_d=1 fast path ---
PATH_B_SHAPES = [
    ((1, 8, 64, 256, 256), (1, 7, 7), "Path B: extreme H/W reduction"),
    ((1, 256, 64, 112, 112), (1, 112, 112), "Path B: T=64→1, spatial preserved"),
    # B → global pool (1,1,1)
    ((1, 4096, 64, 64, 64), (1, 1, 1), "B→C: reduced spatial=4096"),
    ((1, 4, 8, 24, 32), (1, 1, 1), "B→D: reduced spatial=768"),
    # Video model classifier head (B→E: reduced spatial<64)
    ((1, 1024, 8, 7, 7), (1, 1, 1), "B→E: I3D Mixed_5c"),
    # B → large window downstream
    ((8, 64, 16, 112, 112), (1, 7, 7), "B: aggressive spatial reduction"),
    ((2, 256, 16, 112, 112), (1, 56, 56), "B: large HW after reduction"),
]

# --- Path C: Global torch.max (direct: in_d=1, spatial >= 4096) ---
PATH_C_SHAPES = [
    ((1, 256, 1, 64, 64), (1, 1, 1), "Path C: spatial=4096 boundary"),
    ((1, 64, 1, 256, 256), (1, 1, 1), "Path C: spatial=65536"),
]

# --- Path D: Global block_reduce (direct: in_d=1, 64 ≤ spatial < 4096) ---
PATH_D_SHAPES = [
    ((2, 512, 1, 8, 8), (1, 1, 1), "Path D: spatial=64 boundary"),
    ((1, 64, 1, 28, 28), (1, 1, 1), "Path D: spatial=784"),
]

# --- Path E: Global→1D (direct: in_d=1, spatial < 64) ---
PATH_E_SHAPES = [
    ((2, 512, 1, 7, 7), (1, 1, 1), "Path E: R3D-18 head, sp=49"),
    ((1, 1024, 1, 4, 4), (1, 1, 1), "Path E: sp=16"),
]

# --- Temporal compression (D reduced, H/W preserved) → mostly Path G ---
TEMPORAL_COMPRESSION_SHAPES = [
    ((1, 768, 160, 32, 32), (8, 32, 32), "PLLaVA T=160→8"),
    ((2, 1024, 32, 48, 48), (16, 48, 48), "VideoLLaMA T=32→16"),
]

# --- Full 3D compression (all dims reduced) ---
FULL_3D_SHAPES = [
    # Path F: Large window kernel
    ((1, 16, 64, 128, 128), (4, 8, 8), "Path F: win=4913, total=4096"),
    ((1, 3, 16, 224, 224), (8, 7, 7), "Path F: video input"),
    # Path G: 1D kernel (large total, moderate window ≤ 2048)
    ((1, 1280, 48, 36, 50), (8, 8, 8), "Qwen2.5-VL, win=336→G"),
    ((2, 512, 64, 64, 64), (4, 8, 8), "medical 3D, win=1377→G"),
    # Path H: 2D fast (win > 2048)
    ((1, 8, 64, 256, 256), (64, 7, 7), "T keep, extreme H/W→H"),
    # Path I: 2D regular
    ((2, 64, 64, 256, 256), (2, 32, 32), "win=2673>2048, out_h=32→I"),
]

# --- Spatial compression (D preserved, H/W reduced) → mostly Path G ---
SPATIAL_COMPRESSION_SHAPES = [
    ((2, 128, 4, 28, 28), (4, 14, 14), "D keep, H/W 28→14→G"),
    ((1, 3, 32, 224, 224), (16, 56, 56), "video pyramid→G"),
]

# --- 1D kernel specific shapes ---
KERNEL_1D_SHAPES = [
    ((1, 2, 4, 4, 4), (2, 2, 2), "total=16→G (very small)"),
    ((4, 32, 16, 32, 32), (8, 16, 16), "total=262144, win=27→G"),
]

# --- 2D kernel via prefer_2d ---
PREFER_2D_SHAPES = [
    ((16, 4, 4, 4, 4), (2, 2, 2), "total=512, N*C=64→prefer_2d→H"),
    ((2, 1, 4, 4, 4), (2, 2, 2), "!prefer_2d→G (boundary)"),
]

# --- Edge cases: unit dims, non-divisible, stress ---
EDGE_CASE_SHAPES = [
    ((2, 256, 16, 1, 14), (8, 1, 14), "H=1"),
    ((2, 256, 16, 14, 1), (8, 14, 1), "W=1"),
    ((1, 64, 37, 59, 43), (7, 13, 19), "all prime in/out"),
    ((1, 3, 5, 13, 17), (2, 5, 7), "small odd dims"),
]

# --- Stress tests ---
STRESS_SHAPES = [
    ((128, 768, 4, 4, 4), (1, 1, 1), "huge batch→B"),
    ((1, 8192, 2, 2, 2), (1, 1, 1), "extreme C=8192→B"),
    ((1, 1, 128, 256, 256), (8, 8, 8), "C=1, large spatial"),
]

# --- Known model configs (video transformers) ---
MODEL_CONFIG_SHAPES = [
    ((1, 768, 96, 14, 14), (1, 1, 1), "TimeSformer 96f→B"),
    ((2, 256, 2, 14, 14), (1, 7, 7), "R3D-18 layer3"),
    ((1, 832, 16, 14, 14), (4, 7, 7), "I3D Mixed_4f"),
    ((1, 256, 16, 16, 16), (4, 4, 4), "3D U-Net bottleneck"),
]

# --- win_size boundary shapes (near 2048) ---
WIN_BOUNDARY_SHAPES = [
    ((2, 64, 16, 64, 64), (2, 8, 8), "win=729≤2048→G (below)"),
    ((2, 8, 64, 96, 96), (2, 4, 4), "win=20625>2048→H (above)"),
]

# --- Cubic output_size (int → (D,D,D)) ---
CUBIC_SHAPES = [
    ((1, 128, 32, 64, 64), (8, 8, 8), "cubic output"),
    ((1, 512, 64, 128, 128), (2, 2, 2), "cubic output, large win"),
]

# --- Aligned shapes (multiples of 8 for tensor-core) ---
ALIGNED_SHAPES = [
    ((2, 64, 16, 64, 64), (8, 16, 16), "aligned to 8"),
    ((1, 256, 32, 128, 128), (8, 16, 16), "aligned, large"),
]

# --- PLLaVA / InternVideo / Qwen additional variants ---
MORE_MODEL_SHAPES = [
    ((2, 1280, 64, 42, 72), (8, 14, 14), "Qwen2.5-VL scaled"),
    ((4, 1536, 96, 32, 32), (16, 16, 16), "VideoLLaMA large"),
    ((1, 32, 4, 128, 128), (2, 32, 32), "shallow, wide"),
]

# ============================================================================
# Assemble all configs into a flat list for parametrization
# ============================================================================
ALL_CONFIGS = (
    IDENTITY_SHAPES
    + PATH_B_SHAPES
    + PATH_C_SHAPES
    + PATH_D_SHAPES
    + PATH_E_SHAPES
    + TEMPORAL_COMPRESSION_SHAPES
    + FULL_3D_SHAPES
    + SPATIAL_COMPRESSION_SHAPES
    + KERNEL_1D_SHAPES
    + PREFER_2D_SHAPES
    + EDGE_CASE_SHAPES
    + STRESS_SHAPES
    + MODEL_CONFIG_SHAPES
    + WIN_BOUNDARY_SHAPES
    + CUBIC_SHAPES
    + ALIGNED_SHAPES
    + MORE_MODEL_SHAPES
)

# Deduplicate (just in case)
_seen = set()
UNIQUE_CONFIGS = []
for shape, out_size, desc in ALL_CONFIGS:
    key = (shape, out_size)
    if key not in _seen:
        _seen.add(key)
        UNIQUE_CONFIGS.append((shape, out_size, desc))
ALL_CONFIGS = UNIQUE_CONFIGS

# ============================================================================
# Parametrized tests using curated (shape, output_size) pairs
# ============================================================================


@pytest.mark.adaptive_max_pool3d
@pytest.mark.parametrize("shape, output_size, desc", ALL_CONFIGS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_adaptive_max_pool3d_forward(shape, output_size, desc, dtype):
    """Forward correctness with return_indices=True — all curated configs."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out, ref_indices = torch.nn.functional.adaptive_max_pool3d(
        ref_inp, output_size=output_size, return_indices=True
    )
    res_out, res_indices = torch.nn.functional.adaptive_max_pool3d(
        inp, output_size=output_size, return_indices=True
    )

    # Output values must match
    utils.gems_assert_close(res_out, ref_out, dtype)

    # Index consistency: values at indices must match output values
    gems_vals = inp.flatten(2)[
        torch.arange(inp.size(0), device=inp.device)[:, None, None, None, None],
        torch.arange(inp.size(1), device=inp.device)[None, :, None, None, None],
        res_indices,
    ]
    ref_vals = ref_inp.flatten(2)[
        torch.arange(ref_inp.size(0), device=ref_inp.device)[:, None, None, None, None],
        torch.arange(ref_inp.size(1), device=ref_inp.device)[None, :, None, None, None],
        ref_indices,
    ]
    assert torch.allclose(
        gems_vals.float(), res_out.float(), atol=0, rtol=0
    ), f"GEMS indices mismatch for {desc}"
    assert torch.allclose(
        ref_vals.float(), ref_out.float(), atol=0, rtol=0
    ), f"Reference indices mismatch for {desc}"


@pytest.mark.adaptive_max_pool3d
@pytest.mark.parametrize("shape, output_size, desc", ALL_CONFIGS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_adaptive_max_pool3d_forward_no_indices(
    shape, output_size, desc, dtype
):
    """Forward correctness with return_indices=False — all curated configs."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.nn.functional.adaptive_max_pool3d(
        ref_inp, output_size=output_size, return_indices=False
    )
    res_out = torch.nn.functional.adaptive_max_pool3d(
        inp, output_size=output_size, return_indices=False
    )

    assert isinstance(
        res_out, torch.Tensor
    ), f"Expected Tensor for return_indices=False, got {type(res_out)}. {desc}"
    utils.gems_assert_close(res_out, ref_out, dtype)


# ============================================================================
# Integer output_size (scalar → broadcast to cubic (D,D,D))
# ============================================================================

INT_OUTPUT_SIZE_CONFIGS = [
    ((1, 16, 8, 8, 8), 4, "int output: cubic 4"),
    ((2, 128, 32, 64, 64), 8, "int output: cubic 8"),
    ((1, 256, 16, 28, 28), 2, "int output: cubic 2"),
    ((2, 64, 6, 16, 32), 4, "int output: cubic 4, asymmetric input"),
    ((1, 8, 64, 256, 256), 7, "int output: cubic 7, extreme reduction"),
    ((1, 1, 5, 9, 11), 1, "int output: cubic 1 (global pool)"),
    ((2, 512, 1, 7, 7), 1, "int output: cubic 1, in_d=1"),
]


@pytest.mark.adaptive_max_pool3d
@pytest.mark.parametrize("shape, output_size, desc", INT_OUTPUT_SIZE_CONFIGS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_adaptive_max_pool3d_int_output_size(shape, output_size, desc, dtype):
    """Forward correctness with integer output_size → broadcast to (D,D,D)."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    # With return_indices=True
    ref_out, ref_indices = torch.nn.functional.adaptive_max_pool3d(
        ref_inp, output_size=output_size, return_indices=True
    )
    res_out, res_indices = torch.nn.functional.adaptive_max_pool3d(
        inp, output_size=output_size, return_indices=True
    )

    # Verify cubic output shape
    expected_out_size = (output_size, output_size, output_size)
    assert res_out.shape[2:] == torch.Size(
        expected_out_size
    ), f"Expected output shape {expected_out_size}, got {res_out.shape[2:]}. {desc}"

    utils.gems_assert_close(res_out, ref_out, dtype)

    # With return_indices=False
    ref_out = torch.nn.functional.adaptive_max_pool3d(
        ref_inp, output_size=output_size, return_indices=False
    )
    res_out = torch.nn.functional.adaptive_max_pool3d(
        inp, output_size=output_size, return_indices=False
    )
    assert isinstance(
        res_out, torch.Tensor
    ), f"Expected Tensor for int output_size with return_indices=False. {desc}"
    utils.gems_assert_close(res_out, ref_out, dtype)


# ============================================================================
# Special value tests: NaN, ties (equal max), all-negative, all-zero
# ============================================================================

# Use a subset of shapes that exercise each dispatch path
SPECIAL_VALUE_SHAPES = [
    ((2, 512, 1, 7, 7), (1, 1, 1), "Path E: global 1D"),
    ((1, 8, 64, 256, 256), (1, 7, 7), "Path B: out_d=1"),
    ((1, 16, 64, 128, 128), (4, 8, 8), "Path F: large window"),
    ((4, 128, 32, 64, 64), (8, 64, 64), "Path G: 1D kernel"),
    ((16, 4, 4, 4, 4), (2, 2, 2), "Path H: 2D fast"),
    ((2, 64, 64, 256, 256), (2, 32, 32), "Path I: 2D regular"),
    ((2, 256, 2, 14, 14), (2, 14, 14), "Path A: identity"),
]


@pytest.mark.adaptive_max_pool3d
@pytest.mark.parametrize("shape, output_size, desc", SPECIAL_VALUE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_adaptive_max_pool3d_nan(shape, output_size, desc, dtype):
    """NaN propagation: output must be NaN where input window contains NaN."""
    torch.manual_seed(42)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    # Inject NaN at a known position
    nan_pos = (0, 0, min(shape[2] - 1, 1), min(shape[3] - 1, 1), min(shape[4] - 1, 1))
    inp[nan_pos] = float("nan")

    ref_inp = utils.to_reference(inp, True)

    res_out, _ = torch.nn.functional.adaptive_max_pool3d(
        inp, output_size=output_size, return_indices=True
    )
    ref_out, _ = torch.nn.functional.adaptive_max_pool3d(
        ref_inp, output_size=output_size, return_indices=True
    )

    # NaN must be in the same positions
    res_nan = torch.isnan(res_out)
    ref_nan = torch.isnan(ref_out)
    res_nan_cmp = utils.to_cpu(res_nan, ref_nan)
    assert torch.equal(res_nan_cmp, ref_nan), f"NaN mask mismatch for {desc}"
    # Where not NaN, values must match
    utils.gems_assert_close(res_out[~res_nan], ref_out[~ref_nan], dtype)


@pytest.mark.adaptive_max_pool3d
@pytest.mark.parametrize("shape, output_size, desc", SPECIAL_VALUE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_adaptive_max_pool3d_ties(shape, output_size, desc, dtype):
    """Tie-breaking: when multiple elements share the max, indices may differ
    but the value must be correct."""
    torch.manual_seed(42)
    # All-equal input — every window is a tie
    inp = torch.ones(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    res_out, res_indices = torch.nn.functional.adaptive_max_pool3d(
        inp, output_size=output_size, return_indices=True
    )
    ref_out, _ = torch.nn.functional.adaptive_max_pool3d(
        ref_inp, output_size=output_size, return_indices=True
    )

    # All output values must be 1.0
    utils.gems_assert_close(res_out, ref_out, dtype)

    # Indices must be within valid range
    spatial_total = shape[2] * shape[3] * shape[4]
    assert (res_indices >= 0).all(), f"Negative indices in {desc}"
    assert (res_indices < spatial_total).all(), f"Out-of-range indices in {desc}"


@pytest.mark.adaptive_max_pool3d
@pytest.mark.parametrize("shape, output_size, desc", SPECIAL_VALUE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_adaptive_max_pool3d_all_negative(shape, output_size, desc, dtype):
    """All-negative values: max pooling should correctly find the largest
    (least negative) value."""
    torch.manual_seed(42)
    # Values in [-100, -1]
    inp = -torch.rand(shape, dtype=dtype, device=flag_gems.device) * 100 - 1
    ref_inp = utils.to_reference(inp, True)

    res_out, _ = torch.nn.functional.adaptive_max_pool3d(
        inp, output_size=output_size, return_indices=True
    )
    ref_out, _ = torch.nn.functional.adaptive_max_pool3d(
        ref_inp, output_size=output_size, return_indices=True
    )

    utils.gems_assert_close(res_out, ref_out, dtype)
    # All output values must be negative
    assert (res_out < 0).all(), f"Expected all negative output for {desc}"


@pytest.mark.adaptive_max_pool3d
@pytest.mark.parametrize("shape, output_size, desc", SPECIAL_VALUE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_adaptive_max_pool3d_mixed_sign(shape, output_size, desc, dtype):
    """Mixed positive/negative values: max pooling must pick the true max."""
    torch.manual_seed(42)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device) * 10
    ref_inp = utils.to_reference(inp, True)

    res_out, res_indices = torch.nn.functional.adaptive_max_pool3d(
        inp, output_size=output_size, return_indices=True
    )
    ref_out, ref_indices = torch.nn.functional.adaptive_max_pool3d(
        ref_inp, output_size=output_size, return_indices=True
    )

    utils.gems_assert_close(res_out, ref_out, dtype)

    # Verify index consistency
    gems_vals = inp.flatten(2)[
        torch.arange(inp.size(0), device=inp.device)[:, None, None, None, None],
        torch.arange(inp.size(1), device=inp.device)[None, :, None, None, None],
        res_indices,
    ]
    assert torch.allclose(
        gems_vals.float(), res_out.float(), atol=0, rtol=0
    ), f"Index-value mismatch for {desc}"


# ============================================================================
# Empty / zero-dim tensor handling
# ============================================================================

EMPTY_CONFIGS = [
    # output spatial dim = 0
    ((2, 64, 4, 8, 8), (0, 4, 4), "D_out=0"),
    ((2, 64, 4, 8, 8), (4, 0, 8), "H_out=0"),
    ((2, 64, 4, 8, 8), (4, 4, 0), "W_out=0"),
    # input spatial dim = 0 (edge — may not occur in practice but tests robustness)
]


@pytest.mark.adaptive_max_pool3d
@pytest.mark.parametrize("shape, output_size, desc", EMPTY_CONFIGS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.xfail(
    condition=(flag_gems.device == "npu"),
    reason="NPU Ascend operator aclnnAdaptiveMaxPool3d does not support output_size with dimension 0",
    strict=False,
)
def test_accuracy_adaptive_max_pool3d_empty_output(shape, output_size, desc, dtype):
    """Empty output tensor handling (zero in output_size)."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    # return_indices=True
    res_out, res_indices = torch.nn.functional.adaptive_max_pool3d(
        inp, output_size=output_size, return_indices=True
    )
    ref_out, ref_indices = torch.nn.functional.adaptive_max_pool3d(
        ref_inp, output_size=output_size, return_indices=True
    )
    assert res_out.numel() == 0, f"Expected empty output for {desc}"
    assert res_indices.numel() == 0, f"Expected empty indices for {desc}"
    assert res_out.shape == ref_out.shape, f"Shape mismatch for {desc}"

    # return_indices=False
    res_out = torch.nn.functional.adaptive_max_pool3d(
        inp, output_size=output_size, return_indices=False
    )
    ref_out = torch.nn.functional.adaptive_max_pool3d(
        ref_inp, output_size=output_size, return_indices=False
    )
    assert res_out.numel() == 0, f"Expected empty output for {desc}"
    assert isinstance(res_out, torch.Tensor), "Expected Tensor"


# ============================================================================
# Deterministic edge-case: all-zero input
# ============================================================================


@pytest.mark.adaptive_max_pool3d
@pytest.mark.parametrize("shape, output_size, desc", SPECIAL_VALUE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_adaptive_max_pool3d_all_zero(shape, output_size, desc, dtype):
    """All-zero input: output must be zero, indices valid."""
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    res_out, res_indices = torch.nn.functional.adaptive_max_pool3d(
        inp, output_size=output_size, return_indices=True
    )
    ref_out, _ = torch.nn.functional.adaptive_max_pool3d(
        ref_inp, output_size=output_size, return_indices=True
    )

    utils.gems_assert_close(res_out, ref_out, dtype)

    spatial_total = shape[2] * shape[3] * shape[4]
    assert (res_indices >= 0).all(), f"Negative indices in {desc}"
    assert (res_indices < spatial_total).all(), f"Out-of-range indices in {desc}"


# ============================================================================
# C++ wrapper correctness tests — §3.1 torch.ops.flag_gems.* & §3.3 c_operators
#
# These paths call the C++ TritonJITFunction wrapper directly, bypassing the
# Python use_gems() dispatch.  They verify that the C++ wrapper produces
# correct results on CUDA (and other backends where the C++ extension is built).
# ============================================================================

# Representative subset that covers all dispatch paths
CPP_WRAPPER_TEST_SHAPES = [
    # Path A: Identity
    ((2, 256, 2, 14, 14), (2, 14, 14)),
    # Path B: out_d=1
    ((1, 64, 8, 56, 56), (1, 56, 56)),
    # Path B → global
    ((1, 8, 32, 128, 128), (1, 1, 1)),
    # Path C/D/E: in_d=1, global
    ((2, 512, 1, 7, 7), (1, 1, 1)),
    # Temporal compression
    ((1, 768, 160, 32, 32), (8, 32, 32)),
    # Full 3D — Path F
    ((1, 16, 64, 128, 128), (4, 8, 8)),
    # Full 3D — Path G
    ((1, 1280, 48, 36, 50), (8, 8, 8)),
    # Full 3D — Path H
    ((1, 8, 64, 256, 256), (64, 7, 7)),
    # Full 3D — Path I
    ((2, 64, 64, 256, 256), (2, 32, 32)),
    # Spatial compression
    ((8, 64, 16, 56, 56), (16, 14, 14)),
    # Edge cases
    ((2, 256, 16, 1, 14), (8, 1, 14)),
    ((1, 64, 37, 59, 43), (7, 13, 19)),
    # Stress
    ((128, 768, 4, 4, 4), (1, 1, 1)),
    # Cubic int output
    ((1, 128, 32, 64, 64), (8, 8, 8)),
    # Small / fast shape
    ((2, 8, 8, 8, 8), (1, 4, 4)),
]


@pytest.mark.adaptive_max_pool3d
@pytest.mark.parametrize("shape, output_size", CPP_WRAPPER_TEST_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_adaptive_max_pool3d_cpp_c_operators(shape, output_size, dtype):
    """§3.3 c_operators pybind: bypass dispatcher, lowest-overhead C++ path."""
    try:
        from flag_gems import c_operators
    except ImportError:
        pytest.skip("C++ extension (c_operators) not built; rebuild with ENABLE_CPP=1")
    if not hasattr(c_operators, "adaptive_max_pool3d"):
        pytest.skip("adaptive_max_pool3d not in c_operators (rebuild C++ extension)")

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    # Reference: torch native
    ref_out = torch.nn.functional.adaptive_max_pool3d(
        ref_inp, output_size=output_size, return_indices=False
    )

    # c_operators path (returns single output tensor, no indices)
    cpp_out = c_operators.adaptive_max_pool3d(inp, list(output_size))

    assert isinstance(
        cpp_out, torch.Tensor
    ), f"c_operators must return a Tensor, got {type(cpp_out)}"
    assert (
        cpp_out.shape == ref_out.shape
    ), f"Shape mismatch: cpp={list(cpp_out.shape)} ref={list(ref_out.shape)}"
    utils.gems_assert_close(cpp_out, ref_out, dtype)


@pytest.mark.adaptive_max_pool3d
@pytest.mark.parametrize("shape, output_size", CPP_WRAPPER_TEST_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_adaptive_max_pool3d_cpp_flag_gems_ops(shape, output_size, dtype):
    """§3.1 torch.ops.flag_gems.*: custom op through PyTorch dispatcher."""
    try:
        torch.ops.flag_gems.adaptive_max_pool3d
    except (AttributeError, RuntimeError):
        pytest.skip(
            "torch.ops.flag_gems.adaptive_max_pool3d not registered (rebuild C++ extension)"
        )
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    # Reference: torch native
    ref_out, ref_indices = torch.nn.functional.adaptive_max_pool3d(
        ref_inp, output_size=output_size, return_indices=True
    )

    # torch.ops.flag_gems path (returns tuple of (output, indices))
    cpp_result = torch.ops.flag_gems.adaptive_max_pool3d(inp, list(output_size))
    cpp_out, cpp_indices = cpp_result

    assert isinstance(cpp_out, torch.Tensor), "output must be a Tensor"
    assert isinstance(cpp_indices, torch.Tensor), "indices must be a Tensor"
    assert (
        cpp_out.shape == ref_out.shape
    ), f"Output shape mismatch: cpp={list(cpp_out.shape)} ref={list(ref_out.shape)}"

    # Output values must match
    utils.gems_assert_close(cpp_out, ref_out, dtype)

    # Index consistency: gathered values at indices must match output values
    spatial_total = shape[2] * shape[3] * shape[4]
    assert (cpp_indices >= 0).all(), "Negative indices"
    assert (cpp_indices < spatial_total).all(), "Out-of-range indices"

    cpp_vals = inp.flatten(2)[
        torch.arange(inp.size(0), device=inp.device)[:, None, None, None, None],
        torch.arange(inp.size(1), device=inp.device)[None, :, None, None, None],
        cpp_indices,
    ]
    assert torch.allclose(
        cpp_vals.float(), cpp_out.float(), atol=0, rtol=0
    ), "Index-value mismatch in cpp torch.ops.flag_gems path"
