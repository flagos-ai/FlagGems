"""Tests for Radix Select (histogram-based) used in median fallback.

Verifies that _histogram_radix_select correctly finds K-th smallest
elements for various inputs and dtypes.
"""

import torch
import pytest

import flag_gems
from flag_gems.ops.median import _histogram_radix_select

CHUNK_SIZE = 16384


@pytest.mark.median
@pytest.mark.parametrize("shape", [
    (1, 65536),
    (4, 65536),
    (16, 65536),
    (1, 131072),
    (4, 262144),
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_radix_select_float(shape, dtype):
    M, N = shape
    inp = torch.randn(M, N, dtype=dtype, device="cuda")
    sorted_vals = torch.sort(inp, dim=-1).values

    for k_val in [0, N // 4, N // 2 - 1, N // 2, 3 * N // 4, N - 1]:
        out = _histogram_radix_select(inp, k_val)
        ref = sorted_vals[:, k_val]
        assert torch.allclose(out, ref, rtol=1e-3, atol=1e-3), (
            f"shape={shape}, dtype={dtype}, k={k_val}: "
            f"got {out[:4]}, expected {ref[:4]}"
        )


@pytest.mark.median
@pytest.mark.parametrize("shape", [
    (1, 65536),
    (4, 65536),
    (16, 65536),
    (1, 131072),
])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_radix_select_int(shape, dtype):
    M, N = shape
    inp = torch.randint(-10000, 10000, (M, N), dtype=dtype, device="cuda")
    sorted_vals = torch.sort(inp, dim=-1).values

    for k_val in [0, N // 4, N // 2 - 1, N // 2, 3 * N // 4, N - 1]:
        out = _histogram_radix_select(inp, k_val)
        ref = sorted_vals[:, k_val]
        assert torch.equal(out, ref), (
            f"shape={shape}, dtype={dtype}, k={k_val}: "
            f"got {out[:4]}, expected {ref[:4]}"
        )


@pytest.mark.median
def test_radix_select_with_negatives():
    M, N = 4, 65536
    inp = torch.randn(M, N, device="cuda") * 100 - 50
    sorted_vals = torch.sort(inp, dim=-1).values

    k_val = N // 2
    out = _histogram_radix_select(inp, k_val)
    ref = sorted_vals[:, k_val]
    assert torch.allclose(out, ref, rtol=1e-3, atol=1e-3)


@pytest.mark.median
def test_radix_select_with_duplicates():
    M, N = 4, 65536
    inp = torch.randint(0, 10, (M, N), dtype=torch.int32, device="cuda")
    sorted_vals = torch.sort(inp, dim=-1).values

    k_val = N // 2
    out = _histogram_radix_select(inp, k_val)
    ref = sorted_vals[:, k_val]
    assert torch.equal(out, ref), (
        f"got {out[:4]}, expected {ref[:4]}"
    )


@pytest.mark.median
def test_radix_select_int_negatives():
    M, N = 4, 65536
    inp = torch.randint(-100000, 100000, (M, N), dtype=torch.int32, device="cuda")
    sorted_vals = torch.sort(inp, dim=-1).values

    k_val = N // 2
    out = _histogram_radix_select(inp, k_val)
    ref = sorted_vals[:, k_val]
    assert torch.equal(out, ref), (
        f"got {out[:4]}, expected {ref[:4]}"
    )


@pytest.mark.median
def test_radix_select_int64_large():
    M, N = 4, 65536
    inp = torch.randint(
        -1000000000, 1000000000, (M, N), dtype=torch.int64, device="cuda"
    )
    sorted_vals = torch.sort(inp, dim=-1).values

    k_val = N // 2
    out = _histogram_radix_select(inp, k_val)
    ref = sorted_vals[:, k_val]
    assert torch.equal(out, ref), (
        f"got {out[:4]}, expected {ref[:4]}"
    )


@pytest.mark.median
def test_radix_select_all_same():
    M, N = 4, 65536
    inp = torch.full((M, N), 42.0, device="cuda")

    k_val = N // 2
    out = _histogram_radix_select(inp, k_val)
    ref = torch.full((M,), 42.0, device="cuda")
    assert torch.allclose(out, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.median
def test_radix_select_single_element():
    M, N = 4, 1
    inp = torch.randn(M, N, device="cuda")

    k_val = 0
    out = _histogram_radix_select(inp, k_val)
    ref = inp.squeeze().float()
    assert torch.allclose(out, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.median
@pytest.mark.parametrize("shape", [
    (1024, 65536),
    (16, 131072),
    (8, 262144),
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_median_fallback_float(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    ref_values, ref_indices = torch.median(inp, dim=-1)

    with flag_gems.use_gems():
        res_values, res_indices = torch.median(inp, dim=-1)

    assert torch.allclose(res_values.float(), ref_values.float(), rtol=1e-3, atol=1e-3), (
        f"Values mismatch for shape={shape}, dtype={dtype}"
    )
    # Verify index points to correct value
    for i in range(min(4, res_values.shape[0])):
        idx = res_indices[i].item()
        actual = inp[i, idx].float().item()
        expected = ref_values[i].float().item()
        assert abs(actual - expected) < 1e-3, (
            f"Index {idx} points to {actual}, expected {expected}"
        )


@pytest.mark.median
@pytest.mark.parametrize("shape", [
    (1024, 65536),
    (16, 131072),
])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_median_fallback_int(shape, dtype):
    inp = torch.randint(-10000, 10000, shape, dtype=dtype, device="cuda")
    ref_values, ref_indices = torch.median(inp, dim=-1)

    with flag_gems.use_gems():
        res_values, res_indices = torch.median(inp, dim=-1)

    assert torch.equal(res_values, ref_values), (
        f"Values mismatch for shape={shape}, dtype={dtype}"
    )


@pytest.mark.median
def test_median_fallback_with_duplicates():
    inp = torch.randint(0, 5, (1024, 65536), dtype=torch.float32, device="cuda")
    ref_values, ref_indices = torch.median(inp, dim=-1)

    with flag_gems.use_gems():
        res_values, res_indices = torch.median(inp, dim=-1)

    assert torch.allclose(res_values, ref_values, rtol=1e-5, atol=1e-5)


@pytest.mark.median
def test_median_fallback_negatives():
    inp = torch.randn(1024, 65536, device="cuda") * 100 - 50
    ref_values, ref_indices = torch.median(inp, dim=-1)

    with flag_gems.use_gems():
        res_values, res_indices = torch.median(inp, dim=-1)

    assert torch.allclose(res_values, ref_values, rtol=1e-3, atol=1e-3)
