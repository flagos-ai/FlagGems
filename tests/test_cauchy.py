import numpy as np
import pytest
import torch

import flag_gems

from .accuracy_utils import gems_assert_equal, to_reference


@pytest.mark.parametrize("shape", [(1024,), (256, 256)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("median", [0.0, 1.0, -0.5])
@pytest.mark.parametrize("sigma", [1.0, 0.5, 2.0])
def test_cauchy_accuracy(shape, dtype, median, sigma):
    """
    Test that cauchy_ produces samples that follow the expected Cauchy distribution.
    We use statistical tests to verify the distribution properties.
    """
    torch.manual_seed(42)
    x = torch.empty(shape, dtype=dtype, device="cuda")
    ref_x = to_reference(x)

    with flag_gems.use_gems():
        x.cauchy_(median=median, sigma=sigma)

    ref_x.cauchy_(median=median, sigma=sigma)

    # For Cauchy distribution, we can't use standard mean/variance tests
    # because they are undefined. Instead, we check:
    # 1. The samples are in a reasonable range (not too far from median)
    # 2. The distribution is symmetric around the median
    # 3. The samples match the reference distribution

    # Check that distributions are similar using percentiles
    # (Cauchy has heavy tails, so we use robust statistics)
    x_np = x.cpu().numpy().flatten()
    ref_np = ref_x.cpu().numpy().flatten()

    # Check symmetry: median of (x - median) should be close to 0
    x_centered = x_np - median
    ref_centered = ref_np - median

    x_median = np.median(x_centered)
    ref_median = np.median(ref_centered)

    # Median should be close to 0 (Cauchy is symmetric)
    assert abs(x_median) < 0.1 * sigma, f"Median {x_median} too far from 0"
    assert abs(ref_median) < 0.1 * sigma, f"Ref median {ref_median} too far from 0"

    # Check interquartile range (IQR) which is 2*sigma for Cauchy
    x_iqr = np.percentile(x_centered, 75) - np.percentile(x_centered, 25)
    ref_iqr = np.percentile(ref_centered, 75) - np.percentile(ref_centered, 25)

    # IQR should be approximately 2*sigma
    expected_iqr = 2 * sigma
    assert (
        0.5 * expected_iqr < x_iqr < 2 * expected_iqr
    ), f"IQR {x_iqr} not in expected range for sigma={sigma}"

    # Compare with reference
    assert abs(x_iqr - ref_iqr) < 0.5 * sigma, f"IQR mismatch: {x_iqr} vs {ref_iqr}"


@pytest.mark.parametrize("shape", [(1024,), (256, 256)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("median", [0.0, 1.0, -0.5])
@pytest.mark.parametrize("sigma", [1.0, 0.5, 2.0])
def test_cauchy_out_accuracy(shape, dtype, median, sigma):
    """
    Test the out-of-place cauchy function.
    """
    torch.manual_seed(42)
    x = torch.empty(shape, dtype=dtype, device="cuda")
    ref_x = to_reference(x)

    with flag_gems.use_gems():
        result = torch.ops.aten.cauchy(x, median=median, sigma=sigma)

    ref_result = torch.ops.aten.cauchy(ref_x, median=median, sigma=sigma)

    # Same statistical checks as test_cauchy_accuracy
    result_np = result.cpu().numpy().flatten()
    ref_np = ref_result.cpu().numpy().flatten()

    result_centered = result_np - median
    ref_centered = ref_np - median

    result_median = np.median(result_centered)
    ref_median_val = np.median(ref_centered)

    assert abs(result_median) < 0.1 * sigma
    assert abs(ref_median_val) < 0.1 * sigma

    result_iqr = np.percentile(result_centered, 75) - np.percentile(result_centered, 25)
    ref_iqr = np.percentile(ref_centered, 75) - np.percentile(ref_centered, 25)

    expected_iqr = 2 * sigma
    assert 0.5 * expected_iqr < result_iqr < 2 * expected_iqr
    assert abs(result_iqr - ref_iqr) < 0.5 * sigma


@pytest.mark.parametrize("shape", [(1024,), (256, 256)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cauchy_reproducibility(shape, dtype):
    """
    Test that cauchy_ produces reproducible results with the same seed.
    """
    torch.manual_seed(12345)
    x1 = torch.empty(shape, dtype=dtype, device="cuda")

    with flag_gems.use_gems():
        x1.cauchy_(median=0.0, sigma=1.0)

    torch.manual_seed(12345)
    x2 = torch.empty(shape, dtype=dtype, device="cuda")

    with flag_gems.use_gems():
        x2.cauchy_(median=0.0, sigma=1.0)

    # With the same seed, results should be identical
    gems_assert_equal(x1, x2)


@pytest.mark.parametrize("shape", [(1024, 1024), (512, 512, 4)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_cauchy_large_tensors(shape, dtype):
    """
    Test cauchy_ on larger tensors to ensure it works at scale.
    """
    torch.manual_seed(42)
    x = torch.empty(shape, dtype=dtype, device="cuda")

    with flag_gems.use_gems():
        x.cauchy_(median=0.0, sigma=1.0)

    # Check median is reasonable
    x_np = x.cpu().numpy().flatten()
    x_median = np.median(x_np)
    assert abs(x_median) < 0.1, f"Median {x_median} too far from 0"
