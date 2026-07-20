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

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Kernel uses generic int bitcast based on input dtype bitwidth;
# supports fp16, bf16, fp32, and fp64.
NEXTAFTER_DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64]


@pytest.mark.nextafter_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", NEXTAFTER_DTYPES)
def test_nextafter_(shape, dtype):
    # Test nextafter_: returns the next representable value from x toward y
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    y = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x).clone()
    ref_y = utils.to_reference(y)
    ref_x.nextafter_(ref_y)

    with flag_gems.use_gems():
        x_clone = x.clone()
        x_clone.nextafter_(y)
        utils.gems_assert_close(x_clone, ref_x, dtype)


# --- Boundary tests for nextafter_ ---


@pytest.mark.nextafter_
@pytest.mark.parametrize("dtype", NEXTAFTER_DTYPES)
def test_nextafter_zero_boundary(dtype):
    """Test +0/-0 crossing: +0 toward -inf => -0, -0 toward +inf => min pos subnormal."""
    # +0.0 toward -1.0 should produce -0.0
    x_pos_zero = torch.zeros(10, dtype=dtype, device=flag_gems.device)
    y_neg = torch.full((10,), -1.0, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x_pos_zero).clone()
    ref_y = utils.to_reference(y_neg)
    ref_x.nextafter_(ref_y)

    with flag_gems.use_gems():
        imp_x = x_pos_zero.clone()
        imp_x.nextafter_(y_neg)
        # +0 becomes -0; check by comparing to reference
        utils.gems_assert_close(imp_x, ref_x, dtype)

    # -0.0 toward +1.0 should produce the smallest positive subnormal (denormal) number
    x_neg_zero = -torch.zeros(10, dtype=dtype, device=flag_gems.device)
    y_pos = torch.full((10,), 1.0, dtype=dtype, device=flag_gems.device)

    ref_x2 = utils.to_reference(x_neg_zero).clone()
    ref_y2 = utils.to_reference(y_pos)
    ref_x2.nextafter_(ref_y2)

    with flag_gems.use_gems():
        imp_x2 = x_neg_zero.clone()
        imp_x2.nextafter_(y_pos)
        utils.gems_assert_close(imp_x2, ref_x2, dtype)


@pytest.mark.nextafter_
@pytest.mark.parametrize("dtype", NEXTAFTER_DTYPES)
def test_nextafter_nan(dtype):
    """Test nextafter_ with NaN inputs: IEEE 754 requires NaN propagation."""
    # nextafter(NaN, 1.0) => NaN
    x_nan = torch.tensor(
        [float("nan"), 1.0, 2.0, float("nan")], dtype=dtype, device=flag_gems.device
    )
    y = torch.tensor(
        [3.0, float("nan"), 4.0, 5.0], dtype=dtype, device=flag_gems.device
    )

    ref_x = utils.to_reference(x_nan).clone()
    ref_y = utils.to_reference(y)
    ref_x.nextafter_(ref_y)
    # PyTorch: nextafter(NaN, any) = NaN, nextafter(any, NaN) = NaN
    ref_nan_mask = torch.isnan(ref_x)

    with flag_gems.use_gems():
        imp_x = x_nan.clone()
        imp_x.nextafter_(y)
        imp_nan_mask = torch.isnan(imp_x)

    assert torch.equal(
        ref_nan_mask.cpu(), imp_nan_mask.cpu()
    ), f"NaN mask mismatch: ref_nan at {ref_nan_mask}, imp_nan at {imp_nan_mask}"
    # Non-NaN values should match; use cpu mask for cross-device compatibility
    non_nan_mask = ~(imp_nan_mask.cpu())
    if non_nan_mask.any():
        utils.gems_assert_close(
            imp_x[non_nan_mask.to(imp_x.device)],
            ref_x[non_nan_mask.to(ref_x.device)],
            dtype,
        )


@pytest.mark.nextafter_
@pytest.mark.parametrize("dtype", NEXTAFTER_DTYPES)
def test_nextafter_inf(dtype):
    """Test nextafter_ with Inf inputs."""
    x = torch.tensor(
        [1.0, float("inf"), float("-inf"), 0.0], dtype=dtype, device=flag_gems.device
    )
    y = torch.tensor(
        [2.0, float("inf"), float("-inf"), float("inf")],
        dtype=dtype,
        device=flag_gems.device,
    )

    ref_x = utils.to_reference(x).clone()
    ref_y = utils.to_reference(y)
    ref_x.nextafter_(ref_y)

    with flag_gems.use_gems():
        imp_x = x.clone()
        imp_x.nextafter_(y)
        utils.gems_assert_close(imp_x, ref_x, dtype)


@pytest.mark.nextafter_
def test_nextafter_finfo_extremes():
    """Test nextafter_ with torch.finfo extremes (max, min normal, eps)."""
    x = torch.tensor(
        [
            0.0,
            torch.finfo(torch.float32).max,
            -torch.finfo(torch.float32).max,
            torch.finfo(torch.float32).tiny,
            -torch.finfo(torch.float32).tiny,
            torch.finfo(torch.float32).eps,
        ],
        dtype=torch.float32,
        device=flag_gems.device,
    )
    y = torch.tensor(
        [1.0, float("inf"), float("-inf"), 0.0, 0.0, 0.0],
        dtype=torch.float32,
        device=flag_gems.device,
    )

    ref_x = utils.to_reference(x).clone()
    ref_y = utils.to_reference(y)
    ref_x.nextafter_(ref_y)

    with flag_gems.use_gems():
        imp_x = x.clone()
        imp_x.nextafter_(y)
        utils.gems_assert_close(imp_x, ref_x, torch.float32)


# --- Scalar input tests for nextafter_ ---


@pytest.mark.nextafter_
@pytest.mark.parametrize("dtype", NEXTAFTER_DTYPES)
def test_nextafter_scalar_y(dtype):
    """Test nextafter_ with tensor x and 0-dim tensor y (tests tensor-scalar kernel path)."""
    x = torch.randn(16, dtype=dtype, device=flag_gems.device)
    scalar_y = torch.tensor(0.5, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x).clone()
    ref_y = utils.to_reference(scalar_y)
    ref_x.nextafter_(ref_y)

    with flag_gems.use_gems():
        imp_x = x.clone()
        imp_x.nextafter_(scalar_y)
        utils.gems_assert_close(imp_x, ref_x, dtype)


@pytest.mark.nextafter_
@pytest.mark.parametrize("dtype", NEXTAFTER_DTYPES)
def test_nextafter_scalar_x(dtype):
    """Test nextafter_ with scalar x: this goes through Python numpy path."""
    # When x is a scalar Python float, nextafter_ falls back to numpy.nextafter
    # and returns a tensor (consistent with torch op signature)
    scalar_x = 0.5
    y = torch.full((16,), 1.0, dtype=dtype, device=flag_gems.device)

    ref_y = utils.to_reference(y)
    ref_x = torch.full_like(ref_y, scalar_x)
    ref_x.nextafter_(ref_y)

    # Our implementation: scalar x + tensor y -> kernel returns new tensor,
    # but nextafter_ should be in-place.  Test the kernel path.
    with flag_gems.use_gems():
        imp_x = torch.tensor(scalar_x, dtype=dtype, device=flag_gems.device)
        imp_x = imp_x.expand_as(y).contiguous().clone()
        imp_x.nextafter_(y)

    utils.gems_assert_close(imp_x, ref_x, dtype)


@pytest.mark.nextafter_
@pytest.mark.parametrize("dtype", NEXTAFTER_DTYPES)
def test_nextafter_scalar_nan_y(dtype):
    """Test nextafter_ with tensor x and NaN 0-dim tensor y."""
    x = torch.ones(8, dtype=dtype, device=flag_gems.device)
    scalar_y = torch.tensor(float("nan"), dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x).clone()
    ref_y = utils.to_reference(scalar_y)
    ref_x.nextafter_(ref_y)
    # PyTorch: nextafter(any, NaN) = NaN
    assert torch.isnan(ref_x).all(), "Reference should be all NaN"

    with flag_gems.use_gems():
        imp_x = x.clone()
        imp_x.nextafter_(scalar_y)
        assert torch.isnan(imp_x).all(), f"Should be all NaN, got {imp_x}"


@pytest.mark.nextafter_
@pytest.mark.parametrize("dtype", NEXTAFTER_DTYPES)
def test_nextafter_scalar_inf_y(dtype):
    """Test nextafter_ with tensor x and Inf 0-dim tensor y."""
    x = torch.ones(8, dtype=dtype, device=flag_gems.device)
    scalar_y = torch.tensor(float("inf"), dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x).clone()
    ref_y = utils.to_reference(scalar_y)
    ref_x.nextafter_(ref_y)

    with flag_gems.use_gems():
        imp_x = x.clone()
        imp_x.nextafter_(scalar_y)
        utils.gems_assert_close(imp_x, ref_x, dtype)
