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


@pytest.mark.geometric_
@pytest.mark.parametrize("shape", utils.DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_geometric_(shape, dtype):
    p = 0.5
    x = torch.empty(size=shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        x.geometric_(p)

    # Check that all values are positive integers (>= 1)
    positive_mask = (x >= 1).float().to(dtype)
    ref_ones = utils.to_reference(torch.ones_like(x))
    utils.gems_assert_equal(positive_mask, ref_ones)

    # Check that the mean is approximately 1/p
    mean = x.float().mean()
    expected = torch.tensor(1.0 / p, dtype=torch.float32, device=flag_gems.device)
    expected = utils.to_reference(expected)
    utils.gems_assert_close(mean, expected, dtype=torch.float32, atol=0.2)


@pytest.mark.geometric_
@pytest.mark.parametrize("shape", utils.DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("p", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_geometric_various_p(shape, dtype, p):
    x = torch.empty(size=shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        x.geometric_(p)

    # Check that all values are positive integers (>= 1)
    positive_mask = (x >= 1).float().to(dtype)
    ref_ones = utils.to_reference(torch.ones_like(x))
    utils.gems_assert_equal(positive_mask, ref_ones)

    # Check that the mean is approximately 1/p
    mean = x.float().mean()
    expected = torch.tensor(1.0 / p, dtype=torch.float32, device=flag_gems.device)
    expected = utils.to_reference(expected)
    utils.gems_assert_close(mean, expected, dtype=torch.float32, atol=0.3)


@pytest.mark.geometric
@pytest.mark.parametrize("shape", utils.DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_geometric(shape, dtype):
    p = 0.5
    x = torch.empty(size=shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        y = torch.ops.aten.geometric(x, p)

    # Check that the output is a new tensor
    assert y is not x

    # Check that all values are positive integers (>= 1)
    positive_mask = (y >= 1).float().to(dtype)
    ref_ones = utils.to_reference(torch.ones_like(y))
    utils.gems_assert_equal(positive_mask, ref_ones)

    # Check that the mean is approximately 1/p
    mean = y.float().mean()
    expected = torch.tensor(1.0 / p, dtype=torch.float32, device=flag_gems.device)
    expected = utils.to_reference(expected)
    utils.gems_assert_close(mean, expected, dtype=torch.float32, atol=0.2)


def _check_backend_distribution(inplace, dtype, p):
    # Include a masked tail and enough samples to check both mean and variance.
    x = torch.empty((262147,), device=flag_gems.device, dtype=dtype)
    generator = torch.Generator(device=x.device).manual_seed(12345)
    with flag_gems.use_gems():
        if inplace:
            y = x.geometric_(p, generator=generator)
        else:
            y = torch.ops.aten.geometric(x, p, generator=generator)
    values = y.cpu().float()
    assert torch.isfinite(values).all()
    assert (values >= 1).all()
    assert torch.equal(values, values.floor())
    expected_variance = (1 - p) / p**2
    assert (
        abs(values.mean().item() - 1 / p)
        < 8 * (expected_variance / values.numel()) ** 0.5
    )
    assert abs(values.var().item() / expected_variance - 1) < 0.12


def _check_backend_layout_and_generator(inplace, layout):
    base = torch.full((33, 62), -7.0, device=flag_gems.device)
    x = {
        "transpose": base.T,
        "slice": base[:, 1::2],
        "scalar": base[0, 0],
        "empty": base[:0],
    }[layout]
    gen = torch.Generator(device=x.device).manual_seed(712)
    before = gen.get_state().clone()

    def sample():
        with flag_gems.use_gems():
            return (
                x.geometric_(0.3, generator=gen)
                if inplace
                else torch.ops.aten.geometric(x, 0.3, generator=gen)
            )

    y = sample()
    saved = y.cpu().clone()
    after = gen.get_state().clone()
    gen.set_state(before)
    z = sample()
    assert torch.equal(z.cpu(), saved)
    assert y.shape == x.shape
    assert (y is x) == inplace
    if x.numel():
        assert not torch.equal(before, after)
        assert torch.isfinite(saved).all() and (saved >= 1).all()
    else:
        assert torch.equal(before, after)
    if inplace and layout == "slice":
        assert (base[:, ::2].cpu() == -7).all()
    if not inplace:
        assert (base.cpu() == -7).all()


def _check_backend_invalid_p(inplace, p):
    x = torch.empty((17,), device=flag_gems.device)
    with flag_gems.use_gems(), pytest.raises(RuntimeError, match="p"):
        if inplace:
            x.geometric_(p)
        else:
            torch.ops.aten.geometric(x, p)


@pytest.mark.geometric
@pytest.mark.skipif(
    flag_gems.vendor_name not in ("ascend", "mthreads"), reason="Backend regression"
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("p", [0.01, 0.5, 0.99])
def test_geometric_backend_distribution(dtype, p):
    _check_backend_distribution(False, dtype, p)


@pytest.mark.geometric_
@pytest.mark.skipif(
    flag_gems.vendor_name not in ("ascend", "mthreads"), reason="Backend regression"
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("p", [0.01, 0.5, 0.99])
def test_geometric__backend_distribution(dtype, p):
    _check_backend_distribution(True, dtype, p)


@pytest.mark.geometric
@pytest.mark.skipif(
    flag_gems.vendor_name not in ("ascend", "mthreads"), reason="Backend regression"
)
@pytest.mark.parametrize("layout", ["transpose", "slice", "scalar", "empty"])
def test_geometric_backend_layout_and_generator(layout):
    _check_backend_layout_and_generator(False, layout)


@pytest.mark.geometric_
@pytest.mark.skipif(
    flag_gems.vendor_name not in ("ascend", "mthreads"), reason="Backend regression"
)
@pytest.mark.parametrize("layout", ["transpose", "slice", "scalar", "empty"])
def test_geometric__backend_layout_and_generator(layout):
    _check_backend_layout_and_generator(True, layout)


@pytest.mark.geometric
@pytest.mark.skipif(
    flag_gems.vendor_name not in ("ascend", "mthreads"), reason="Backend regression"
)
@pytest.mark.parametrize("p", [0.0, -0.1, 1.0, 1.1, float("nan")])
def test_geometric_backend_invalid_p(p):
    _check_backend_invalid_p(False, p)


@pytest.mark.geometric_
@pytest.mark.skipif(
    flag_gems.vendor_name not in ("ascend", "mthreads"), reason="Backend regression"
)
@pytest.mark.parametrize("p", [0.0, -0.1, 1.0, 1.1, float("nan")])
def test_geometric__backend_invalid_p(p):
    _check_backend_invalid_p(True, p)
