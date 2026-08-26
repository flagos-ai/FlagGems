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

import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

_FLOAT_DTYPES = [torch.float32]
if flag_gems.runtime.device.support_fp64:
    _FLOAT_DTYPES.append(torch.float64)


def _native_reference(x, n):
    ref_x = utils.to_reference(x) if isinstance(x, torch.Tensor) else x
    ref_n = utils.to_reference(n) if isinstance(n, torch.Tensor) else n
    return torch.special.laguerre_polynomial_l(ref_x, ref_n)


def _scalar_reference(x, n):
    n = int(n)
    if n < 0:
        return 0.0
    if x == 0.0 or n == 0:
        return 1.0
    if n == 1:
        return 1.0 - x

    p = 1.0
    q = 1.0 - x
    result = q
    k = 1
    while k < n and not math.isnan(q):
        result = ((2.0 * k + 1.0 - x) * q - k * p) / (k + 1.0)
        p = q
        q = result
        k += 1
    return result


@pytest.mark.special_laguerre_polynomial_l
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test_special_laguerre_polynomial_l_tensor_tensor_broadcast(dtype):
    x = torch.linspace(-0.75, 0.75, 15, dtype=dtype, device=flag_gems.device).reshape(
        3, 1, 5
    )
    n = torch.tensor(
        [0.0, 1.9, 2.1, 7.9], dtype=dtype, device=flag_gems.device
    ).reshape(1, 4, 1)

    ref = _native_reference(x, n)
    with flag_gems.use_gems():
        actual = torch.special.laguerre_polynomial_l(x, n)

    assert actual.shape == (3, 4, 5)
    utils.gems_assert_close(actual, ref, dtype, equal_nan=True)


@pytest.mark.special_laguerre_polynomial_l
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test_special_laguerre_polynomial_l_fast_degrees(dtype):
    x = torch.tensor(
        [-0.5, 0.25, 0.75, 1.25, -1.0, 0.5, 0.125],
        dtype=dtype,
        device=flag_gems.device,
    )
    n = torch.tensor(
        [-3.8, -0.9, 0.9, 1.9, 2.9, 3.9, 4.9],
        dtype=dtype,
        device=flag_gems.device,
    )

    ref = _native_reference(x, n)
    with flag_gems.use_gems():
        actual = torch.special.laguerre_polynomial_l(x, n)

    utils.gems_assert_close(actual, ref, dtype, equal_nan=True)


@pytest.mark.special_laguerre_polynomial_l
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test_special_laguerre_polynomial_l_special_x(dtype):
    x_values = [
        float("-inf"),
        float("-inf"),
        float("inf"),
        float("inf"),
        float("nan"),
        float("nan"),
        -0.0,
        0.0,
    ]
    n_values = [2, 3, 2, 3, 0, 2, 2, -1]
    x = torch.tensor(x_values, dtype=dtype, device=flag_gems.device)
    n = torch.tensor(n_values, dtype=torch.int64, device=flag_gems.device)
    expected = torch.tensor(
        [_scalar_reference(a, b) for a, b in zip(x_values, n_values)],
        dtype=dtype,
        device=flag_gems.device,
    )

    with flag_gems.use_gems():
        actual = torch.special.laguerre_polynomial_l(x, n)

    torch.testing.assert_close(
        actual.cpu(), expected.cpu(), rtol=0, atol=0, equal_nan=True
    )


@pytest.mark.special_laguerre_polynomial_l
def test_special_laguerre_polynomial_l_promotion_and_nonfinite_degree():
    x = torch.tensor([2, 0, 3, 0], dtype=torch.int32, device=flag_gems.device)
    n = torch.tensor(
        [True, False, True, True], dtype=torch.bool, device=flag_gems.device
    )
    with flag_gems.use_gems():
        promoted = torch.special.laguerre_polynomial_l(x, n)

    assert promoted.dtype == torch.get_default_dtype()
    expected = torch.tensor([-1.0, 1.0, -2.0, 1.0], device=flag_gems.device)
    torch.testing.assert_close(promoted.cpu(), expected.cpu(), rtol=0, atol=0)

    x = torch.tensor([0.25, 0.25, 0.0, 0.25, 0.25], device=flag_gems.device)
    n = torch.tensor(
        [float("nan"), float("-inf"), float("inf"), 1e30, -1e30],
        device=flag_gems.device,
    )
    with flag_gems.use_gems():
        actual = torch.special.laguerre_polynomial_l(x, n)
    expected = torch.zeros(5, device=flag_gems.device)
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)


@pytest.mark.special_laguerre_polynomial_l
def test_special_laguerre_polynomial_l_large_degree_fast_exit_and_recurrence():
    x = torch.tensor([0.0, 0.0, 0.5], device=flag_gems.device)
    n = torch.tensor([2**31, 2**40, -(2**40)], device=flag_gems.device)
    with flag_gems.use_gems():
        actual = torch.special.laguerre_polynomial_l(x, n)
    expected = torch.tensor([1.0, 1.0, 0.0])
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

    # A non-trivial large degree exercises the runtime loop without approaching
    # a device watchdog threshold.
    x = torch.tensor([0.125], device=flag_gems.device)
    n = torch.tensor([1024], device=flag_gems.device)
    ref = torch.special.laguerre_polynomial_l(x.cpu(), n.cpu())
    with flag_gems.use_gems():
        actual = torch.special.laguerre_polynomial_l(x, n)
    torch.testing.assert_close(actual.cpu(), ref, rtol=2e-4, atol=2e-5)


@pytest.mark.special_laguerre_polynomial_l
def test_special_laguerre_polynomial_l_is_not_differentiable():
    x = torch.tensor([0.25, 0.5], device=flag_gems.device, requires_grad=True)
    n = torch.tensor([2.0, 3.0], device=flag_gems.device, requires_grad=True)
    with flag_gems.use_gems():
        actual = torch.special.laguerre_polynomial_l(x, n)
    assert not actual.requires_grad


@pytest.mark.special_laguerre_polynomial_l_n_scalar
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test_special_laguerre_polynomial_l_n_scalar(dtype):
    x = torch.linspace(-0.75, 0.75, 33, dtype=dtype, device=flag_gems.device)
    n = 6.9
    ref = _native_reference(x, n)
    with flag_gems.use_gems():
        actual = torch.special.laguerre_polynomial_l(x, n)
    utils.gems_assert_close(actual, ref, dtype, equal_nan=True)


@pytest.mark.special_laguerre_polynomial_l_x_scalar
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test_special_laguerre_polynomial_l_x_scalar(dtype):
    x = -0.25
    n = torch.tensor(
        [[-1.2], [0.2], [1.2], [2.2], [8.2]],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref = _native_reference(x, n)
    with flag_gems.use_gems():
        actual = torch.special.laguerre_polynomial_l(x, n)
    utils.gems_assert_close(actual, ref, dtype, equal_nan=True)


@pytest.mark.special_laguerre_polynomial_l_n_scalar
@pytest.mark.special_laguerre_polynomial_l_x_scalar_out
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test_special_laguerre_polynomial_l_scalar_noncontiguous_layouts(dtype):
    tensor = torch.linspace(
        -0.75, 0.75, 60, dtype=dtype, device=flag_gems.device
    ).reshape(4, 3, 5)
    x = tensor.transpose(0, 1)
    n = 5.8
    expected_n_scalar = _native_reference(x, n)
    with flag_gems.use_gems():
        actual_n_scalar = torch.special.laguerre_polynomial_l(x, n)
    utils.gems_assert_close(actual_n_scalar, expected_n_scalar, x.dtype, equal_nan=True)

    degree = (tensor.abs() * 8.0).transpose(0, 1)
    expected_x_scalar = _native_reference(0.25, degree)
    storage = torch.empty((4, 3, 5), dtype=dtype, device=flag_gems.device)
    out = storage.transpose(0, 1)
    with flag_gems.use_gems():
        actual_x_scalar = torch.special.laguerre_polynomial_l(0.25, degree, out=out)
    assert actual_x_scalar.data_ptr() == out.data_ptr()
    assert not actual_x_scalar.is_contiguous()
    utils.gems_assert_close(
        actual_x_scalar, expected_x_scalar, degree.dtype, equal_nan=True
    )


@pytest.mark.special_laguerre_polynomial_l_n_scalar
@pytest.mark.special_laguerre_polynomial_l_x_scalar
@pytest.mark.skipif(
    not flag_gems.runtime.device.support_fp64,
    reason="float64 is not supported on this backend",
)
def test_special_laguerre_polynomial_l_scalar_float64_precision():
    x = torch.tensor([0.125], dtype=torch.float64, device=flag_gems.device)
    n = 2.9999999999999996
    expected_n_scalar = _native_reference(x, n)
    with flag_gems.use_gems():
        actual_n_scalar = torch.special.laguerre_polynomial_l(x, n)
    torch.testing.assert_close(
        actual_n_scalar.cpu(), expected_n_scalar.cpu(), rtol=0, atol=0
    )

    scalar_x = 0.123456789012345
    degree = torch.tensor([2.0], dtype=torch.float64, device=flag_gems.device)
    expected_x_scalar = _native_reference(scalar_x, degree)
    with flag_gems.use_gems():
        actual_x_scalar = torch.special.laguerre_polynomial_l(scalar_x, degree)
    torch.testing.assert_close(
        actual_x_scalar.cpu(), expected_x_scalar.cpu(), rtol=1e-15, atol=1e-15
    )


@pytest.mark.special_laguerre_polynomial_l_out
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test_special_laguerre_polynomial_l_out(dtype):
    x = torch.linspace(-0.5, 0.5, 15, dtype=dtype, device=flag_gems.device).reshape(
        3, 1, 5
    )
    n = torch.tensor([0, 1, 2, 6], device=flag_gems.device).reshape(1, 4, 1)
    expected = _native_reference(x, n)
    out = torch.empty((3, 4, 5), dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        actual = torch.special.laguerre_polynomial_l(x, n, out=out)
    assert actual.data_ptr() == out.data_ptr()
    utils.gems_assert_close(actual, expected, dtype, equal_nan=True)


@pytest.mark.special_laguerre_polynomial_l_out
def test_special_laguerre_polynomial_l_out_resize_and_noncontiguous():
    x = torch.linspace(-0.5, 0.5, 15, device=flag_gems.device).reshape(3, 1, 5)
    n = torch.tensor([0, 1, 2, 6], device=flag_gems.device).reshape(1, 4, 1)
    expected = _native_reference(x, n)

    resized_out = torch.empty(0, device=flag_gems.device)
    with flag_gems.use_gems():
        resized_actual = torch.special.laguerre_polynomial_l(x, n, out=resized_out)
    assert resized_actual.data_ptr() == resized_out.data_ptr()
    assert resized_actual.shape == expected.shape
    utils.gems_assert_close(resized_actual, expected, x.dtype, equal_nan=True)

    storage = torch.empty((4, 3, 5), device=flag_gems.device)
    noncontiguous_out = storage.transpose(0, 1)
    with flag_gems.use_gems():
        noncontiguous_actual = torch.special.laguerre_polynomial_l(
            x, n, out=noncontiguous_out
        )
    assert noncontiguous_actual.data_ptr() == noncontiguous_out.data_ptr()
    assert not noncontiguous_actual.is_contiguous()
    utils.gems_assert_close(noncontiguous_actual, expected, x.dtype, equal_nan=True)


@pytest.mark.special_laguerre_polynomial_l_n_scalar_out
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test_special_laguerre_polynomial_l_n_scalar_out(dtype):
    x = torch.linspace(-0.5, 0.5, 31, dtype=dtype, device=flag_gems.device)
    n = 5.8
    expected = _native_reference(x, n)
    out = torch.empty_like(x)
    with flag_gems.use_gems():
        actual = torch.special.laguerre_polynomial_l(x, n, out=out)
    assert actual.data_ptr() == out.data_ptr()
    utils.gems_assert_close(actual, expected, dtype, equal_nan=True)


@pytest.mark.special_laguerre_polynomial_l_x_scalar_out
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test_special_laguerre_polynomial_l_x_scalar_out(dtype):
    x = 0.25
    n = torch.tensor([0.1, 1.1, 2.1, 7.1], dtype=dtype, device=flag_gems.device)
    expected = _native_reference(x, n)
    out = torch.empty_like(n)
    with flag_gems.use_gems():
        actual = torch.special.laguerre_polynomial_l(x, n, out=out)
    assert actual.data_ptr() == out.data_ptr()
    utils.gems_assert_close(actual, expected, dtype, equal_nan=True)


@pytest.mark.special_laguerre_polynomial_l
def test_special_laguerre_polynomial_l_rejects_unsupported_compute_dtype():
    x = torch.ones(4, dtype=torch.float16, device=flag_gems.device)
    n = torch.ones(4, dtype=torch.int64, device=flag_gems.device)
    with flag_gems.use_gems(), pytest.raises(RuntimeError, match="float32, float64"):
        torch.special.laguerre_polynomial_l(x, n)
