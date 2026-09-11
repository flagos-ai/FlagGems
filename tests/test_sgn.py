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

import warnings

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

REAL_DTYPES = (
    utils.ALL_FLOAT_DTYPES
    + [torch.int8, torch.uint8]
    + utils.ALL_INT_DTYPES
    + utils.BOOL_TYPES
)
COMPLEX_SHAPES = [(17,), (7, 11), (3, 5, 7)]
SGN_CASES = [
    (dtype, shape) for dtype in REAL_DTYPES for shape in utils.POINTWISE_SHAPES
] + [(dtype, shape) for dtype in utils.COMPLEX_DTYPES for shape in COMPLEX_SHAPES]


def _make_input(shape, dtype):
    if dtype == torch.bool:
        return torch.randint(
            0, 2, shape, dtype=torch.int32, device=flag_gems.device
        ).bool()
    if dtype == torch.uint8:
        return torch.randint(0, 100, shape, dtype=dtype, device=flag_gems.device)
    if dtype in utils.ALL_INT_DTYPES or dtype == torch.int8:
        return torch.randint(-100, 100, shape, dtype=dtype, device=flag_gems.device)
    if dtype == torch.complex32:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="ComplexHalf support is experimental.*",
                category=UserWarning,
            )
            return torch.randn(shape, dtype=dtype, device=flag_gems.device)
    return torch.randn(shape, dtype=dtype, device=flag_gems.device)


def _to_reference(inp):
    return utils.to_reference(inp, inp.dtype in utils.COMPLEX_DTYPES)


def _assert_sgn_equal(result, reference, dtype):
    if dtype in utils.COMPLEX_DTYPES:
        utils.gems_assert_close(result, reference, dtype, equal_nan=True)
    else:
        utils.gems_assert_equal(result, reference, equal_nan=True)


def _assert_matches_torch(inp, out=None):
    ref_inp = _to_reference(inp)
    if out is None:
        ref_out = torch.sgn(ref_inp)
        with flag_gems.use_gems():
            result = torch.sgn(inp)
    else:
        ref_out = torch.empty_like(ref_inp)
        torch.sgn(ref_inp, out=ref_out)
        with flag_gems.use_gems():
            result = torch.sgn(inp, out=out)
        assert result is out

    _assert_sgn_equal(result, ref_out, inp.dtype)


@pytest.mark.sgn
@pytest.mark.parametrize("dtype,shape", SGN_CASES)
def test_sgn(dtype, shape):
    inp = _make_input(shape, dtype)
    _assert_matches_torch(inp)


@pytest.mark.sgn_out
@pytest.mark.parametrize("dtype,shape", SGN_CASES)
def test_sgn_out(dtype, shape):
    inp = _make_input(shape, dtype)
    out = torch.empty_like(inp)
    _assert_matches_torch(inp, out)


@pytest.mark.sgn_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sgn_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = ref_inp.sgn_()
    with flag_gems.use_gems():
        res_out = inp.sgn_()
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.sgn
def test_sgn_special_values():
    inp = torch.tensor(
        [float("-inf"), -1.0, -0.0, 0.0, 1.0, float("inf"), float("nan")],
        dtype=torch.float32,
        device=flag_gems.device,
    )
    _assert_matches_torch(inp)


@pytest.mark.sgn
def test_sgn_complex_extreme_values():
    inp = torch.tensor(
        [0j, 3 + 4j, -5 + 12j, 1e20 + 1e20j, 1e-20 - 1e-20j],
        dtype=torch.complex64,
        device=flag_gems.device,
    )
    _assert_matches_torch(inp)


@pytest.mark.sgn
def test_sgn_complex_nonfinite_values():
    inp = torch.tensor(
        [
            complex(float("nan"), 1.0),
            complex(float("inf"), 1.0),
            complex(1.0, float("inf")),
            complex(float("nan"), float("nan")),
        ],
        dtype=torch.complex64,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.sgn(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.sgn(inp)

    res_real = torch.view_as_real(utils.to_cpu(res_out, ref_out))
    ref_real = torch.view_as_real(ref_out).to(dtype=res_real.dtype)
    assert torch.equal(torch.isnan(res_real), torch.isnan(ref_real))
    torch.testing.assert_close(res_real, ref_real, equal_nan=True)


@pytest.mark.sgn
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_sgn_noncontiguous(dtype):
    inp = _make_input((7, 11), dtype).transpose(0, 1)
    _assert_matches_torch(inp)


@pytest.mark.sgn_out
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_sgn_out_noncontiguous(dtype):
    inp = _make_input((11, 7), dtype)
    out = torch.empty((7, 11), dtype=dtype, device=flag_gems.device).transpose(0, 1)
    _assert_matches_torch(inp, out)


@pytest.mark.sgn_out
def test_sgn_out_resizes_empty_tensor():
    inp = _make_input((3, 5), torch.float32)
    out = torch.empty(0, dtype=inp.dtype, device=flag_gems.device)

    _assert_matches_torch(inp, out)
    assert out.shape == inp.shape


@pytest.mark.sgn_out
def test_sgn_out_rejects_mismatched_dtype():
    inp = _make_input((3, 5), torch.float32)
    out = torch.empty_like(inp, dtype=torch.int32)

    with pytest.raises(RuntimeError):
        torch.sgn(utils.to_reference(inp), out=utils.to_reference(out))
    with flag_gems.use_gems(), pytest.raises(RuntimeError):
        torch.sgn(inp, out=out)


@pytest.mark.sgn
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_sgn_empty(dtype):
    inp = torch.empty((0, 3), dtype=dtype, device=flag_gems.device)
    _assert_matches_torch(inp)
