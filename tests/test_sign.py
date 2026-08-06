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

REAL_DTYPES = (
    utils.ALL_FLOAT_DTYPES
    + [torch.int8, torch.uint8]
    + utils.ALL_INT_DTYPES
    + utils.BOOL_TYPES
)
SIGN_CASES = [
    (dtype, shape) for dtype in REAL_DTYPES for shape in utils.POINTWISE_SHAPES
]


def _make_input(shape, dtype):
    if dtype == torch.bool:
        return torch.randint(
            0, 2, shape, dtype=torch.int32, device=flag_gems.device
        ).bool()
    if dtype == torch.uint8:
        return torch.randint(0, 100, shape, dtype=dtype, device=flag_gems.device)
    if dtype in utils.ALL_INT_DTYPES or dtype == torch.int8:
        return torch.randint(-100, 100, shape, dtype=dtype, device=flag_gems.device)
    return torch.randn(shape, dtype=dtype, device=flag_gems.device)


def _to_reference(inp):
    return utils.to_reference(inp, False)


def _assert_matches_torch(inp, out=None):
    ref_inp = _to_reference(inp)
    if out is None:
        ref_out = torch.sign(ref_inp)
        with flag_gems.use_gems():
            result = torch.sign(inp)
    else:
        ref_out = torch.empty_like(ref_inp)
        torch.sign(ref_inp, out=ref_out)
        with flag_gems.use_gems():
            result = torch.sign(inp, out=out)
        assert result is out

    # torch.sign(nan) returns 0.0, not nan
    utils.gems_assert_equal(result, ref_out, equal_nan=False)


def _assert_inplace_matches_torch(inp):
    ref_inp = _to_reference(inp.clone())
    ref_out = ref_inp.sign_()
    with flag_gems.use_gems():
        result = inp.sign_()
    assert result is inp
    # torch.sign(nan) returns 0.0, not nan
    utils.gems_assert_equal(result, ref_out, equal_nan=False)
    # the input tensor itself must have been mutated in place
    utils.gems_assert_equal(inp, ref_out, equal_nan=False)


@pytest.mark.sign
@pytest.mark.parametrize("dtype,shape", SIGN_CASES)
def test_sign(dtype, shape):
    inp = _make_input(shape, dtype)
    _assert_matches_torch(inp)


@pytest.mark.sign_out
@pytest.mark.parametrize("dtype,shape", SIGN_CASES)
def test_sign_out(dtype, shape):
    inp = _make_input(shape, dtype)
    out = torch.empty_like(inp)
    _assert_matches_torch(inp, out)


@pytest.mark.sign_
@pytest.mark.parametrize("dtype,shape", SIGN_CASES)
def test_sign_(dtype, shape):
    inp = _make_input(shape, dtype)
    _assert_inplace_matches_torch(inp)


@pytest.mark.sign_
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_sign__noncontiguous(dtype):
    inp = _make_input((7, 11), dtype).transpose(0, 1)
    _assert_inplace_matches_torch(inp)


@pytest.mark.sign
def test_sign_special_values():
    """Test special values: -inf, -0, 0, +inf, nan"""
    inp = torch.tensor(
        [float("-inf"), -1.0, -0.0, 0.0, 1.0, float("inf"), float("nan")],
        dtype=torch.float32,
        device=flag_gems.device,
    )
    _assert_matches_torch(inp)


@pytest.mark.sign
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_sign_noncontiguous(dtype):
    inp = _make_input((7, 11), dtype).transpose(0, 1)
    _assert_matches_torch(inp)


@pytest.mark.sign_out
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_sign_out_noncontiguous(dtype):
    inp = _make_input((11, 7), dtype)
    out = torch.empty((7, 11), dtype=dtype, device=flag_gems.device).transpose(0, 1)
    _assert_matches_torch(inp, out)


@pytest.mark.sign_out
def test_sign_out_resizes_empty_tensor():
    inp = _make_input((3, 5), torch.float32)
    out = torch.empty(0, dtype=inp.dtype, device=flag_gems.device)

    _assert_matches_torch(inp, out)
    assert out.shape == inp.shape


@pytest.mark.sign_out
def test_sign_out_rejects_mismatched_dtype():
    inp = _make_input((3, 5), torch.float32)
    out = torch.empty_like(inp, dtype=torch.int32)

    with pytest.raises(RuntimeError):
        torch.sign(utils.to_reference(inp), out=utils.to_reference(out))
    with flag_gems.use_gems(), pytest.raises(RuntimeError):
        torch.sign(inp, out=out)


@pytest.mark.sign
def test_sign_empty():
    inp = torch.empty((0, 3), dtype=torch.float32, device=flag_gems.device)
    _assert_matches_torch(inp)


@pytest.mark.sign
def test_sign_rejects_complex():
    """Complex dtypes should raise NotImplementedError"""
    inp = torch.tensor([1 + 1j, 2 + 2j], dtype=torch.complex64, device=flag_gems.device)
    with pytest.raises(NotImplementedError):
        with flag_gems.use_gems():
            torch.sign(inp)
