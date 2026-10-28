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


@pytest.fixture(autouse=True)
def _seed_random_inputs():
    utils.init_seed(0)


def _make_input(shape, dtype):
    if flag_gems.vendor_name == "ascend" and dtype.is_complex:
        return torch.randn(shape, dtype=dtype, device="cpu").to(flag_gems.device)
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
    if (
        flag_gems.vendor_name == "ascend"
        and inp.dtype == torch.complex32
        and not utils.TO_CPU
    ):
        # NPU can store ComplexHalf, but its device cast cannot upcast it.
        return utils.to_reference(inp.cpu(), True).to(inp.device)
    return utils.to_reference(inp, inp.is_floating_point() or inp.is_complex())


def _assert_sgn_equal(result, reference, dtype):
    assert result.dtype == dtype
    if dtype.is_complex:
        utils.gems_assert_close(result, reference, dtype, equal_nan=True)
    else:
        utils.gems_assert_equal(result, reference.to(dtype), equal_nan=True)


def _assert_matches_torch(inp, out=None):
    if (
        out is not None
        and flag_gems.vendor_name == "ascend"
        and not utils.TO_CPU
        and inp.dtype in (torch.int8, torch.uint8, torch.int16)
    ):
        pytest.skip("Ascend native sgn out reference does not support int8/uint8/int16")
    if (
        out is not None
        and flag_gems.vendor_name == "kunlunxin"
        and not utils.TO_CPU
        and inp.dtype in (torch.int8, torch.uint8, torch.int16)
    ):
        pytest.skip(
            "Kunlunxin native sgn out reference does not support this integer dtype"
        )
    ref_inp = _to_reference(inp)
    if out is None:
        ref_out = torch.sgn(ref_inp)
        with flag_gems.use_gems():
            result = torch.sgn(inp)
    else:
        ref_out = torch.empty_like(ref_inp)
        torch.sgn(ref_inp, out=ref_out)
        result = flag_gems.sgn_out(inp, out=out)
        assert result is out
        assert result.device == inp.device
        assert result.dtype == inp.dtype
        assert result.shape == inp.shape
    _assert_sgn_equal(result, ref_out, inp.dtype)


@pytest.mark.sgn_out
@pytest.mark.parametrize("dtype,shape", SGN_CASES)
def test_sgn_out(dtype, shape):
    inp = _make_input(shape, dtype)
    out = torch.empty_like(inp)
    _assert_matches_torch(inp, out)


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
        torch.sgn(utils.to_reference(inp, True), out=out.detach().cpu())
    with pytest.raises(RuntimeError):
        flag_gems.sgn_out(inp, out=out)
