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
    return utils.to_reference(inp, inp.is_complex())


def _assert_sgn_equal(result, reference, dtype):
    if dtype in utils.COMPLEX_DTYPES:
        utils.gems_assert_close(result, reference, dtype, equal_nan=True)
    else:
        utils.gems_assert_equal(result, reference, equal_nan=True)


def _make_sgn_out_input(shape, dtype):
    if flag_gems.vendor_name == "enflame" and dtype == torch.complex64:
        # Device normal does not support Complex64; preserve the normal recipe
        # and transport real complex storage without a narrowing typed cast.
        host = torch.randn(shape, dtype=dtype, device="cpu")
        return (
            host.contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .to(flag_gems.device)
            .view(dtype)
            .reshape(shape)
        )
    return _make_input(shape, dtype)


def _sgn_out_logical_cpu(tensor):
    logical = torch.empty(0, dtype=tensor.dtype, device="cpu").set_(
        tensor.untyped_storage().cpu(),
        tensor.storage_offset(),
        tensor.shape,
        tensor.stride(),
    )
    # Storage bytes exclude lazy view bits; restore their logical meaning only
    # on CPU. Snapshot validation deliberately keeps physical bytes + flags.
    if tensor.is_conj():
        logical = logical.conj_physical()
    if tensor.is_neg():
        logical = logical.neg()
    return logical


def _assert_sgn_out_equal(result, reference, dtype):
    if flag_gems.vendor_name in {"enflame", "kunlunxin", "ascend"} and dtype.is_complex:
        # Device complex comparison/cast support is incomplete. Preserve the
        # declared complex comparison policy after checking the target ABI.
        utils.gems_assert_close(
            _sgn_out_logical_cpu(result),
            _sgn_out_logical_cpu(reference),
            dtype,
            equal_nan=True,
        )
    else:
        _assert_sgn_equal(result, reference, dtype)


def _sgn_out_snapshot(tensor):
    metadata = (
        tensor.device,
        tensor.dtype,
        tuple(tensor.shape),
        tensor.stride(),
        tensor.storage_offset(),
        tensor.untyped_storage()._cdata,
        tensor.data_ptr(),
        tensor.requires_grad,
        tensor.is_conj(),
        tensor.is_neg(),
        tensor.untyped_storage().nbytes(),
    )
    logical = torch.empty(0, dtype=tensor.dtype, device="cpu").set_(
        tensor.untyped_storage().cpu(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tensor.stride(),
    )
    return metadata, logical.contiguous().reshape(-1).view(torch.uint8).clone()


def _sgn_out_assert_unchanged(tensor, snapshot, *, values=True):
    metadata, payload = _sgn_out_snapshot(tensor)
    assert metadata == snapshot[0]
    if values:
        assert torch.equal(payload, snapshot[1])


def _call_sgn_out_checked(inp, *, out):
    original = _sgn_out_snapshot(inp)
    original_out = _sgn_out_snapshot(out)
    aliases_out = all(
        original[0][index] == original_out[0][index] for index in (1, 2, 3, 4, 5, 8, 9)
    )
    try:
        result = flag_gems.sgn_out(inp, out=out)
    except BaseException:
        _sgn_out_assert_unchanged(inp, original)
        _sgn_out_assert_unchanged(out, original_out)
        raise
    _sgn_out_assert_unchanged(inp, original, values=not aliases_out)
    assert result is out
    assert result.device == original[0][0] and result.dtype == original[0][1]
    assert tuple(result.shape) == original[0][2]
    metadata, _ = _sgn_out_snapshot(out)
    if original_out[0][2] == original[0][2]:
        assert metadata == original_out[0]
    else:
        assert metadata[:2] == original_out[0][:2]
        assert metadata[4] == original_out[0][4]
        assert metadata[7:-1] == original_out[0][7:-1]
    return result


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
        result = _call_sgn_out_checked(inp, out=out)
        assert result is out
    if out is not None:
        _assert_sgn_out_equal(result, ref_out, inp.dtype)
    else:
        _assert_sgn_equal(result, ref_out, inp.dtype)


@pytest.mark.sgn_out
@pytest.mark.parametrize("dtype,shape", SGN_CASES)
def test_sgn_out(dtype, shape):
    inp = _make_sgn_out_input(shape, dtype)
    out = torch.empty_like(inp)
    _assert_matches_torch(inp, out)


@pytest.mark.sgn_out
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_sgn_out_noncontiguous(dtype):
    inp = _make_sgn_out_input((11, 7), dtype)
    out = torch.empty((7, 11), dtype=dtype, device=flag_gems.device).transpose(0, 1)
    _assert_matches_torch(inp, out)


@pytest.mark.sgn_out
def test_sgn_out_resizes_empty_tensor():
    inp = _make_sgn_out_input((3, 5), torch.float32)
    out = torch.empty(0, dtype=inp.dtype, device=flag_gems.device)
    _assert_matches_torch(inp, out)
    assert out.shape == inp.shape


@pytest.mark.sgn_out
def test_sgn_out_rejects_mismatched_dtype():
    inp = _make_sgn_out_input((3, 5), torch.float32)
    out = torch.empty_like(inp, dtype=torch.int32)
    reference_error = (
        (RuntimeError, ValueError)
        if flag_gems.vendor_name == "kunlunxin" and not utils.TO_CPU
        else RuntimeError
    )
    ref_input = _to_reference(inp)
    ref_out = utils.to_reference(out, False)
    with pytest.raises(reference_error):
        torch.sgn(ref_input, out=ref_out)
    with pytest.raises(RuntimeError):
        _call_sgn_out_checked(inp, out=out)
