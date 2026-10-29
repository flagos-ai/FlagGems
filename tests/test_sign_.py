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
    if (
        flag_gems.vendor_name == "enflame"
        and not utils.TO_CPU
        and inp.dtype == torch.bool
    ):
        pytest.skip(
            "Enflame native sign does not support Bool (topsatenSign NOT_SUPPORT)"
        )
    if (
        flag_gems.vendor_name in ("ascend", "kunlunxin")
        and not utils.TO_CPU
        and inp.dtype in (torch.int8, torch.uint8, torch.int16)
    ):
        pytest.skip("backend native sign reference does not support int8/uint8/int16")
    return utils.to_reference(inp, False)


def _sign_storage_overlaps(left, right):
    if left.device != right.device:
        return False
    left_storage = left.untyped_storage()
    right_storage = right.untyped_storage()
    if left_storage._cdata == right_storage._cdata:
        return True
    left_bytes = left_storage.nbytes()
    right_bytes = right_storage.nbytes()
    if left_bytes == 0 or right_bytes == 0:
        return False
    left_start = left_storage.data_ptr()
    right_start = right_storage.data_ptr()
    return (
        left_start < right_start + right_bytes and right_start < left_start + left_bytes
    )


def _sign_snapshot(tensor):
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


def _sign_assert_unchanged(tensor, snapshot, *, values=True):
    metadata, payload = _sign_snapshot(tensor)
    assert metadata == snapshot[0]
    if values:
        assert torch.equal(payload, snapshot[1])


def _call_sign_checked(operator, inp, *, out=None):
    assert operator in {"sign", "sign_", "sign_out"}
    original = _sign_snapshot(inp)
    original_out = _sign_snapshot(out) if out is not None else None
    aliases_out = out is not None and original_out[0][5] == original[0][5]
    try:
        if operator == "sign":
            result = flag_gems.sign(inp)
        elif operator == "sign_":
            result = flag_gems.sign_(inp)
        else:
            result = flag_gems.sign_out(inp, out=out)
    except BaseException:
        _sign_assert_unchanged(inp, original)
        if out is not None:
            _sign_assert_unchanged(out, original_out)
        raise
    _sign_assert_unchanged(
        inp, original, values=operator != "sign_" and not aliases_out
    )
    assert isinstance(result, torch.Tensor)
    assert result.device == original[0][0] and result.dtype == original[0][1]
    assert tuple(result.shape) == original[0][2]
    if operator == "sign_":
        assert result is inp
    elif out is None:
        assert result is not inp
        assert not _sign_storage_overlaps(
            result, inp
        ), "candidate output aliases input storage"
    else:
        assert result is out
        metadata, _ = _sign_snapshot(out)
        if original_out[0][2] == original[0][2]:
            assert metadata == original_out[0]
        else:
            assert metadata[:2] == original_out[0][:2]
            assert metadata[4] == original_out[0][4]
            assert metadata[7:-1] == original_out[0][7:-1]
    return result


def _assert_matches_torch(inp, out=None):
    ref_inp = _to_reference(inp)
    if out is None:
        ref_out = torch.sign(ref_inp)
        result = _call_sign_checked("sign", inp)
    else:
        ref_out = torch.empty_like(ref_inp)
        torch.sign(ref_inp, out=ref_out)
        result = _call_sign_checked("sign_out", inp, out=out)
        assert result is out
    utils.gems_assert_equal(result, ref_out, equal_nan=False)


def _assert_inplace_matches_torch(inp):
    ref_inp = _to_reference(inp.clone())
    ref_out = ref_inp.sign_()
    result = _call_sign_checked("sign_", inp)
    assert result is inp
    utils.gems_assert_equal(result, ref_out, equal_nan=False)
    utils.gems_assert_equal(inp, ref_out, equal_nan=False)


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
    reference_error = (
        (RuntimeError, ValueError)
        if flag_gems.vendor_name == "kunlunxin" and not utils.TO_CPU
        else RuntimeError
    )
    with pytest.raises(reference_error):
        torch.sign(utils.to_reference(inp), out=utils.to_reference(out))
    with pytest.raises(RuntimeError):
        _call_sign_checked("sign_out", inp, out=out)
