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


@pytest.mark.ldexp
@pytest.mark.parametrize("shape", [*utils.POINTWISE_SHAPES, (2, 0, 3)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_ldexp(shape, dtype):
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    exponent = torch.randint(-8, 9, shape, dtype=torch.int32, device=inp.device)
    reference = torch.ops.aten.ldexp.Tensor(
        utils.to_reference(inp, True), utils.to_reference(exponent, False)
    )

    result = flag_gems.ldexp(inp, exponent)

    assert result.shape == reference.shape
    assert result.dtype == reference.dtype
    utils.gems_assert_close(result, reference, result.dtype)


@pytest.mark.ldexp
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_ldexp_broadcast_noncontiguous(dtype):
    torch.manual_seed(0)
    inp = torch.randn((19, 7), dtype=dtype, device=flag_gems.device).T
    exponent = torch.randint(-4, 5, (19,), dtype=torch.int64, device=inp.device)
    assert not inp.is_contiguous()
    reference = torch.ops.aten.ldexp.Tensor(
        utils.to_reference(inp, True), utils.to_reference(exponent, False)
    )

    result = flag_gems.ldexp(inp, exponent)

    utils.gems_assert_close(result, reference, result.dtype)


@pytest.mark.ldexp
@pytest.mark.parametrize(
    "self_dtype,other_dtype,expected_dtype",
    [
        (torch.bool, torch.int32, torch.float32),
        (torch.int32, torch.int64, torch.float32),
        (torch.float16, torch.int64, torch.float16),
        (torch.float32, torch.float64, torch.float64),
    ],
)
def test_ldexp_dtype_promotion(self_dtype, other_dtype, expected_dtype):
    torch.manual_seed(0)
    if self_dtype == torch.bool:
        inp = torch.randint(0, 2, (37,), device=flag_gems.device).bool()
    elif self_dtype.is_floating_point:
        inp = torch.randn((37,), dtype=self_dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-8, 9, (37,), dtype=self_dtype, device=flag_gems.device)
    exponent = torch.randint(-4, 5, (37,), device=inp.device).to(other_dtype)
    reference = torch.ops.aten.ldexp.Tensor(
        utils.to_reference(inp, False), utils.to_reference(exponent, False)
    )

    result = flag_gems.ldexp(inp, exponent)

    assert result.dtype == expected_dtype
    utils.gems_assert_close(result, reference, expected_dtype)


@pytest.mark.ldexp
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_ldexp_special_values(dtype):
    inp = torch.tensor(
        [0.0, -0.0, 1.0, -1.0, float("inf"), -float("inf"), float("nan")],
        dtype=dtype,
        device=flag_gems.device,
    )
    exponent = torch.tensor([128, -128, 0, 1, -1, 0, 2], device=inp.device)
    reference = torch.ops.aten.ldexp.Tensor(
        utils.to_reference(inp, True), utils.to_reference(exponent, False)
    )

    result = flag_gems.ldexp(inp, exponent)

    utils.gems_assert_close(result, reference, dtype, equal_nan=True)


@pytest.mark.ldexp
def test_ldexp_autograd():
    torch.manual_seed(0)
    inp = torch.randn((11, 7), device=flag_gems.device, requires_grad=True)
    exponent = torch.randint(-3, 4, inp.shape, dtype=torch.int32, device=inp.device)
    grad = torch.randn_like(inp)
    ref_inp = utils.to_reference(inp, True).requires_grad_(True)
    ref_exponent = utils.to_reference(exponent, False)
    ref_grad = utils.to_reference(grad, True)
    ref_out = torch.ops.aten.ldexp.Tensor(ref_inp, ref_exponent)
    (ref_input_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)

    result = flag_gems.ldexp(inp, exponent)
    (input_grad,) = torch.autograd.grad(result, inp, grad)

    utils.gems_assert_close(result, ref_out, result.dtype)
    utils.gems_assert_close(input_grad, ref_input_grad, input_grad.dtype)


def _ldexp_out_reference(inp, exponent):
    if flag_gems.vendor_name == "enflame":
        # The native GCU Out reference is unavailable; both reference modes
        # use genuine CPU same-dtype math without changing global ref selection.
        lhs = inp.to(device="cpu")
        rhs = exponent.to(device="cpu")
        assert lhs.dtype == inp.dtype and rhs.dtype == exponent.dtype
    else:
        lhs = utils.to_reference(inp, True)
        rhs = utils.to_reference(exponent, False)
    return torch.ops.aten.ldexp.Tensor(lhs, rhs)


def _ldexp_out_metadata(tensor):
    storage = tensor.untyped_storage()
    metadata = (
        tensor.device,
        tensor.dtype,
        tuple(tensor.shape),
        tensor.stride(),
        tensor.storage_offset(),
        storage._cdata,
        tensor.data_ptr(),
        storage.nbytes(),
        tensor.requires_grad,
        tensor.is_conj(),
        tensor.is_neg(),
    )
    return metadata


def _ldexp_out_snapshot(tensor):
    metadata = _ldexp_out_metadata(tensor)
    # These original cases use finite Float3 values and small integer exponents.
    # Use logical CPU values, avoiding unsafe raw reads of vendor Long storage.
    host = tensor.detach().cpu()
    assert host.dtype == tensor.dtype and host.shape == tensor.shape
    contents = host.contiguous().reshape(-1).view(torch.uint8).clone()
    return metadata, contents


def _ldexp_out_assert_unchanged(tensor, before):
    after = _ldexp_out_snapshot(tensor)
    assert after[0] == before[0], "candidate changed original input metadata"
    assert torch.equal(after[1], before[1]), "candidate changed original input values"


def _call_ldexp_out_checked(inp, exponent, out):
    if flag_gems.vendor_name != "enflame":
        return flag_gems.ldexp_out(inp, exponent, out=out)
    lhs = _ldexp_out_snapshot(inp)
    rhs = _ldexp_out_snapshot(exponent)
    out_metadata = _ldexp_out_metadata(out)
    try:
        result = flag_gems.ldexp_out(inp, exponent, out=out)
    finally:
        _ldexp_out_assert_unchanged(inp, lhs)
        _ldexp_out_assert_unchanged(exponent, rhs)
    assert isinstance(result, torch.Tensor) and result is out
    # Validate original target ABI before numerical CPU comparisons.
    after_out = _ldexp_out_metadata(result)
    # Out promises the original object, pointer/layout/capacity and public ABI;
    # a zero-byte output need not retain an internal StorageImpl identity.
    assert after_out[:5] == out_metadata[:5]
    assert after_out[6:] == out_metadata[6:]
    return result


@pytest.mark.ldexp_out
@pytest.mark.parametrize("shape", [*utils.POINTWISE_SHAPES, (2, 0, 3)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_ldexp_out(shape, dtype):
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    exponent = torch.randint(-8, 9, shape, dtype=torch.int32, device=inp.device)
    inp_before, exponent_before = inp.clone(), exponent.clone()
    reference = _ldexp_out_reference(inp, exponent)
    out = torch.empty_like(inp)
    pointer = out.data_ptr()
    result = _call_ldexp_out_checked(inp, exponent, out)
    assert result is out and result.data_ptr() == pointer
    assert result.device == inp.device and result.dtype == dtype
    assert result.shape == reference.shape
    compared = result.to(device="cpu") if flag_gems.vendor_name == "enflame" else result
    utils.gems_assert_close(compared, reference, dtype)
    assert inp.device == inp_before.device and inp.dtype == inp_before.dtype
    assert (
        exponent.device == exponent_before.device
        and exponent.dtype == exponent_before.dtype
    )
    utils.gems_assert_equal(inp, utils.to_reference(inp_before, False))
    utils.gems_assert_equal(exponent, utils.to_reference(exponent_before, False))


@pytest.mark.ldexp_out
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_ldexp_out_broadcast_noncontiguous(dtype):
    torch.manual_seed(0)
    inp = torch.randn((19, 7), dtype=dtype, device=flag_gems.device).T
    exponent = torch.randint(-4, 5, (19,), dtype=torch.int64, device=inp.device)
    reference = _ldexp_out_reference(inp, exponent)
    backing = torch.full((7, 38), 123, dtype=dtype, device=inp.device)
    out = backing[:, ::2]
    pointer, stride = out.data_ptr(), out.stride()
    result = _call_ldexp_out_checked(inp, exponent, out)
    assert result is out and result.data_ptr() == pointer and result.stride() == stride
    assert result.device == inp.device and result.dtype == dtype
    assert result.shape == reference.shape
    compared = result.to(device="cpu") if flag_gems.vendor_name == "enflame" else result
    utils.gems_assert_close(compared, reference, dtype)
    assert torch.equal(backing[:, 1::2].cpu(), torch.full((7, 19), 123, dtype=dtype))
