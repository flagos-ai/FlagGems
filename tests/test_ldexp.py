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


# Current pinned master functional cases.
@pytest.mark.ldexp
@pytest.mark.parametrize(
    "dtype,exponent", [(torch.float32, 128), (torch.float64, 1024)]
)
def test_ldexp_integral_exponent_boundaries(dtype, exponent):
    if flag_gems.vendor_name == "kunlunxin" and (dtype, exponent) in ((torch.float32, 128), (torch.float64, 1024)):
        pytest.skip('klx native aten.ldexp.Tensor fails master integral-exponent boundary values (master-batch01 9e41c8b, both refs)')
    inp = torch.tensor(
        [2.0 ** (-exponent + 2), 2.0 ** (-exponent + 1), 0.0, float("inf")],
        dtype=dtype,
        device=flag_gems.device,
    )
    other = torch.tensor(
        [exponent, exponent, exponent, -exponent], dtype=torch.int64, device=inp.device
    )
    reference = utils.to_reference(
        torch.tensor([4.0, 2.0, 0.0, float("inf")], dtype=dtype, device=inp.device)
    )
    result = flag_gems.ldexp(inp, other)
    utils.gems_assert_equal(result, reference, equal_nan=True)


@pytest.mark.ldexp
@pytest.mark.parametrize("cpu_operand", ["self", "other"])
@pytest.mark.parametrize("exponent_dtype", [torch.int64, torch.float32])
def test_ldexp_cpu_scalar_operand(cpu_operand, exponent_dtype):
    inp = torch.randn(17, device=flag_gems.device)
    other = torch.randint(-3, 4, (17,), device=inp.device).to(exponent_dtype)
    if cpu_operand == "self":
        inp = torch.tensor(1.5)
    else:
        other = torch.tensor(3, dtype=exponent_dtype)
    try:
        torch.ldexp(inp, other)
    except RuntimeError:
        with pytest.raises(RuntimeError):
            flag_gems.ldexp(inp, other)
        return
    reference = torch.ldexp(utils.to_reference(inp), utils.to_reference(other))
    result = flag_gems.ldexp(inp, other)
    utils.gems_assert_close(result, reference, result.dtype)


@pytest.mark.ldexp
@pytest.mark.parametrize(
    "conj_input,conj_other", [(True, False), (False, True), (True, True)]
)
def test_ldexp_lazy_conjugation(conj_input, conj_other):
    if flag_gems.vendor_name == "kunlunxin":
        pytest.skip('klx native ldexp complex full kernel is not implemented (master-batch01 9e41c8b, both refs)')
    inp = torch.randn(17, dtype=torch.complex64, device=flag_gems.device)
    other = torch.randn_like(inp)
    if conj_input:
        inp = inp.conj()
    if conj_other:
        other = other.conj()
    reference = torch.ldexp(utils.to_reference(inp), utils.to_reference(other))
    result = flag_gems.ldexp(inp, other)
    utils.gems_assert_close(result, reference, inp.dtype)


@pytest.mark.ldexp
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_ldexp(shape, dtype):
    self = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    other = torch.randint(-8, 9, shape, device=flag_gems.device, dtype=torch.int32)

    ref_self = utils.to_reference(self, True)
    ref_other = utils.to_reference(other)
    ref_out = torch.ops.aten.ldexp.Tensor(ref_self, ref_other)

    res_out = flag_gems.ldexp(self, other)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.ldexp
@pytest.mark.parametrize(
    "self_dtype,other_dtype,expected_dtype",
    [
        (torch.bool, torch.int32, torch.float32),
        (torch.int32, torch.int64, torch.float32),
        (torch.float16, torch.int64, torch.float16),
        (torch.bfloat16, torch.int32, torch.bfloat16),
        (torch.float16, torch.float32, torch.float32),
        (torch.float32, torch.float64, torch.float64),
    ],
)
def test_ldexp_dtype_promotion(self_dtype, other_dtype, expected_dtype):
    if flag_gems.vendor_name == "kunlunxin" and (self_dtype, other_dtype, expected_dtype) == (torch.float32, torch.float64, torch.float64):
        pytest.skip('klx native fixture/reference returns Float for requested Double promotion (master-batch01 9e41c8b, both refs)')
    shape = (37, 53)
    if self_dtype == torch.bool:
        self = torch.randint(0, 2, shape, device=flag_gems.device).bool()
    elif self_dtype.is_floating_point:
        self = torch.randn(shape, dtype=self_dtype, device=flag_gems.device)
    else:
        self = torch.randint(-8, 9, shape, dtype=self_dtype, device=flag_gems.device)
    if other_dtype.is_floating_point:
        other = torch.randn(shape, dtype=other_dtype, device=flag_gems.device) * 3
    else:
        other = torch.randint(-8, 9, shape, dtype=other_dtype, device=flag_gems.device)

    ref_self = utils.to_reference(self)
    ref_other = utils.to_reference(other)
    ref_out = torch.ops.aten.ldexp.Tensor(ref_self, ref_other)

    res_out = flag_gems.ldexp(self, other)

    assert ref_out.dtype == expected_dtype
    assert res_out.dtype == ref_out.dtype
    utils.gems_assert_close(res_out, ref_out, expected_dtype)


@pytest.mark.ldexp
@pytest.mark.parametrize(
    "self_dtype,other_dtype,expected_dtype",
    [
        (torch.complex64, torch.complex128, torch.complex64),
        (torch.float32, torch.float64, torch.float32),
        (torch.float16, torch.float32, torch.float16),
    ],
)
def test_ldexp_zero_dim_promotion(self_dtype, other_dtype, expected_dtype):
    if flag_gems.vendor_name == "kunlunxin" and (self_dtype, other_dtype, expected_dtype) == (torch.complex64, torch.complex128, torch.complex64):
        pytest.skip('klx native ldexp complex full kernel is not implemented (master-batch01 9e41c8b, both refs)')
    if self_dtype.is_complex and flag_gems.vendor_name in ("ascend", "tsingmicro"):
        pytest.skip("The backend does not support complex tensors")
    self = torch.randn((17,), dtype=self_dtype, device=flag_gems.device)
    other = torch.randn((), dtype=other_dtype, device=flag_gems.device)
    ref_self = utils.to_reference(self)
    ref_other = utils.to_reference(other)

    ref_out = torch.ops.aten.ldexp.Tensor(ref_self, ref_other)
    res_out = flag_gems.ldexp(self, other)

    assert ref_out.dtype == expected_dtype
    assert res_out.dtype == ref_out.dtype
    utils.gems_assert_close(res_out, ref_out, expected_dtype)


@pytest.mark.ldexp
def test_ldexp_broadcast_noncontiguous_and_empty():
    self = torch.randn((19, 7), device=flag_gems.device).T
    other = torch.randint(-8, 9, (19,), device=flag_gems.device, dtype=torch.int64)

    ref_self = utils.to_reference(self, True)
    ref_other = utils.to_reference(other)
    ref_out = torch.ops.aten.ldexp.Tensor(ref_self, ref_other)
    res_out = flag_gems.ldexp(self, other)
    utils.gems_assert_close(res_out, ref_out, torch.float32)

    empty = torch.empty((0, 7), device=flag_gems.device)
    empty_other = torch.empty((1, 7), dtype=torch.int32, device=flag_gems.device)
    empty_out = flag_gems.ldexp(empty, empty_other)
    assert empty_out.shape == (0, 7)
    assert empty_out.numel() == 0


@pytest.mark.ldexp
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_ldexp_special_values(dtype):
    if flag_gems.vendor_name == "kunlunxin" and utils.TO_CPU and dtype in (torch.float16, torch.float32, torch.bfloat16):
        pytest.skip('klx native ldexp special-value sign bits differ from CPU reference; default reference passes (master-batch01 9e41c8b)')
    self = torch.tensor(
        [0.0, -0.0, 1.0, -1.0, float("inf"), -float("inf"), float("nan")],
        dtype=dtype,
        device=flag_gems.device,
    )
    other = torch.tensor(
        [float("inf"), -float("inf"), 128.0, -128.0, 0.5, 0.0, 2.0],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_self = utils.to_reference(self, True)
    ref_other = utils.to_reference(other, True)
    ref_out = torch.ops.aten.ldexp.Tensor(ref_self, ref_other)

    res_out = flag_gems.ldexp(self, other)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
    res_cpu = res_out.cpu()
    ref_cpu = ref_out.cpu()
    valid = ~(torch.isnan(res_cpu) | torch.isnan(ref_cpu))
    utils.gems_assert_equal(
        torch.signbit(res_cpu)[valid],
        torch.signbit(ref_cpu)[valid],
    )


@pytest.mark.ldexp
@pytest.mark.skipif(
    flag_gems.vendor_name in ("ascend", "tsingmicro"),
    reason="The backend does not support complex tensors",
)
def test_ldexp_complex():
    if flag_gems.vendor_name == "kunlunxin":
        pytest.skip('klx native ldexp complex full kernel is not implemented (master-batch01 9e41c8b, both refs)')
    self = torch.randn((19, 7), dtype=torch.complex64, device=flag_gems.device)
    other = torch.randn((7,), dtype=torch.complex64, device=flag_gems.device)
    ref_self = utils.to_reference(self, True)
    ref_other = utils.to_reference(other, True)
    ref_out = torch.ops.aten.ldexp.Tensor(ref_self, ref_other)

    res_out = flag_gems.ldexp(self, other)

    utils.gems_assert_close(res_out, ref_out, torch.complex64)
