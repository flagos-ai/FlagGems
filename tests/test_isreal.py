# Copyright 2026, The FlagOS Contributors.
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


def _isreal_input_metadata(inp):
    return {
        "device": inp.device,
        "dtype": inp.dtype,
        "shape": tuple(inp.shape),
        "stride": tuple(inp.stride()),
        "storage_offset": inp.storage_offset(),
        "data_ptr": inp.untyped_storage().data_ptr(),
        "is_conj": inp.is_conj(),
        "is_neg": inp.is_neg(),
        "requires_grad": inp.requires_grad,
    }


def _isreal_input_bytes(inp):
    # Preserve NaN bits; resolve existing lazy flags only on this CPU snapshot.
    return (
        inp.detach()
        .cpu()
        .resolve_conj()
        .resolve_neg()
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .clone()
    )


def _assert_isreal_output(result, original):
    # Compare against the original target ABI, even if candidate changed input.
    assert isinstance(result, torch.Tensor)
    assert result.shape == original["shape"] and result.dtype == torch.bool
    assert result.device == original["device"]
    if result.numel():
        assert result.untyped_storage().data_ptr() != original["data_ptr"]


def _call_isreal(inp):
    original = _isreal_input_metadata(inp)
    original_bytes = _isreal_input_bytes(inp)
    result = flag_gems.isreal(inp)
    assert _isreal_input_metadata(inp) == original
    assert torch.equal(_isreal_input_bytes(inp), original_bytes)
    _assert_isreal_output(result, original)
    return result


def _skip_complex_reference(dtype, special=False):
    if dtype == torch.complex128 and flag_gems.vendor_name == "enflame":
        pytest.skip(
            "Enflame target Complex128 construction/typed observation does not preserve values"
        )
    if utils.TO_CPU:
        return
    if dtype == torch.complex128 and flag_gems.vendor_name == "kunlunxin":
        pytest.skip("Kunlunxin native isreal does not support Complex128 input")
    if special and dtype == torch.complex128 and (flag_gems.vendor_name == "iluvatar"):
        pytest.skip(
            "Iluvatar native isreal mishandles nonfinite Complex128 imaginary values"
        )


def _complex_target(value, device):
    if value.dtype == torch.complex128 and flag_gems.vendor_name == "kunlunxin":
        target = (
            value.contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .to(device)
            .view(value.dtype)
            .reshape(value.shape)
        )
    else:
        target = value.to(device)
    assert target.dtype == value.dtype and target.shape == value.shape
    assert target.device.type == torch.device(device).type
    return target


def _make_complex(shape, dtype, device):
    """Preserve the requested complex dtype when target factories narrow it."""
    _skip_complex_reference(dtype)
    float_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    factory_device = (
        "cpu"
        if dtype == torch.complex128 and flag_gems.vendor_name == "kunlunxin"
        else device
    )
    real = torch.randn(shape, dtype=float_dtype, device=factory_device)
    imag = torch.randn(shape, dtype=float_dtype, device=factory_device)
    if imag.dim() > 0:
        imag[::2] = 0
    else:
        imag = torch.zeros_like(imag)
    return _complex_target(torch.complex(real, imag).to(dtype), device)


@pytest.mark.isreal
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize(
    "dtype",
    utils.FLOAT_DTYPES
    + utils.ALL_INT_DTYPES
    + utils.BOOL_TYPES
    + [torch.complex64, torch.complex128],
)
def test_isreal(shape, dtype):
    if dtype.is_complex:
        inp = _make_complex(shape, dtype, flag_gems.device)
    else:
        inp = torch.randn(shape, dtype=torch.float32, device=flag_gems.device).to(dtype)
    assert inp.dtype == dtype and inp.device.type == torch.device(flag_gems.device).type
    ref_inp = utils.to_reference(inp)

    ref_out = torch.isreal(ref_inp)
    res_out = _call_isreal(inp)

    assert res_out.dtype == torch.bool
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.isreal
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64, torch.complex128])
def test_isreal_special_values(dtype):
    """Zero, negative zero, NaN and infinity, including a NaN imaginary part."""
    _skip_complex_reference(dtype, special=True)
    if dtype.is_complex:
        float_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
        factory_device = (
            "cpu"
            if dtype == torch.complex128 and flag_gems.vendor_name == "kunlunxin"
            else flag_gems.device
        )
        real = [0.0, -0.0, 1.0, -1.0, float("nan"), float("inf"), 1.0, 1.0, -0.0]
        imag = [0.0, -0.0, 0.0, 0.0, 0.0, 0.0, float("nan"), float("inf"), -0.0]
        inp = _complex_target(
            torch.complex(
                torch.tensor(real, dtype=float_dtype, device=factory_device),
                torch.tensor(imag, dtype=float_dtype, device=factory_device),
            ).to(dtype),
            flag_gems.device,
        )
    else:
        inp = torch.tensor(
            [0.0, -0.0, 1.0, -1.0, float("nan"), float("inf"), 1.0, 1.0, -0.0],
            dtype=dtype,
            device=flag_gems.device,
        )
    assert inp.dtype == dtype and inp.device.type == torch.device(flag_gems.device).type
    ref_inp = utils.to_reference(inp)
    ref_out = torch.isreal(ref_inp)
    res_out = _call_isreal(inp)
    assert res_out.dtype == torch.bool
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.isreal
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64, torch.complex128])
def test_isreal_non_contiguous(dtype):
    if dtype.is_complex:
        base = _make_complex((32, 48), dtype, flag_gems.device)
    else:
        base = torch.randn((32, 48), dtype=dtype, device=flag_gems.device)
    inp = base.t()
    assert not inp.is_contiguous()
    assert inp.dtype == dtype and inp.device.type == torch.device(flag_gems.device).type
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.isreal(ref_inp)
    res_out = _call_isreal(inp)

    assert res_out.dtype == torch.bool
    assert res_out.shape == inp.shape
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.isreal
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_isreal_empty(dtype):
    inp = torch.empty(0, dtype=dtype, device=flag_gems.device)
    assert inp.dtype == dtype and inp.device.type == torch.device(flag_gems.device).type
    ref_inp = utils.to_reference(inp)

    ref_out = torch.isreal(ref_inp)
    res_out = _call_isreal(inp)

    assert res_out.dtype == torch.bool
    assert res_out.numel() == 0
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.isreal
def test_isreal_conj_bit():
    """A lazy conjugate only flips the sign of the imaginary part."""
    base = _make_complex((1024,), torch.complex64, flag_gems.device)
    inp = base.conj()
    assert inp.is_conj()
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.isreal(ref_inp)
    res_out = _call_isreal(inp)

    assert res_out.dtype == torch.bool
    utils.gems_assert_equal(res_out, ref_out)
