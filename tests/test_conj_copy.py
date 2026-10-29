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


def _conj_copy_input(real, imag, dtype):
    if flag_gems.vendor_name == "ascend" and dtype == torch.complex32:
        return torch.complex(real.cpu(), imag.cpu()).to(dtype).to(real.device)
    return torch.complex(real, imag).to(dtype)


def _conj_copy_reference(inp):
    if (
        flag_gems.vendor_name == "ascend"
        and utils.TO_CPU
        and inp.dtype == torch.complex32
    ):
        # Transfer physical values, then restore the logical conjugate view on CPU.
        value = inp.conj() if inp.is_conj() else inp
        reference = value.cpu()
        return reference.conj() if inp.is_conj() else reference
    return utils.to_reference(inp)


@pytest.mark.underscore_conj_copy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
# _conj_copy only operates on complex dtypes (FLOAT_DTYPES/INT_DTYPES not applicable)
@pytest.mark.parametrize("dtype", utils.COMPLEX_DTYPES)
def test_conj_copy(shape, dtype):
    if (
        flag_gems.vendor_name == "ascend"
        and not utils.TO_CPU
        and dtype == torch.complex32
    ):
        pytest.skip("Ascend native conjugate copy does not support complex32")
    # Create complex input from independent real/imaginary parts.
    real = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    imag = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    inp = _conj_copy_input(real, imag, dtype)
    ref_inp = _conj_copy_reference(inp)

    ref_out = torch._conj_copy(ref_inp)
    expected_device, expected_dtype = inp.device, inp.dtype
    expected_shape, pointer = tuple(inp.shape), inp.data_ptr()
    expected_stride, expected_conj = inp.stride(), inp.is_conj()
    physical = inp.conj() if expected_conj else inp
    input_bytes = physical.cpu().contiguous().reshape(-1).view(torch.uint8).clone()
    res_out = flag_gems._conj_copy(inp)
    assert isinstance(res_out, torch.Tensor)
    assert res_out.device == expected_device and res_out.dtype == expected_dtype
    assert tuple(res_out.shape) == expected_shape
    assert not res_out.is_conj()
    if res_out.numel():
        assert res_out.data_ptr() != pointer
    assert (
        inp.device,
        inp.dtype,
        tuple(inp.shape),
        inp.stride(),
        inp.data_ptr(),
        inp.is_conj(),
    ) == (
        expected_device,
        expected_dtype,
        expected_shape,
        expected_stride,
        pointer,
        expected_conj,
    )
    physical = inp.conj() if inp.is_conj() else inp
    assert torch.equal(
        physical.cpu().contiguous().reshape(-1).view(torch.uint8), input_bytes
    )

    assert ref_out.dtype == dtype
    comparison_dtype = torch.complex64 if dtype == torch.complex32 else dtype
    torch.testing.assert_close(
        res_out.cpu().to(comparison_dtype),
        ref_out.cpu().to(comparison_dtype),
        atol=1e-4,
        rtol=flag_gems.testing.RESOLUTION[dtype],
    )


@pytest.mark.conj_copy_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.COMPLEX_DTYPES)
def test_conj_copy_out(shape, dtype):
    real = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    imag = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    inp = torch.complex(real, imag).to(dtype)
    out = torch.empty_like(inp)
    ref_inp = utils.to_reference(inp)
    ref_out = torch.empty_like(ref_inp)

    torch.ops.aten._conj_copy.out(ref_inp, out=ref_out)
    flag_gems._conj_copy_out(inp, out=out)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.underscore_conj_copy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.COMPLEX_DTYPES)
def test_conj_copy_with_conj_bit(shape, dtype):
    if (
        flag_gems.vendor_name == "ascend"
        and not utils.TO_CPU
        and dtype == torch.complex32
    ):
        pytest.skip("Ascend native conjugate copy does not support complex32")
    # _conj_copy of an already-conjugated tensor should yield the original
    # values: conj(conj(x)) == x.
    real = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    imag = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    base = _conj_copy_input(real, imag, dtype)
    inp = base.conj()  # logical conj view, conj bit set
    ref_inp = _conj_copy_reference(inp)

    ref_out = torch._conj_copy(ref_inp)
    expected_device, expected_dtype = inp.device, inp.dtype
    expected_shape, pointer = tuple(inp.shape), inp.data_ptr()
    expected_stride, expected_conj = inp.stride(), inp.is_conj()
    physical = inp.conj() if expected_conj else inp
    input_bytes = physical.cpu().contiguous().reshape(-1).view(torch.uint8).clone()
    res_out = flag_gems._conj_copy(inp)
    assert isinstance(res_out, torch.Tensor)
    assert res_out.device == expected_device and res_out.dtype == expected_dtype
    assert tuple(res_out.shape) == expected_shape
    assert not res_out.is_conj()
    if res_out.numel():
        assert res_out.data_ptr() != pointer
    assert (
        inp.device,
        inp.dtype,
        tuple(inp.shape),
        inp.stride(),
        inp.data_ptr(),
        inp.is_conj(),
    ) == (
        expected_device,
        expected_dtype,
        expected_shape,
        expected_stride,
        pointer,
        expected_conj,
    )
    physical = inp.conj() if inp.is_conj() else inp
    assert torch.equal(
        physical.cpu().contiguous().reshape(-1).view(torch.uint8), input_bytes
    )

    assert ref_out.dtype == dtype
    comparison_dtype = torch.complex64 if dtype == torch.complex32 else dtype
    torch.testing.assert_close(
        res_out.cpu().to(comparison_dtype),
        ref_out.cpu().to(comparison_dtype),
        atol=1e-4,
        rtol=flag_gems.testing.RESOLUTION[dtype],
    )
