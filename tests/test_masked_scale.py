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


FLOAT_DTYPES = [torch.float32]
_SHAPES = [*utils.POINTWISE_SHAPES, (2, 0, 3)]
_SCALES = [-1.0, 0.0, 0.5, 2.0]


def _make_mask(shape):
    if flag_gems.vendor_name == "cambricon":
        return torch.randint(0, 2, shape, dtype=torch.uint8, device="cpu").to(
            flag_gems.device
        )
    return torch.randint(0, 2, shape, dtype=torch.uint8, device=flag_gems.device)


def _reference(inp, mask, scale):
    ref_inp = utils.to_reference(inp, True)
    ref_mask = utils.to_reference(mask, False)
    if flag_gems.vendor_name == "enflame":
        result = ref_inp.cpu() * ref_mask.cpu().to(ref_inp.dtype) * scale
        return result.to(ref_inp.device)
    return ref_inp * ref_mask.to(ref_inp.dtype) * scale


def _capture_inputs(inp, mask):
    return [
        (x, x.device, x.dtype, tuple(x.shape), x.stride(), x.storage_offset(),
         x.data_ptr(), utils.to_reference(x.clone()))
        for x in (inp, mask)
    ]


def _assert_result_and_inputs(result, saved):
    assert isinstance(result, torch.Tensor)
    assert result.device == saved[0][1]
    assert result.dtype == saved[0][2]
    assert tuple(result.shape) == saved[0][3]
    for inp, device, dtype, shape, stride, offset, pointer, before in saved:
        assert inp.device == device and inp.dtype == dtype
        assert tuple(inp.shape) == shape and inp.stride() == stride
        assert inp.storage_offset() == offset and inp.data_ptr() == pointer
        assert result is not inp
        if result.numel() and inp.numel():
            assert result.untyped_storage().data_ptr() != inp.untyped_storage().data_ptr()
        # Byte comparison preserves readonly NaN payloads and signed zeros too.
        utils.gems_assert_equal(
            inp.contiguous().reshape(-1).view(torch.uint8),
            before.contiguous().reshape(-1).view(torch.uint8)
        )


@pytest.mark.masked_scale
@pytest.mark.underscore_masked_scale
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("scale", _SCALES)
def test_masked_scale(shape, dtype, scale):
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mask = _make_mask(shape)
    saved_abi = _capture_inputs(inp, mask)
    inp_before, mask_before = (x[-1] for x in saved_abi)
    ref_out = _reference(inp, mask, scale)

    res_out = flag_gems._masked_scale(inp, mask, scale)
    _assert_result_and_inputs(res_out, saved_abi)

    assert res_out.shape == inp.shape
    assert res_out.dtype == dtype
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
    utils.gems_assert_equal(inp, inp_before)
    utils.gems_assert_equal(mask, mask_before)


@pytest.mark.masked_scale
@pytest.mark.underscore_masked_scale
def test_masked_scale_noncontiguous():
    torch.manual_seed(0)
    inp = torch.randn((4, 7, 9), dtype=torch.float32, device=flag_gems.device).transpose(0, 2)
    mask = _make_mask((4, 7, 9)).transpose(0, 2)
    assert not inp.is_contiguous()
    assert not mask.is_contiguous()
    ref_out = _reference(inp, mask, 2.0)
    saved_abi = _capture_inputs(inp, mask)

    res_out = flag_gems._masked_scale(inp, mask, 2.0)
    _assert_result_and_inputs(res_out, saved_abi)

    utils.gems_assert_close(res_out, ref_out, torch.float32)


@pytest.mark.masked_scale
@pytest.mark.underscore_masked_scale
def test_masked_scale_special_values():
    inp = torch.tensor(
        [float("nan"), float("inf"), -float("inf"), -0.0, 0.0, 1.0],
        dtype=torch.float32,
        device=flag_gems.device,
    )
    mask = torch.tensor([1, 1, 1, 0, 1, 0], dtype=torch.uint8, device=flag_gems.device)
    ref_out = _reference(inp, mask, -0.5)
    saved_abi = _capture_inputs(inp, mask)

    res_out = flag_gems._masked_scale(inp, mask, -0.5)
    _assert_result_and_inputs(res_out, saved_abi)

    utils.gems_assert_close(res_out, ref_out, torch.float32, equal_nan=True)


@pytest.mark.masked_scale
@pytest.mark.underscore_masked_scale
@pytest.mark.parametrize("scale", _SCALES)
def test_masked_scale_native_mask_values(scale):
    inp = torch.tensor(
        [float("nan"), float("inf"), -float("inf"), -0.0, 0.0, 1.0, -2.0, 0.5],
        dtype=torch.float32, device=flag_gems.device,
    )
    mask = torch.tensor([0, 0, 0, 0, 0, 0, 2, 255], dtype=torch.uint8, device=inp.device)
    ref_out = _reference(inp, mask, scale)
    saved_abi = _capture_inputs(inp, mask)
    res_out = flag_gems._masked_scale(inp, mask, scale)
    _assert_result_and_inputs(res_out, saved_abi)
    utils.gems_assert_close(res_out, ref_out, torch.float32, equal_nan=True)
    result_cpu, reference_cpu = res_out.detach().cpu(), ref_out.detach().cpu()
    zeros = reference_cpu == 0
    assert torch.equal(torch.signbit(result_cpu)[zeros], torch.signbit(reference_cpu)[zeros])
