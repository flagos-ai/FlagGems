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
        ref_inp, ref_mask = ref_inp.cpu(), ref_mask.cpu()
    return torch.where(
        ref_mask.to(torch.bool), ref_inp * scale, torch.zeros_like(ref_inp)
    )


def _assert_masked_scale_close(result, reference, inp, mask, dtype, **kwargs):
    assert result.device == inp.device
    assert result.dtype == dtype and result.shape == inp.shape
    assert result is not inp and result is not mask
    if result.numel():
        assert result.untyped_storage().data_ptr() not in {
            inp.untyped_storage().data_ptr(),
            mask.untyped_storage().data_ptr(),
        }
    if flag_gems.vendor_name == "enflame":
        result = result.cpu()
    utils.gems_assert_close(result, reference, dtype, **kwargs)


@pytest.mark.masked_scale
@pytest.mark.underscore_masked_scale
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("scale", _SCALES)
def test_masked_scale(shape, dtype, scale):
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mask = _make_mask(shape)
    inp_before = utils.to_reference(inp.clone())
    mask_before = utils.to_reference(mask.clone())
    ref_out = _reference(inp, mask, scale)

    res_out = flag_gems._masked_scale(inp, mask, scale)

    assert res_out.shape == inp.shape
    assert res_out.dtype == dtype
    _assert_masked_scale_close(res_out, ref_out, inp, mask, dtype, equal_nan=True)
    utils.gems_assert_equal(inp, inp_before)
    utils.gems_assert_equal(mask, mask_before)


@pytest.mark.masked_scale
@pytest.mark.underscore_masked_scale
def test_masked_scale_noncontiguous():
    torch.manual_seed(0)
    inp = torch.randn(
        (4, 7, 9), dtype=torch.float32, device=flag_gems.device
    ).transpose(0, 2)
    mask = _make_mask((4, 7, 9)).transpose(0, 2)
    assert not inp.is_contiguous()
    assert not mask.is_contiguous()
    ref_out = _reference(inp, mask, 2.0)

    res_out = flag_gems._masked_scale(inp, mask, 2.0)

    _assert_masked_scale_close(res_out, ref_out, inp, mask, torch.float32)


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

    res_out = flag_gems._masked_scale(inp, mask, -0.5)

    _assert_masked_scale_close(
        res_out, ref_out, inp, mask, torch.float32, equal_nan=True
    )
