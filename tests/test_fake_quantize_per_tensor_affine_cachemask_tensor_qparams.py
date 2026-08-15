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

# int8 symmetric quantization range used by the fake-quant tests below.
QUANT_MIN = -128
QUANT_MAX = 127


def _make_inputs(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    scale = torch.tensor(0.1, dtype=torch.float32, device=device)
    zero_point = torch.tensor(0, dtype=torch.int32, device=device)
    fake_quant_enabled = torch.tensor(1, dtype=torch.int64, device=device)
    return inp, scale, zero_point, fake_quant_enabled


@pytest.mark.fake_quantize_per_tensor_affine_cachemask_tensor_qparams
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_fake_quantize_per_tensor_affine_cachemask_tensor_qparams(shape, dtype):
    inp, scale, zero_point, fake_quant_enabled = _make_inputs(
        shape, dtype, flag_gems.device
    )
    ref_inp = utils.to_reference(inp)
    ref_scale = utils.to_reference(scale)
    ref_zero_point = utils.to_reference(zero_point)
    ref_fake_quant_enabled = utils.to_reference(fake_quant_enabled)

    ref_out = torch.ops.aten._fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
        ref_inp,
        ref_scale,
        ref_zero_point,
        ref_fake_quant_enabled,
        QUANT_MIN,
        QUANT_MAX,
    )
    with flag_gems.use_gems():
        res_out = (
            torch.ops.aten._fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
                inp, scale, zero_point, fake_quant_enabled, QUANT_MIN, QUANT_MAX
            )
        )

    utils.gems_assert_close(res_out[0], ref_out[0], dtype)
    utils.gems_assert_equal(res_out[1], ref_out[1])


@pytest.mark.fake_quantize_per_tensor_affine_cachemask_tensor_qparams
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_fake_quantize_per_tensor_affine_cachemask_tensor_qparams_disabled(
    shape, dtype
):
    # When fake_quant_enabled is False, the op is a pass-through and mask is all True.
    inp, scale, zero_point, _ = _make_inputs(shape, dtype, flag_gems.device)
    fake_quant_enabled = torch.tensor(0, dtype=torch.int64, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref_scale = utils.to_reference(scale)
    ref_zero_point = utils.to_reference(zero_point)
    ref_fake_quant_enabled = utils.to_reference(fake_quant_enabled)

    ref_out = torch.ops.aten._fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
        ref_inp,
        ref_scale,
        ref_zero_point,
        ref_fake_quant_enabled,
        QUANT_MIN,
        QUANT_MAX,
    )
    with flag_gems.use_gems():
        res_out = (
            torch.ops.aten._fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
                inp, scale, zero_point, fake_quant_enabled, QUANT_MIN, QUANT_MAX
            )
        )

    utils.gems_assert_close(res_out[0], ref_out[0], dtype)
    utils.gems_assert_equal(res_out[1], ref_out[1])
