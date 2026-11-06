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


@pytest.mark.fake_quantize_per_tensor_affine_cachemask_tensor_qparams
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("zero_point", [-17, 0, 3])
@pytest.mark.parametrize("scale", [0.1, 0.25, 0.3])
@pytest.mark.parametrize("enabled", [0, 1])
def test_fake_quantize_tensor_qparams_rounding(dtype, zero_point, scale, enabled):
    # Exact ties expose rounding after adding an odd zero point. Adjacent fp32
    # values also distinguish reciprocal multiplication from direct division.
    levels = torch.arange(-150, 151, dtype=torch.float32) + 0.5
    ties = levels * scale
    values = torch.cat(
        [
            ties,
            torch.nextafter(ties, torch.full_like(ties, -float("inf"))),
            torch.nextafter(ties, torch.full_like(ties, float("inf"))),
            torch.tensor([-1e4, 1e4]),
        ]
    ).to(dtype)
    # Exercise the contiguous-copy path as well as a non-power-of-two tail.
    inp_cpu = torch.stack([values, -values], dim=1).t()
    scale_cpu = torch.tensor(scale, dtype=torch.float32)
    zero_cpu = torch.tensor(zero_point, dtype=torch.int32)
    enabled_cpu = torch.tensor(enabled, dtype=torch.int64)
    ref = torch.ops.aten._fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
        inp_cpu, scale_cpu, zero_cpu, enabled_cpu, QUANT_MIN, QUANT_MAX
    )
    actual = flag_gems._fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
        inp_cpu.to(flag_gems.device),
        scale_cpu.to(flag_gems.device),
        zero_cpu.to(flag_gems.device),
        enabled_cpu.to(flag_gems.device),
        QUANT_MIN,
        QUANT_MAX,
    )
    torch.testing.assert_close(actual[0].cpu(), ref[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1].cpu(), ref[1], rtol=0, atol=0)


@pytest.mark.fake_quantize_per_tensor_affine_cachemask_tensor_qparams
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("enabled", [0, 1])
def test_fake_quantize_tensor_qparams_extreme_values(dtype, enabled):
    inp = torch.tensor(
        [
            float("nan"),
            float("inf"),
            -float("inf"),
            1e20,
            -1e20,
            2**22 + 0.5,
            -(2**22) - 0.5,
            0.0,
            -0.0,
        ],
        dtype=dtype,
    )
    scale = torch.tensor(0.1, dtype=torch.float32)
    zero = torch.tensor(3, dtype=torch.int32)
    flag = torch.tensor(enabled, dtype=torch.int64)
    expected = torch.ops.aten._fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
        inp, scale, zero, flag, QUANT_MIN, QUANT_MAX
    )
    actual = flag_gems._fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
        inp.to(flag_gems.device),
        scale.to(flag_gems.device),
        zero.to(flag_gems.device),
        flag.to(flag_gems.device),
        QUANT_MIN,
        QUANT_MAX,
    )
    torch.testing.assert_close(
        actual[0].cpu(), expected[0], rtol=0, atol=0, equal_nan=True
    )
    torch.testing.assert_close(actual[1].cpu(), expected[1], rtol=0, atol=0)
    if not enabled:
        bits = torch.int32 if dtype == torch.float32 else torch.int16
        assert torch.equal(actual[0].cpu().view(bits), inp.view(bits))


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
    res_out = flag_gems._fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
        inp, scale, zero_point, fake_quant_enabled, QUANT_MIN, QUANT_MAX
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
    res_out = flag_gems._fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
        inp, scale, zero_point, fake_quant_enabled, QUANT_MIN, QUANT_MAX
    )

    utils.gems_assert_close(res_out[0], ref_out[0], dtype)
    utils.gems_assert_equal(res_out[1], ref_out[1])
