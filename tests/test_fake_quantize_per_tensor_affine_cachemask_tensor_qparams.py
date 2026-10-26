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

QUANT_MIN = -128
QUANT_MAX = 127


def _seed():
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)


def _reference(tensor):
    return utils.to_reference(tensor, True).to(tensor.dtype)


def _make_inputs(shape, dtype, enabled=1, noncontiguous=False):
    _seed()
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if noncontiguous:
        inp = inp.T
    scale = torch.tensor(0.1, dtype=torch.float32, device=flag_gems.device)
    zero_point = torch.tensor(0, dtype=torch.int32, device=flag_gems.device)
    fake_quant_enabled = torch.tensor(
        enabled, dtype=torch.int64, device=flag_gems.device
    )
    return inp, scale, zero_point, fake_quant_enabled


@pytest.mark.fake_quantize_per_tensor_affine_cachemask_tensor_qparams
@pytest.mark.underscore_fake_quantize_per_tensor_affine_cachemask_tensor_qparams
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("enabled", [0, 1])
def test_fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
    shape, dtype, enabled
):
    if flag_gems.vendor_name == "hygon" and dtype == torch.bfloat16:
        pytest.skip("Hygon native GPU and CPU fake-quant references lack bfloat16")
    inp, scale, zero_point, fake_quant_enabled = _make_inputs(
        shape, dtype, enabled
    )
    snapshots = tuple(
        tensor.clone() for tensor in (inp, scale, zero_point, fake_quant_enabled)
    )
    ref_out = torch.ops.aten._fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
        _reference(inp),
        utils.to_reference(scale, True).to(scale.dtype),
        utils.to_reference(zero_point, False),
        utils.to_reference(fake_quant_enabled, False),
        QUANT_MIN,
        QUANT_MAX,
    )
    result = flag_gems._fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
        inp, scale, zero_point, fake_quant_enabled, QUANT_MIN, QUANT_MAX
    )

    assert result[0].device == inp.device
    assert result[1].device == inp.device
    assert result[0].shape == inp.shape
    assert result[0].dtype == inp.dtype
    assert result[1].shape == inp.shape
    assert result[1].dtype == torch.bool
    utils.gems_assert_close(result[0], ref_out[0], dtype)
    utils.gems_assert_equal(result[1], ref_out[1])
    for value, snapshot in zip(
        (inp, scale, zero_point, fake_quant_enabled), snapshots
    ):
        assert value.device == snapshot.device
        assert value.dtype == snapshot.dtype and value.shape == snapshot.shape
        utils.gems_assert_equal(value, utils.to_reference(snapshot, False))
    if not enabled:
        utils.gems_assert_equal(result[0], utils.to_reference(inp, False))
        utils.gems_assert_equal(
            result[1], utils.to_reference(torch.ones_like(result[1]), False)
        )


@pytest.mark.fake_quantize_per_tensor_affine_cachemask_tensor_qparams
@pytest.mark.underscore_fake_quantize_per_tensor_affine_cachemask_tensor_qparams
@pytest.mark.parametrize("enabled", [0, 1])
def test_fake_quantize_per_tensor_affine_cachemask_tensor_qparams_layout(enabled):
    inp, scale, zero_point, fake_quant_enabled = _make_inputs(
        (8, 16), torch.float32, enabled, noncontiguous=True
    )
    ref_out = torch.ops.aten._fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
        _reference(inp),
        utils.to_reference(scale, True).to(scale.dtype),
        utils.to_reference(zero_point, False),
        utils.to_reference(fake_quant_enabled, False),
        QUANT_MIN,
        QUANT_MAX,
    )

    result = flag_gems._fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
        inp, scale, zero_point, fake_quant_enabled, QUANT_MIN, QUANT_MAX
    )

    assert result[0].device == inp.device
    assert result[1].device == inp.device
    assert result[0].shape == inp.shape
    assert result[0].dtype == inp.dtype
    assert result[1].shape == inp.shape
    assert result[1].dtype == torch.bool
    utils.gems_assert_close(result[0], ref_out[0], torch.float32)
    utils.gems_assert_equal(result[1], ref_out[1])


@pytest.mark.fake_quantize_per_tensor_affine_cachemask_tensor_qparams
@pytest.mark.underscore_fake_quantize_per_tensor_affine_cachemask_tensor_qparams
def test_fake_quantize_per_tensor_affine_cachemask_tensor_qparams_empty():
    inp, scale, zero_point, fake_quant_enabled = _make_inputs(
        (2, 0, 3), torch.float32
    )
    result = flag_gems._fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
        inp, scale, zero_point, fake_quant_enabled, QUANT_MIN, QUANT_MAX
    )

    assert result[0].device == inp.device
    assert result[1].device == inp.device
    assert result[0].shape == inp.shape
    assert result[0].dtype == inp.dtype
    assert result[1].shape == inp.shape
    assert result[1].dtype == torch.bool
    utils.gems_assert_equal(result[0], utils.to_reference(inp, False))
