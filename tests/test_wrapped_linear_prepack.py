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
from flag_gems.ops._wrapped_linear_prepack import (
    _PACK_MAGIC,
    _PACK_VERSION,
    unpack_linear_weight,
)

from . import accuracy_utils as utils
from . import conftest as cfg

SHAPES = [(4, 8)] if cfg.QUICK_MODE else [(0, 7), (5, 0), (4, 8), (33, 65), (128, 256)]


def _quantization_reference(weight):
    # Quantized tensors do not exist on Enflame; the contract's output is a
    # genuine target UInt8 tensor, and only this independent oracle is CPU.
    if flag_gems.vendor_name == "enflame":
        return weight.to("cpu")
    return utils.to_reference(weight)


def _snapshot(weight, scale, zero_point, bias):
    return dict(device=weight.device, shape=tuple(weight.shape),
                # Preserve metadata checks previously coupled to the native
                # reference aliases; freeze mode and expectations before entry.
                native_bias_metadata=(bias.device, bias.dtype) if not utils.TO_CPU else None,
                inputs=[v.detach().cpu().clone() for v in (weight, scale, zero_point, bias)],
                packed_size=16 + 4 * bias.numel() + weight.numel())


def _assert_contract(packed, state, weight, scale, zero_point, bias):
    assert packed.device == state["device"]
    assert weight.device == state["device"]
    if state["native_bias_metadata"] is not None:
        assert (bias.device, bias.dtype) == state["native_bias_metadata"]
    assert packed.dtype == torch.uint8
    assert packed.ndim == 1 and packed.numel() == state["packed_size"]
    assert all(torch.equal(actual.detach().cpu(), expected)
               for actual, expected in zip((weight, scale, zero_point, bias), state["inputs"]))
    assert all(packed.data_ptr() != value.data_ptr()
               for value in (weight, scale, zero_point, bias) if value.numel())


@pytest.mark.wrapped_linear_prepack
@pytest.mark.underscore_wrapped_linear_prepack
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_wrapped_linear_prepack(shape, noncontiguous):
    torch.manual_seed(0)
    N, K = shape
    if noncontiguous:
        weight = torch.randn((K, N), dtype=torch.float32, device=flag_gems.device).T
        bias = torch.randn((N * 2,), dtype=torch.float32, device=flag_gems.device)[::2]
    else:
        weight = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
        bias = torch.randn(N, dtype=torch.float32, device=flag_gems.device)
    scale = torch.tensor(0.03125, dtype=torch.float32, device=flag_gems.device)
    zero_point = torch.tensor(-3, dtype=torch.int64, device=flag_gems.device)

    if flag_gems.vendor_name == "kunlunxin" and not utils.TO_CPU and weight.numel():
        pytest.skip("Kunlunxin native quantize_per_tensor rejects nonempty qint8 input")
    ref_weight = _quantization_reference(weight)
    ref_bias = _quantization_reference(bias).clone()
    ref_quantized = torch.quantize_per_tensor(
        ref_weight, float(scale), int(zero_point), torch.qint8
    ).int_repr()

    state = _snapshot(weight, scale, zero_point, bias)
    packed = flag_gems._wrapped_linear_prepack(weight, scale, zero_point, bias)
    _assert_contract(packed, state, weight, scale, zero_point, bias)
    actual_weight, metadata, actual_bias = unpack_linear_weight(packed, N, K)

    utils.gems_assert_equal(actual_weight.to(ref_quantized.device), ref_quantized)
    utils.gems_assert_equal(actual_bias.to(ref_bias.device), ref_bias)
    assert metadata[0].item() == pytest.approx(float(state["inputs"][1]))
    assert metadata[1].item() == pytest.approx(float(state["inputs"][2]))
    assert metadata[2].item() == pytest.approx(_PACK_MAGIC)
    assert metadata[3].item() == pytest.approx(_PACK_VERSION)


@pytest.mark.wrapped_linear_prepack
@pytest.mark.underscore_wrapped_linear_prepack
def test_wrapped_linear_prepack_rounds_ties_to_even_and_clamps():
    if flag_gems.vendor_name == "ascend" and not utils.TO_CPU:
        pytest.skip("Ascend native quantize_per_tensor rounds after adding zero_point")
    weight = torch.tensor(
        [[-100.0, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 100.0]],
        device=flag_gems.device,
    )
    bias = torch.zeros(1, device=flag_gems.device)
    scale = torch.tensor(0.5, device=flag_gems.device)
    zero_point = torch.tensor(-3, device=flag_gems.device)

    if flag_gems.vendor_name == "kunlunxin" and not utils.TO_CPU:
        pytest.skip("Kunlunxin native quantize_per_tensor rejects nonempty qint8 input")
    ref_weight = torch.quantize_per_tensor(
        _quantization_reference(weight), 0.5, -3, torch.qint8
    ).int_repr()
    state = _snapshot(weight, scale, zero_point, bias)
    packed = flag_gems._wrapped_linear_prepack(weight, scale, zero_point, bias)
    _assert_contract(packed, state, weight, scale, zero_point, bias)
    actual_weight, _, _ = unpack_linear_weight(packed, 1)

    utils.gems_assert_equal(actual_weight.to(ref_weight.device), ref_weight)


@pytest.mark.wrapped_linear_prepack
@pytest.mark.underscore_wrapped_linear_prepack
def test_wrapped_linear_prepack_rejects_invalid_shapes():
    torch.manual_seed(0)
    weight = torch.randn((4, 8), device=flag_gems.device)
    bias = torch.randn((4,), device=flag_gems.device)
    scale_tensor = torch.tensor(0.1, device=flag_gems.device)
    zero_point_tensor = torch.tensor(0, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems._wrapped_linear_prepack(
            weight, scale_tensor.expand(2), zero_point_tensor, bias
        )
    with pytest.raises(RuntimeError):
        flag_gems._wrapped_linear_prepack(
            weight, scale_tensor, zero_point_tensor, bias[:3]
        )
