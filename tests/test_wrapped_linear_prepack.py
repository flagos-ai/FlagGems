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
    _wrapped_linear_prepack,
    unpack_linear_weight,
)

from . import accuracy_utils as utils
from . import conftest as cfg

SHAPES = [(4, 8)] if cfg.QUICK_MODE else [(0, 7), (5, 0), (4, 8), (33, 65), (128, 256)]


@pytest.mark.wrapped_linear_prepack
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_wrapped_linear_prepack(shape, noncontiguous):
    N, K = shape
    if noncontiguous:
        weight = torch.randn((K, N), dtype=torch.float32, device=flag_gems.device).T
        bias = torch.randn((N * 2,), dtype=torch.float32, device=flag_gems.device)[::2]
    else:
        weight = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
        bias = torch.randn(N, dtype=torch.float32, device=flag_gems.device)
    scale = torch.tensor(0.03125, dtype=torch.float32, device=flag_gems.device)
    zero_point = torch.tensor(-3, dtype=torch.int64, device=flag_gems.device)

    ref_weight = utils.to_reference(weight)
    ref_bias = utils.to_reference(bias)
    ref_quantized = torch.quantize_per_tensor(
        ref_weight, float(scale), int(zero_point), torch.qint8
    ).int_repr()

    packed = _wrapped_linear_prepack(weight, scale, zero_point, bias)
    actual_weight, metadata, actual_bias = unpack_linear_weight(packed, N, K)

    utils.gems_assert_equal(actual_weight, ref_quantized)
    utils.gems_assert_equal(actual_bias, ref_bias)
    assert metadata[0].item() == pytest.approx(float(scale))
    assert metadata[1].item() == pytest.approx(float(zero_point))
    assert metadata[2].item() == pytest.approx(_PACK_MAGIC)
    assert metadata[3].item() == pytest.approx(_PACK_VERSION)


@pytest.mark.wrapped_linear_prepack
def test_wrapped_linear_prepack_rounds_ties_to_even_and_clamps():
    weight = torch.tensor(
        [[-100.0, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 100.0]],
        device=flag_gems.device,
    )
    bias = torch.zeros(1, device=flag_gems.device)
    scale = torch.tensor(0.5, device=flag_gems.device)
    zero_point = torch.tensor(-3, device=flag_gems.device)

    packed = _wrapped_linear_prepack(weight, scale, zero_point, bias)
    actual_weight, _, _ = unpack_linear_weight(packed, 1)
    ref_weight = torch.quantize_per_tensor(
        utils.to_reference(weight), 0.5, -3, torch.qint8
    ).int_repr()

    utils.gems_assert_equal(actual_weight, ref_weight)


@pytest.mark.wrapped_linear_prepack
def test_wrapped_linear_prepack_rejects_invalid_shapes():
    weight = torch.randn((4, 8), device=flag_gems.device)
    bias = torch.randn((4,), device=flag_gems.device)
    scale_tensor = torch.tensor(0.1, device=flag_gems.device)
    zero_point_tensor = torch.tensor(0, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        _wrapped_linear_prepack(weight, scale_tensor.expand(2), zero_point_tensor, bias)
    with pytest.raises(RuntimeError):
        _wrapped_linear_prepack(weight, scale_tensor, zero_point_tensor, bias[:3])
