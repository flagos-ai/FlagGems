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


@pytest.mark.dyn_quant_pack_4bit_weight
# Covers a single full-width block and multiple 32-element quantization groups.
@pytest.mark.parametrize(
    "block_size,in_features,out_features", [(64, 64, 8), (32, 128, 4)]
)
@pytest.mark.parametrize("with_bias", [False, True])
# The packed scale/zero and output representation is float32.
@pytest.mark.parametrize("scale_dtype", [torch.float32])
def test_dyn_quant_pack_4bit_weight(
    block_size, in_features, out_features, with_bias, scale_dtype
):
    weights_cpu = torch.randint(
        0,
        256,
        (out_features, in_features // 2),
        dtype=torch.uint8,
    )
    groups = in_features // block_size
    scales_cpu = torch.randn((out_features, groups), dtype=scale_dtype)
    bias_cpu = torch.randn(out_features, dtype=scale_dtype) if with_bias else None
    # Reference runs on the to_reference device (CPU under --ref=cpu), so the
    # packed output stays there; do not force it onto flag_gems.device.
    expected = torch.ops.aten._dyn_quant_pack_4bit_weight(
        utils.to_reference(weights_cpu),
        utils.to_reference(scales_cpu),
        utils.to_reference(bias_cpu),
        block_size,
        in_features,
        out_features,
    )
    weights = weights_cpu.to(flag_gems.device)
    scales = scales_cpu.to(flag_gems.device)
    bias = None if bias_cpu is None else bias_cpu.to(flag_gems.device)

    with flag_gems.use_gems():
        actual = torch.ops.aten._dyn_quant_pack_4bit_weight(
            weights, scales, bias, block_size, in_features, out_features
        )

    assert actual.dtype == torch.float32
    # Align devices with the quick-cpu reference convention (see test_addmm.py).
    if utils.TO_CPU:
        actual = actual.to("cpu")
    else:
        expected = expected.to(flag_gems.device)
    utils.gems_assert_close(actual, expected, torch.float32)


@pytest.mark.dyn_quant_pack_4bit_weight
def test_dyn_quant_pack_4bit_weight_rejects_float_weights():
    weights = torch.randn((4, 32), device=flag_gems.device)
    scales = torch.randn((4, 1), device=flag_gems.device)
    with flag_gems.use_gems(), pytest.raises(RuntimeError, match="uint8"):
        torch.ops.aten._dyn_quant_pack_4bit_weight(weights, scales, None, 64, 64, 4)
