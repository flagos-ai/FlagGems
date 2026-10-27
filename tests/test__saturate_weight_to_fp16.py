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

# FP16 representable range
FP16_MAX = 65504.0
FP16_MIN = -65504.0


def reference_saturate_weight_to_fp16(x):
    """Manual reference implementation since torch._saturate_weight_to_fp16 has bugs."""
    return torch.clamp(x, FP16_MIN, FP16_MAX)


@pytest.mark.saturate_weight_to_fp16
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__saturate_weight_to_fp16(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = reference_saturate_weight_to_fp16(ref_inp)
    res_out = flag_gems._saturate_weight_to_fp16(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.saturate_weight_to_fp16
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__saturate_weight_to_fp16_extreme_values(dtype):
    # Test with values that exceed fp16 range
    inp = torch.tensor(
        [
            [65504.0, 65505.0, 70000.0, 100000.0],
            [-65504.0, -65505.0, -70000.0, -100000.0],
            [0.0, 1.0, -1.0, 3.14159],
        ],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(inp)

    ref_out = reference_saturate_weight_to_fp16(ref_inp)
    res_out = flag_gems._saturate_weight_to_fp16(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.saturate_weight_to_fp16
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__saturate_weight_to_fp16_boundary(dtype):
    # Test exact boundary values
    inp = torch.tensor(
        [FP16_MAX, FP16_MIN, FP16_MAX + 1, FP16_MIN - 1, 0.0],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(inp)

    ref_out = reference_saturate_weight_to_fp16(ref_inp)
    res_out = flag_gems._saturate_weight_to_fp16(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.saturate_weight_to_fp16
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__saturate_weight_to_fp16_nan(dtype):
    # Test NaN propagation
    inp = torch.tensor(
        [float("nan"), 1.0, -1.0, float("nan"), 65505.0, float("nan")],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(inp)

    ref_out = reference_saturate_weight_to_fp16(ref_inp)
    res_out = flag_gems._saturate_weight_to_fp16(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
