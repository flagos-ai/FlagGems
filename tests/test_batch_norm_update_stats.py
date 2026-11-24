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

# Hardcoded shapes: batch_norm_update_stats reduces over N and spatial dims per
# channel, so coverage is driven by the (N, C, spatial) layout rather than total
# element count. The first two entries cover small/3D inputs, the rest mirror
# typical CNN feature-map shapes to exercise varying C vs spatial ratios.
SHAPES = [
    (16, 3, 32),
    (32, 32, 32),
    (8, 32, 224, 224),
    (32, 64, 56, 56),
    (64, 128, 28, 28),
    (16, 256, 14, 14),
]


@pytest.mark.batch_norm_update_stats
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("has_running", [True, False])
def test_batch_norm_update_stats(shape, dtype, has_running):
    C = shape[1]
    input_t = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    if has_running:
        running_mean = torch.randn(C, dtype=torch.float32, device=flag_gems.device)
        running_var = torch.rand(C, dtype=torch.float32, device=flag_gems.device) + 0.1
    else:
        running_mean = None
        running_var = None

    # The reference is the native op on the reference device (CPU under --ref
    # cpu). Build its input in float32, the accumulator dtype of the op: with no
    # running stats the CPU kernel returns statistics rounded to the input dtype,
    # which is too coarse to validate a float32 result for half/bfloat16 inputs.
    ref_input = utils.to_reference(input_t).to(torch.float32)
    ref_running_mean = utils.to_reference(running_mean)
    ref_running_var = utils.to_reference(running_var)

    ref_out = torch.batch_norm_update_stats(
        ref_input,
        ref_running_mean.clone() if ref_running_mean is not None else None,
        ref_running_var.clone() if ref_running_var is not None else None,
        0.1,
    )

    res_out = flag_gems.batch_norm_update_stats(
        input_t,
        running_mean.clone() if running_mean is not None else None,
        running_var.clone() if running_var is not None else None,
        0.1,
    )

    for ref_val, res_val in zip(ref_out, res_out):
        utils.gems_assert_close(res_val, ref_val, torch.float32)
