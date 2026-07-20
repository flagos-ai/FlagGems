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


@pytest.mark.rrelu_with_noise_functional
@pytest.mark.parametrize("shape", [(2, 19, 7), (1024, 1024), (16, 128, 64, 60)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_rrelu_with_noise_functional(shape, dtype):
    # Note: training=True case cannot be accurately tested in FlagGems because
    # PyTorch's reference implementation internally generates random noise,
    # while FlagGems uses the provided noise tensor. Since generator is not
    # supported in FlagGems, we only test training=False (deterministic) case.
    training = False

    # Create input tensor and noise tensor
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    lower, upper = 0.125, 1.0 / 3.0
    # Generate noise uniformly in [lower, upper]
    noise = (
        torch.rand(shape, dtype=dtype, device=flag_gems.device) * (upper - lower)
        + lower
    )

    ref_inp = utils.to_reference(inp)
    ref_noise = utils.to_reference(noise)

    ref_out, ref_noise_out = torch.ops.aten.rrelu_with_noise_functional(
        ref_inp, ref_noise, lower, upper, training, None
    )
    with flag_gems.use_gems():
        res_out, res_noise_out = torch.ops.aten.rrelu_with_noise_functional(
            inp, noise, lower, upper, training, None
        )

    # Compare output tensors
    utils.gems_assert_close(res_out, ref_out, dtype)
    # Compare noise_out tensors
    utils.gems_assert_close(res_noise_out, ref_noise_out, dtype)
