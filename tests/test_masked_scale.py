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

# _masked_scale only supports float32 on most backends.
# CUDA reference does not support float16/bf16 for this private op.
FLOAT_DTYPES = [torch.float32]


@pytest.mark.masked_scale
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_masked_scale(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mask = torch.randint(0, 2, shape, dtype=torch.uint8, device=flag_gems.device)
    scale = 2.0

    ref_inp = utils.to_reference(inp)
    ref_mask = utils.to_reference(mask)
    # aten._masked_scale has no CPU backend; compute reference manually.
    ref_out = torch.where(
        ref_mask.to(torch.bool), ref_inp * scale, torch.zeros_like(ref_inp)
    )

    with flag_gems.use_gems():
        res_out = torch.ops.aten._masked_scale(inp, mask, scale)

    utils.gems_assert_close(res_out, ref_out, dtype)
