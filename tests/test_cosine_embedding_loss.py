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
from . import conftest as cfg

# Shapes mirror the sibling hinge_embedding_loss test to exercise the two-stage
# reduction across block counts. Inputs are (N, D); target is (N,).
if cfg.QUICK_MODE:
    # Single tiny shape keeps QUICK_MODE smoke runs fast.
    COSINE_EMBEDDING_LOSS_SHAPES = [(2, 3)]
else:
    # Small/medium/large trio covers single- and multi-block reductions.
    COSINE_EMBEDDING_LOSS_SHAPES = [(2, 3), (128, 256), (512, 512)]


@pytest.mark.cosine_embedding_loss
@pytest.mark.parametrize("shape", COSINE_EMBEDDING_LOSS_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [0, 1, 2])
@pytest.mark.parametrize("margin", [0.0, 0.5])
def test_cosine_embedding_loss(shape, dtype, reduction, margin):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = (
        torch.randint(0, 2, (shape[0],), device=flag_gems.device).to(dtype) * 2
    ) - 1

    # The kernel upcasts to float32 for the dot/norm reductions, so the
    # reference must upcast too — otherwise a pure-fp16 cosine is numerically
    # ill-conditioned near cos==0 and diverges from the (more accurate) result.
    ref_inp1 = utils.to_reference(inp1, upcast=True)
    ref_inp2 = utils.to_reference(inp2, upcast=True)
    ref_target = utils.to_reference(target, upcast=True)
    ref_out = torch.ops.aten.cosine_embedding_loss(
        ref_inp1, ref_inp2, ref_target, margin, reduction
    )

    with flag_gems.use_gems():
        res_out = torch.ops.aten.cosine_embedding_loss(
            inp1, inp2, target, margin, reduction
        )

    utils.gems_assert_close(res_out, ref_out, dtype)
