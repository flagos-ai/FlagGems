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

# Shapes mirror the sibling soft_margin_loss test to exercise the two-stage
# reduction across block counts.
if cfg.QUICK_MODE:
    # Single tiny shape keeps QUICK_MODE smoke runs fast.
    HINGE_EMBEDDING_LOSS_SHAPES = [(2, 3)]
else:
    # Small/medium/large trio covers single- and multi-block reductions.
    HINGE_EMBEDDING_LOSS_SHAPES = [(2, 3), (128, 256), (512, 512)]


@pytest.mark.hinge_embedding_loss
@pytest.mark.parametrize("shape", HINGE_EMBEDDING_LOSS_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [0, 1, 2])
@pytest.mark.parametrize("margin", [1.0, 0.5])
def test_hinge_embedding_loss(shape, dtype, reduction, margin):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = (torch.randint(0, 2, shape, device=flag_gems.device).to(dtype) * 2) - 1

    ref_inp = utils.to_reference(inp)
    ref_target = utils.to_reference(target)
    ref_out = torch.ops.aten.hinge_embedding_loss(
        ref_inp, ref_target, margin, reduction
    )

    res_out = flag_gems.hinge_embedding_loss(inp, target, margin, reduction)

    utils.gems_assert_close(res_out, ref_out, dtype)
