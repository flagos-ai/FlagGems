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

if cfg.QUICK_MODE:
    TRIPLET_SHAPES = [(4, 8)]
    TRIPLET_MARGINS = [1.0]
    TRIPLET_P_VALUES = [2.0]
    TRIPLET_REDUCTIONS = [1]
    TRIPLET_SWAP_VALUES = [False]
else:
    TRIPLET_SHAPES = [(4, 8), (32, 128), (128, 256), (1024, 512)]
    TRIPLET_MARGINS = [0.0, 1.0, 2.0]
    TRIPLET_P_VALUES = [1.0, 2.0, 3.0]
    TRIPLET_REDUCTIONS = [0, 1, 2]  # none, mean, sum
    TRIPLET_SWAP_VALUES = [False, True]


@pytest.mark.triplet_margin_loss
@pytest.mark.parametrize("shape", TRIPLET_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("margin", TRIPLET_MARGINS)
@pytest.mark.parametrize("p", TRIPLET_P_VALUES)
@pytest.mark.parametrize("reduction", TRIPLET_REDUCTIONS)
@pytest.mark.parametrize("swap", TRIPLET_SWAP_VALUES)
def test_triplet_margin_loss(shape, dtype, margin, p, reduction, swap):
    anchor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    positive = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    negative = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_anchor = utils.to_reference(anchor)
    ref_positive = utils.to_reference(positive)
    ref_negative = utils.to_reference(negative)

    ref_out = torch.ops.aten.triplet_margin_loss(
        ref_anchor, ref_positive, ref_negative, margin, p, 1e-6, swap, reduction
    )
    res_out = flag_gems.triplet_margin_loss(
        anchor, positive, negative, margin, p, 1e-6, swap, reduction
    )

    # Use dtype and reduction-specific tolerances due to exp2/log2 approximation errors
    # swap=True adds an extra distance computation which increases error
    if dtype == torch.float16:
        atol = 1.5e-2 if reduction == 0 else 5e-3
    elif dtype == torch.bfloat16:
        atol = 2e-2 if reduction == 0 else 1e-2
    else:  # float32
        atol = 5e-4 if reduction == 0 else 1e-4

    # For sum reduction, scale tolerance by number of elements being summed
    if reduction == 2:  # sum
        atol = atol * shape[0]

    utils.gems_assert_close(res_out, ref_out, dtype, atol=atol)
