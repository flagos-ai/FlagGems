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

import random
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

# ---------------------------------------------------------------------------
# Custom shapes spanning the masked_select kernel's two code paths:
#   - N <= 4096  ->  single-pass (1 block, tl.cumsum)
#   - N >  4096  ->  multi-CTA  (mask_part_sum + write_back)
#
# Shapes are chosen to cover small / boundary / medium / large at even
# intervals so regressions in any regime are caught early.
# ---------------------------------------------------------------------------
CUSTOM_SHAPES = [
    # ---- small: single-pass, BLOCK_SIZE <= 1024 ----
    (16,),  # N = 16
    (128,),  # N = 128
    (512,),  # N = 512
    # ---- boundary: single-pass, BLOCK_SIZE 1024..4096 ----
    (1024,),  # N = 1024
    (2048,),  # N = 2048
    (4096,),  # N = 4096  (max single-pass)
    # ---- above boundary: multi-CTA ----
    (4097,),  # N = 4097  (just above, 1-block degenerate)
    (8192,),  # N = 8192
    (64, 128),  # N = 8192  (2-D variant)
    # ---- large: multi-CTA, multiple blocks ----
    (128, 256),  # N = 32768
    (256, 1024),  # N = 262144
    (1024, 1024),  # N = 1048576
]

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    # One shape from each code path
    THRESHOLD_SHAPE = [(0.5, (16,)), (0.5, (4097,))]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    THRESHOLD_SHAPE = list(
        zip([0.3, 0.5, 0.7] * (len(CUSTOM_SHAPES) // 3 + 1), CUSTOM_SHAPES)
    )

# Make sure every thread has same seed.
random.seed(time.time() // 100)


@pytest.mark.masked_scatter_backward
@pytest.mark.parametrize("threshold, shape", THRESHOLD_SHAPE)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_masked_scatter_backward(shape, dtype, threshold):
    grad_output = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mask = torch.randn(shape, dtype=dtype, device=flag_gems.device) < threshold
    sizes = shape

    ref_grad = utils.to_reference(grad_output)
    ref_mask = utils.to_reference(mask)
    ref_out = torch.ops.aten.masked_scatter_backward(ref_grad, ref_mask, sizes)
    with flag_gems.use_gems():
        res_out = flag_gems.masked_scatter_backward(grad_output, mask, sizes)

    utils.gems_assert_equal(res_out, ref_out)
