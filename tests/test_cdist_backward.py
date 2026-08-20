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

# (batch, n1, dim) shapes covering small/medium/large cases
CDIST_BACKWARD_SHAPES = [(2, 16, 32), (4, 32, 64), (8, 64, 128)]


@pytest.mark.cdist_backward
@pytest.mark.parametrize("shape", CDIST_BACKWARD_SHAPES)
# _cdist_backward uses intermediate fp32 accumulation; only float32 is numerically stable
@pytest.mark.parametrize("dtype", [torch.float32])
def test_cdist_backward(shape, dtype):
    # shape is (batch, n1, dim), n2 is separate
    batch, n1, dim = shape
    n2 = n1 // 2 + 1  # Use different n2 for variety

    res_x1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_x2 = torch.randn(batch, n2, dim, dtype=dtype, device=flag_gems.device)
    res_grad = torch.randn(batch, n1, n2, dtype=dtype, device=flag_gems.device)

    ref_x1 = utils.to_reference(res_x1)
    ref_x2 = utils.to_reference(res_x2)
    ref_grad = utils.to_reference(res_grad)

    # Compute cdist first
    p = 2.0
    ref_cdist = torch.cdist(ref_x1, ref_x2, p=p)
    res_cdist = ref_cdist.clone().to(flag_gems.device)

    ref_out = torch.ops.aten._cdist_backward(ref_grad, ref_x1, ref_x2, p, ref_cdist)
    with flag_gems.use_gems():
        res_out = torch.ops.aten._cdist_backward(res_grad, res_x1, res_x2, p, res_cdist)

    utils.gems_assert_close(res_out, ref_out, dtype)
