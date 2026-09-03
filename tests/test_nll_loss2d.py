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

# reduction: 0=none, 1=mean, 2=sum
_REDUCTION = {"none": 0, "mean": 1, "sum": 2}


@pytest.mark.nll_loss2d
@pytest.mark.parametrize("reduction", ["mean", "none", "sum"])
@pytest.mark.parametrize("weight", [True, False])
@pytest.mark.parametrize("shape", [(2, 4, 4, 8), (3, 6, 8, 8)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("ignore_index", [1, 200, -100])
def test_nll_loss2d(shape, dtype, ignore_index, reduction, weight):
    # torch.nn.functional.nll_loss on 4D input dispatches to aten.nll_loss2d_forward,
    # so call aten.nll_loss2d directly to exercise the FlagGems dispatch for this op.
    N, C, H, W = shape
    reduction_val = _REDUCTION[reduction]

    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_target = torch.randint(0, C, (N, H, W), device=flag_gems.device)
    if weight:
        res_weight = torch.randn(C, dtype=dtype, device=flag_gems.device)
    else:
        res_weight = None

    ref_inp = utils.to_reference(res_inp, True)
    ref_target = utils.to_reference(res_target)
    ref_weight = utils.to_reference(res_weight, True)

    ref_out = torch.ops.aten.nll_loss2d(
        ref_inp, ref_target, ref_weight, reduction_val, ignore_index
    )
    with flag_gems.use_gems():
        res_out = torch.ops.aten.nll_loss2d(
            res_inp, res_target, res_weight, reduction_val, ignore_index
        )

    reduce_dim = 1 if reduction == "none" else res_target.numel()
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=reduce_dim)
