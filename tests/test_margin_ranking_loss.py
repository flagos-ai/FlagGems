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
    MARGIN_RANKING_SHAPES = [(2, 3)]
    MARGIN_RANKING_MARGINS = [0.5]
    MARGIN_RANKING_REDUCTIONS = [1]
else:
    MARGIN_RANKING_SHAPES = [(2, 3), (128, 256), (1024, 256)]
    MARGIN_RANKING_MARGINS = [0.0, 0.5, 1.0]
    MARGIN_RANKING_REDUCTIONS = [0, 1, 2]


@pytest.mark.margin_ranking_loss
@pytest.mark.parametrize("shape", MARGIN_RANKING_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("margin", MARGIN_RANKING_MARGINS)
@pytest.mark.parametrize("reduction", MARGIN_RANKING_REDUCTIONS)
def test_margin_ranking_loss(shape, dtype, margin, reduction):
    input1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    input2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    target = (
        torch.randint(0, 2, shape, device=flag_gems.device, dtype=torch.int8) * 2 - 1
    ).to(dtype)

    ref_input1 = utils.to_reference(input1)
    ref_input2 = utils.to_reference(input2)
    ref_target = utils.to_reference(target)
    ref_out = torch.ops.aten.margin_ranking_loss(
        ref_input1, ref_input2, ref_target, margin, reduction
    )
    with flag_gems.use_gems():
        res_out = torch.ops.aten.margin_ranking_loss(
            input1, input2, target, margin, reduction
        )

    utils.gems_assert_close(res_out, ref_out, dtype)


REDUCTION_MAP = {0: "none", 1: "mean", 2: "sum"}


@pytest.mark.margin_ranking_loss
@pytest.mark.parametrize("shape", MARGIN_RANKING_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("margin", MARGIN_RANKING_MARGINS)
@pytest.mark.parametrize("reduction", MARGIN_RANKING_REDUCTIONS)
def test_margin_ranking_loss_backward(shape, dtype, margin, reduction):
    input1 = torch.randn(
        shape, dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    input2 = torch.randn(
        shape, dtype=dtype, device=flag_gems.device, requires_grad=True
    )

    target = (
        torch.randint(0, 2, shape, device=flag_gems.device, dtype=torch.int8) * 2 - 1
    ).to(dtype)

    # Avoid boundary elements where val = -y*(x1-x2)+margin ≈ 0, since the
    # discontinuous gradient at the boundary can differ between Triton (float32
    # internal compute) and PyTorch native (input dtype compute).
    with torch.no_grad():
        val = -target * (input1 - input2) + margin
        boundary_mask = val.abs() < 0.01
        if boundary_mask.any():
            # Push boundary elements away from zero by adjusting x1
            input1.data[boundary_mask] += 0.1 * target.data[boundary_mask]

    ref_input1 = utils.to_reference(input1)
    ref_input2 = utils.to_reference(input2)
    ref_target = utils.to_reference(target)

    ref_out = torch.nn.functional.margin_ranking_loss(
        ref_input1,
        ref_input2,
        ref_target,
        margin=margin,
        reduction=REDUCTION_MAP[reduction],
    )

    with flag_gems.use_gems():
        res_out = torch.nn.functional.margin_ranking_loss(
            input1,
            input2,
            target,
            margin=margin,
            reduction=REDUCTION_MAP[reduction],
        )

    out_grad = torch.randn_like(res_out)
    ref_grad = utils.to_reference(out_grad)

    ref_in_grad1, ref_in_grad2 = torch.autograd.grad(
        ref_out,
        (ref_input1, ref_input2),
        ref_grad,
    )
    res_in_grad1, res_in_grad2 = torch.autograd.grad(
        res_out,
        (input1, input2),
        out_grad,
    )

    utils.gems_assert_close(res_in_grad1, ref_in_grad1, dtype)
    utils.gems_assert_close(res_in_grad2, ref_in_grad2, dtype)
