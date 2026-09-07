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

import numpy as np
import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES

random.seed(time.time() // 100)


@pytest.mark.nll_loss_backward
@pytest.mark.parametrize("reduction", ["mean", "none", "sum"])
@pytest.mark.parametrize("weight", [True, False])
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("ignore_index", [1, 200, -100])
def test_nll_loss_backward(shape, dtype, ignore_index, reduction, weight):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        np.random.seed(0)
        random.seed(0)

    dim = 1
    target_shape = list(shape)
    del target_shape[dim]

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    target = torch.randint(0, shape[dim], target_shape, device=flag_gems.device)
    if weight:
        weight = torch.randn(shape[dim], dtype=dtype, device=flag_gems.device)
    else:
        weight = None
    ref_inp = utils.to_reference(inp, True)
    ref_target = utils.to_reference(target)
    ref_weight = utils.to_reference(weight, True)

    ref_out = torch.nn.functional.nll_loss(
        ref_inp, ref_target, ref_weight, reduction=reduction, ignore_index=ignore_index
    )
    with flag_gems.use_gems():
        res_out = torch.nn.functional.nll_loss(
            inp, target, weight, reduction=reduction, ignore_index=ignore_index
        )

    out_grad = torch.randn_like(res_out)
    ref_grad = utils.to_reference(out_grad, True)
    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)

    with flag_gems.use_gems():
        (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=shape[dim])


@pytest.mark.nll_loss_backward
@pytest.mark.parametrize("reduction", ["mean", "none", "sum"])
@pytest.mark.parametrize("weight", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("ignore_index", [-100, 4])
def test_nll_loss_backward_ignore_index_out_of_range(
    dtype, ignore_index, reduction, weight
):
    # ignore_index may be outside [0, C) -- torch's own default is -100 -- and target
    # then really does hold that sentinel on the ignored rows. The backward kernel must
    # not derive a store address from it: with C = 4 and ignore_index = 4, ignored row n
    # would address n * C + 4, which is row n + 1's own gradient cell.
    N, C = 6, 4
    valid = [0, 1, 3]
    target = torch.tensor(
        [ignore_index, valid[0], valid[1], ignore_index, valid[0], valid[2]],
        device=flag_gems.device,
    )

    inp = torch.randn((N, C), dtype=dtype, device=flag_gems.device, requires_grad=True)
    if weight:
        # strictly positive so the 'mean' reduction's total weight cannot land near
        # zero and make the comparison flaky
        weight = torch.rand(C, dtype=dtype, device=flag_gems.device) + 0.5
    else:
        weight = None

    ref_inp = utils.to_reference(inp, True)
    ref_target = utils.to_reference(target)
    ref_weight = utils.to_reference(weight, True)

    ref_out = torch.nn.functional.nll_loss(
        ref_inp, ref_target, ref_weight, reduction=reduction, ignore_index=ignore_index
    )
    with flag_gems.use_gems():
        res_out = torch.nn.functional.nll_loss(
            inp, target, weight, reduction=reduction, ignore_index=ignore_index
        )

    out_grad = torch.randn_like(res_out)
    ref_grad = utils.to_reference(out_grad, True)
    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)

    with flag_gems.use_gems():
        (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=C)
