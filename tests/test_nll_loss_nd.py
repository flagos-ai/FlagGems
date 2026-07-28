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
    NLL_LOSS_ND_SHAPES = [(2, 3, 4, 5)]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    NLL_LOSS_ND_SHAPES = [(2, 32), (16, 50, 128), (4, 3, 5, 5), (2, 10, 8, 16, 16)]

random.seed(time.time() // 100)


@pytest.mark.nll_loss_nd_forward
@pytest.mark.parametrize("reduction", ["mean", "none", "sum"])
@pytest.mark.parametrize("weight", [True, False])
@pytest.mark.parametrize("shape", NLL_LOSS_ND_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("ignore_index", [1, 200, -100])
def test_nll_loss_nd_forward(shape, dtype, ignore_index, reduction, weight):
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
    reduce_dim = 1 if reduction == "none" else target.numel()
    utils.gems_assert_close(
        res_out, ref_out, dtype, reduce_dim=reduce_dim, equal_nan=True
    )


@pytest.mark.nll_loss_nd_backward
@pytest.mark.parametrize("reduction", ["mean", "none", "sum"])
@pytest.mark.parametrize("weight", [True, False])
@pytest.mark.parametrize("shape", NLL_LOSS_ND_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("ignore_index", [1, 200, -100])
def test_nll_loss_nd_backward(shape, dtype, ignore_index, reduction, weight):
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
