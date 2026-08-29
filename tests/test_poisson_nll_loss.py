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

SHAPES = [(2, 3)] if cfg.QUICK_MODE else [(2, 3), (128, 256), (512, 512)]


@pytest.mark.poisson_nll_loss
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [0, 1, 2])
@pytest.mark.parametrize("log_input", [True, False])
@pytest.mark.parametrize("full", [False, True])
def test_accuracy_poisson_nll_loss(shape, dtype, reduction, log_input, full):
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if not log_input:
        input = input.abs() + 0.1
    target = torch.randint(0, 5, shape, device=flag_gems.device).to(dtype)
    ref = torch.ops.aten.poisson_nll_loss(
        utils.to_reference(input, upcast=True),
        utils.to_reference(target, upcast=True),
        log_input,
        full,
        1e-8,
        reduction,
    )
    result = flag_gems.poisson_nll_loss(input, target, log_input, full, 1e-8, reduction)
    utils.gems_assert_close(result, ref, dtype, equal_nan=True)


@pytest.mark.poisson_nll_loss
@pytest.mark.parametrize("reduction", [0, 1, 2, 3])
def test_accuracy_poisson_nll_loss_broadcast_noncontiguous(reduction):
    input = torch.randn((7, 5), device=flag_gems.device).T
    target = torch.randint(0, 5, (1, 7), device=flag_gems.device).float()
    ref = torch.ops.aten.poisson_nll_loss(
        utils.to_reference(input),
        utils.to_reference(target),
        True,
        True,
        1e-8,
        reduction,
    )
    result = flag_gems.poisson_nll_loss(input, target, True, True, 1e-8, reduction)
    utils.gems_assert_close(result, ref, torch.float32, equal_nan=True)


@pytest.mark.poisson_nll_loss
@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_accuracy_poisson_nll_loss_empty(reduction):
    input = torch.empty((0, 3), device=flag_gems.device)
    target = torch.empty_like(input)
    ref = torch.ops.aten.poisson_nll_loss(
        utils.to_reference(input),
        utils.to_reference(target),
        True,
        False,
        1e-8,
        reduction,
    )
    result = flag_gems.poisson_nll_loss(input, target, True, False, 1e-8, reduction)
    utils.gems_assert_close(result, ref, torch.float32, equal_nan=True)


@pytest.mark.poisson_nll_loss
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_poisson_nll_loss_boundary(dtype):
    input = torch.tensor(
        [0.0, -0.0, 1e-5, float("inf"), float("nan")],
        dtype=dtype,
        device=flag_gems.device,
    )
    target = torch.tensor([1.0, 0.0, 2.0, 0.0, 1.0], dtype=dtype, device=input.device)
    ref = torch.ops.aten.poisson_nll_loss(
        utils.to_reference(input),
        utils.to_reference(target),
        False,
        True,
        1e-8,
        0,
    )
    result = flag_gems.poisson_nll_loss(input, target, False, True, 1e-8, 0)
    utils.gems_assert_close(result, ref, dtype, equal_nan=True)
