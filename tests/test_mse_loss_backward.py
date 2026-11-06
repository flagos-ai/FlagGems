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
from flag_gems.ops.mse_loss_backward import mse_loss_backward as generic_backward

from . import accuracy_utils as utils


@pytest.mark.mse_loss_backward
@pytest.mark.parametrize("reduction", [0, 1, 2])
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_mse_loss_backward(shape, dtype, reduction):
    if flag_gems.vendor_name == "metax":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    grad_output = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_grad_output = utils.to_reference(grad_output)
    ref_inp = utils.to_reference(inp)
    ref_target = utils.to_reference(target)

    ref_out = torch.ops.aten.mse_loss_backward(
        ref_grad_output, ref_inp, ref_target, reduction
    )
    res_out = flag_gems.mse_loss_backward(grad_output, inp, target, reduction)
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.mse_loss_backward
@pytest.mark.parametrize("reduction", [0, 1, 2])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("shape", [(4,), (2, 3), (1025,)])
@pytest.mark.parametrize("implementation", ["generic", "dispatch"])
def test_mse_loss_backward_scalar_grad(shape, dtype, reduction, implementation):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = torch.randn_like(inp)
    # A scalar view with known adjacent values exposes incorrect offset loads
    # deterministically, without relying on unallocated device memory.
    storage = torch.tensor([1.0, 7.0, 11.0, 13.0], dtype=dtype, device=flag_gems.device)
    grad_output = storage[0].reshape(())
    ref_out = torch.ops.aten.mse_loss_backward(
        utils.to_reference(grad_output),
        utils.to_reference(inp),
        utils.to_reference(target),
        reduction,
    )
    if implementation == "generic":
        res_out = generic_backward(grad_output, inp, target, reduction)
    else:
        with flag_gems.use_gems():
            res_out = torch.ops.aten.mse_loss_backward(
                grad_output, inp, target, reduction
            )
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
