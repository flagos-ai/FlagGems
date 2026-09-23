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
    with flag_gems.use_gems():
        res_out = torch.ops.aten.mse_loss_backward(grad_output, inp, target, reduction)
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
