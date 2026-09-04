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
from flag_gems.ops._batch_norm_with_update import _batch_norm_with_update as gems_bn

from . import accuracy_utils as utils


@pytest.mark.batch_norm_with_update
@pytest.mark.parametrize(
    "shape",
    [
        (16, 3),
        (32, 32, 32),
        (8, 32, 224, 224),
        (2050, 16, 32, 32),
        (8, 16, 3, 224, 224),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("affine", [True, False])
def test_batch_norm_with_update(shape, dtype, affine):
    C = shape[1]
    inp = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)
    weight = (
        torch.randn(size=(C,), dtype=dtype, device=flag_gems.device) if affine else None
    )
    bias = (
        torch.randn(size=(C,), dtype=dtype, device=flag_gems.device) if affine else None
    )

    running_mean = torch.randn(size=(C,), dtype=dtype, device=flag_gems.device)
    running_var = (
        torch.abs(torch.randn(size=(C,), dtype=dtype, device=flag_gems.device)) + 0.1
    )

    momentum = 0.1
    eps = 1e-5

    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias, True)
    ref_running_mean = utils.to_reference(running_mean.clone(), True)
    ref_running_var = utils.to_reference(running_var.clone(), True)

    # Reference: training-mode batch norm updates running stats in place.
    (
        ref_out,
        ref_save_mean,
        ref_save_invstd,
        ref_reserve,
    ) = torch.ops.aten._batch_norm_with_update(
        ref_inp,
        ref_weight,
        ref_bias,
        ref_running_mean,
        ref_running_var,
        momentum,
        eps,
    )

    res_running_mean = running_mean.clone()
    res_running_var = running_var.clone()
    (
        res_out,
        res_save_mean,
        res_save_invstd,
        res_reserve,
    ) = gems_bn(
        inp,
        weight,
        bias,
        res_running_mean,
        res_running_var,
        momentum,
        eps,
    )

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(res_running_mean, ref_running_mean, dtype)
    utils.gems_assert_close(res_running_var, ref_running_var, dtype)
