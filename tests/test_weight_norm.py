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

# Use smaller shapes to avoid Triton autotuning timeout on large reductions
WEIGHT_NORM_SHAPES = [(1, 2), (4096, 256), (4, 256, 3)]

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    DIM_LIST = [-1]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    DIM_LIST = [0, -1]


@pytest.mark.weight_norm
# @pytest.mark.skip(reason="Issue #2860: fails assertion")
@pytest.mark.parametrize("shape", WEIGHT_NORM_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_weight_norm(shape, dtype, dim):
    if flag_gems.vendor_name == "cambricon":
        torch.manual_seed(42)
        torch.mlu.manual_seed_all(42)

    dim = dim % len(shape)
    v = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    g = torch.randn(
        [1 if i != dim else shape[i] for i in range(v.ndim)],
        dtype=dtype,
        device=flag_gems.device,
        requires_grad=True,
    )
    reduce_size = v.numel() // shape[dim]

    ref_v = utils.to_reference(v, True)
    ref_g = utils.to_reference(g, True)
    ref_w_out = torch._weight_norm(ref_v, ref_g, dim)
    res_w_out = flag_gems.weight_norm(v, g, dim)
    utils.gems_assert_close(res_w_out, ref_w_out, dtype, reduce_dim=reduce_size)

    res_w_grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_w_grad = utils.to_reference(res_w_grad, True)

    ref_v_grad, ref_g_grad = torch.autograd.grad(
        ref_w_out, (ref_v, ref_g), grad_outputs=ref_w_grad
    )
    res_v_grad, res_g_grad = torch.autograd.grad(
        res_w_out, (v, g), grad_outputs=res_w_grad
    )
    utils.gems_assert_close(
        res_v_grad, ref_v_grad, dtype, reduce_dim=reduce_size, equal_nan=True
    )
    utils.gems_assert_close(
        res_g_grad, ref_g_grad, dtype, reduce_dim=reduce_size, equal_nan=True
    )
