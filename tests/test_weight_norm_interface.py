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
    FLOAT_DTYPES = [torch.float32]
    DIM_LIST = [-1]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    DIM_LIST = [0, -1, -1]


@pytest.mark.weight_norm_interface
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_weight_norm_interface(shape, dtype, dim):
    if flag_gems.vendor_name == "cambricon":
        torch.manual_seed(42)
        torch.mlu.manual_seed_all(42)
    dim = dim % len(shape)
    v = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    g = torch.randn(shape[dim], dtype=dtype, device=flag_gems.device)
    reduce_size = v.numel() // shape[dim]

    ref_v = utils.to_reference(v, True)
    ref_g = utils.to_reference(g, True)

    ref_w_out, ref_norm_out = torch._weight_norm_interface(ref_v, ref_g, dim)
    with flag_gems.use_gems():
        res_w_out, res_norm_out = torch._weight_norm_interface(v, g, dim)
    utils.gems_assert_close(res_w_out, ref_w_out, dtype, reduce_dim=reduce_size)
    utils.gems_assert_close(
        res_norm_out, ref_norm_out, torch.float32, reduce_dim=reduce_size
    )


@pytest.mark.weight_norm_interface_backward
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_weight_norm_interface_backward(shape, dtype, dim):
    dim = dim % len(shape)
    res_w_grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_v = torch.randn_like(res_w_grad)
    if flag_gems.vendor_name == "kunlunxin":
        if shape == (4096, 256):
            res_v = res_v.uniform_(-0.01, 0.01)
    res_g = torch.randn(shape[dim], dtype=dtype, device=flag_gems.device)

    ref_w_grad = utils.to_reference(res_w_grad, True)
    ref_v = utils.to_reference(res_v, True)
    ref_g = utils.to_reference(res_g, True)
    _, ref_norm = torch._weight_norm_interface(ref_v, ref_g, dim)

    ref_v_grad, ref_g_grad = torch.ops.aten._weight_norm_interface_backward(
        ref_w_grad, ref_v, ref_g, ref_norm, dim
    )
    with flag_gems.use_gems():
        _, res_norm = torch._weight_norm_interface(res_v, res_g, dim)
        res_v_grad, res_g_grad = torch.ops.aten._weight_norm_interface_backward(
            res_w_grad, res_v, res_g, res_norm, dim
        )
    reduce_size = res_v.numel() // shape[dim]
    utils.gems_assert_close(
        res_v_grad, ref_v_grad, dtype, reduce_dim=reduce_size, equal_nan=True
    )
    utils.gems_assert_close(
        res_g_grad, ref_g_grad, dtype, reduce_dim=reduce_size, equal_nan=True
    )
