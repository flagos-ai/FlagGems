# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

SHAPES = [(1, 2), (64, 32), (4, 32, 3)]
FLOAT_DTYPES = [torch.float32] if cfg.QUICK_MODE else utils.FLOAT_DTYPES
DIM_LIST = [-1] if cfg.QUICK_MODE else [0, -1]


def _make_case(shape, dim, dtype):
    torch.manual_seed(0)
    grad_w = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    saved_v = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    saved_g = torch.randn(shape[dim], dtype=dtype, device=flag_gems.device)
    return grad_w, saved_v, saved_g


@pytest.mark.weight_norm_interface_backward
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_weight_norm_interface_backward_values(shape, dim, dtype):
    dim %= len(shape)
    grad_w, saved_v, saved_g = _make_case(shape, dim, dtype)
    ref_grad_w = utils.to_reference(grad_w, True)
    ref_v = utils.to_reference(saved_v, True)
    ref_g = utils.to_reference(saved_g, True)
    _, ref_norms = torch._weight_norm_interface(ref_v.contiguous(), ref_g, dim)
    saved_norms = ref_norms.to(device=saved_v.device, dtype=torch.float32)

    ref_v_grad, ref_g_grad = torch.ops.aten._weight_norm_interface_backward(
        ref_grad_w.contiguous(), ref_v.contiguous(), ref_g, ref_norms, dim
    )
    v_grad, g_grad = flag_gems.weight_norm_interface_backward(
        grad_w, saved_v, saved_g, saved_norms, dim
    )

    reduce_size = saved_v.numel() // shape[dim]
    utils.gems_assert_close(
        v_grad, ref_v_grad, dtype, reduce_dim=reduce_size, equal_nan=True
    )
    utils.gems_assert_close(
        g_grad, ref_g_grad, dtype, reduce_dim=reduce_size, equal_nan=True
    )
    assert v_grad.shape == saved_v.shape
    assert g_grad.shape == saved_g.shape
    assert v_grad.dtype == saved_v.dtype
    assert g_grad.dtype == saved_g.dtype


@pytest.mark.weight_norm_interface_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_weight_norm_interface_backward_noncontiguous(dtype):
    torch.manual_seed(3)
    grad_w = torch.randn((7, 5), dtype=dtype, device=flag_gems.device).transpose(0, 1)
    saved_v = torch.randn((7, 5), dtype=dtype, device=flag_gems.device).transpose(0, 1)
    saved_g = torch.randn((7,), dtype=dtype, device=flag_gems.device)
    dim = 1
    ref_grad_w = utils.to_reference(grad_w, True)
    ref_v = utils.to_reference(saved_v, True)
    ref_g = utils.to_reference(saved_g, True)
    _, ref_norms = torch._weight_norm_interface(ref_v.contiguous(), ref_g, dim)
    saved_norms = ref_norms.to(device=saved_v.device, dtype=torch.float32)

    ref_v_grad, ref_g_grad = torch.ops.aten._weight_norm_interface_backward(
        ref_grad_w.contiguous(), ref_v.contiguous(), ref_g, ref_norms, dim
    )
    v_grad, g_grad = flag_gems.weight_norm_interface_backward(
        grad_w, saved_v, saved_g, saved_norms, dim
    )

    utils.gems_assert_close(v_grad, ref_v_grad, dtype, reduce_dim=5, equal_nan=True)
    utils.gems_assert_close(g_grad, ref_g_grad, dtype, reduce_dim=5, equal_nan=True)
