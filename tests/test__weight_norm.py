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

WEIGHT_NORM_SHAPES = [(1, 2), (4096, 256), (4, 256, 3)]
FLOAT_DTYPES = [torch.float32] if cfg.QUICK_MODE else utils.FLOAT_DTYPES
DIM_LIST = [-1] if cfg.QUICK_MODE else [0, -1]


def _g_shape(shape, dim):
    return tuple(shape[index] if index == dim else 1 for index in range(len(shape)))


@pytest.mark.underscore_weight_norm
@pytest.mark.parametrize("shape", WEIGHT_NORM_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_underscore_weight_norm_forward_and_autograd(shape, dim, dtype):
    torch.manual_seed(0)
    dim %= len(shape)
    v = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    g = torch.randn(
        _g_shape(shape, dim), dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    ref_v = utils.to_reference(v, True)
    ref_g = utils.to_reference(g, True)

    ref_out = torch.ops.aten._weight_norm(ref_v, ref_g, dim)
    result = flag_gems._weight_norm(v, g, dim)
    reduce_size = v.numel() // shape[dim]
    utils.gems_assert_close(result, ref_out, dtype, reduce_dim=reduce_size)
    assert result.shape == v.shape
    assert result.dtype == dtype

    torch.manual_seed(1)
    grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_grad = utils.to_reference(grad, True)
    ref_v_grad, ref_g_grad = torch.autograd.grad(
        ref_out, (ref_v, ref_g), grad_outputs=ref_grad
    )
    v_grad, g_grad = torch.autograd.grad(result, (v, g), grad_outputs=grad)
    utils.gems_assert_close(
        v_grad, ref_v_grad, dtype, reduce_dim=reduce_size, equal_nan=True
    )
    utils.gems_assert_close(
        g_grad, ref_g_grad, dtype, reduce_dim=reduce_size, equal_nan=True
    )


@pytest.mark.underscore_weight_norm
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_underscore_weight_norm_noncontiguous(dtype):
    torch.manual_seed(2)
    v = torch.randn((7, 5), dtype=dtype, device=flag_gems.device).transpose(0, 1)
    g = torch.randn((1, 7), dtype=dtype, device=flag_gems.device)
    ref_v = utils.to_reference(v, True)
    ref_g = utils.to_reference(g, True)

    ref = torch.ops.aten._weight_norm(ref_v, ref_g, 1)
    result = flag_gems._weight_norm(v, g, 1)

    utils.gems_assert_close(result, ref, dtype, reduce_dim=5)
    assert result.shape == v.shape


@pytest.mark.underscore_weight_norm
@pytest.mark.parametrize("dim", [0, 1])
def test_underscore_weight_norm_zero_norm(dim):
    shape = (4, 5)
    v = torch.zeros(shape, dtype=torch.float32, device=flag_gems.device)
    g = torch.arange(1, shape[dim] + 1, dtype=torch.float32, device=flag_gems.device)
    g = g.reshape(_g_shape(shape, dim))
    ref_v = utils.to_reference(v, True)
    ref_g = utils.to_reference(g, True)

    ref = torch.ops.aten._weight_norm(ref_v, ref_g, dim)
    result = flag_gems._weight_norm(v, g, dim)

    utils.gems_assert_close(result, ref, torch.float32, equal_nan=True)
    utils.gems_assert_equal(torch.isfinite(result), torch.isfinite(ref))


@pytest.mark.underscore_weight_norm
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float64])
def test_underscore_weight_norm_rejects_unsupported_dtype(dtype):
    if flag_gems.vendor_name == "kunlunxin" and dtype == torch.float64:
        pytest.skip("Kunlunxin cannot preserve the required float64 input storage")
    v = torch.ones((4, 5), dtype=dtype, device=flag_gems.device)
    g = torch.ones((4, 1), dtype=dtype, device=flag_gems.device)

    with pytest.raises(TypeError):
        flag_gems._weight_norm(v, g, 0)
