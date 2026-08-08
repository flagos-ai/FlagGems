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

import numpy as np
import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    # QUICK_MODE restricts to a single dtype to speed up smoke runs; the full
    # utils.FLOAT_DTYPES set is used otherwise (matches tests/test_rms_norm.py).
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES


@pytest.mark.fused_rms_norm
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_fused_rms_norm(shape, dtype):
    N = shape[1]
    layer_shape = [
        N,
    ]
    np.random.seed(0)
    np_inp = np.random.uniform(-0.1, 0.1, shape[:2]).astype(np.float32)
    np_grad = np.random.uniform(-0.01, 0.01, shape[:2]).astype(np.float32)
    np_weight = np.random.uniform(-0.1, 0.1, layer_shape).astype(np.float32)

    inp = torch.tensor(np_inp, dtype=dtype, device=flag_gems.device, requires_grad=True)
    weight = torch.tensor(
        np_weight, dtype=dtype, device=flag_gems.device, requires_grad=True
    )

    eps = 1e-5

    ref_inp = utils.to_reference(inp)
    ref_weight = utils.to_reference(weight)

    def _torch_rms_norm(x, weight, eps):
        upcast_x = x.to(torch.float32)
        variance = upcast_x.pow(2).mean(-1, keepdim=True)
        inv_rms = torch.rsqrt(variance + eps)
        hidden_states = upcast_x * inv_rms.to(torch.float32)
        hidden_states = hidden_states.to(x.dtype)
        return weight * hidden_states, inv_rms.squeeze(-1)

    ref_out, ref_inv_rms = _torch_rms_norm(ref_inp, weight=ref_weight, eps=eps)
    res_out, res_inv_rms = flag_gems._fused_rms_norm(
        inp, list(layer_shape), weight=weight, eps=eps
    )

    # Check forward outputs
    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(res_inv_rms, ref_inv_rms, torch.float32)

    # Check backward
    res_grad = torch.tensor(
        np_grad, dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    ref_grad = utils.to_reference(res_grad)

    res_in_grad, res_weight_grad = torch.autograd.grad(res_out, (inp, weight), res_grad)
    ref_in_grad, ref_weight_grad = torch.autograd.grad(
        ref_out, (ref_inp, ref_weight), ref_grad
    )

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)
    utils.gems_assert_close(res_weight_grad, ref_weight_grad, dtype, reduce_dim=N)


@pytest.mark.fused_rms_norm
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_fused_rms_norm_no_weight(shape, dtype):
    """Test _fused_rms_norm without weight parameter."""
    N = shape[1]
    layer_shape = [
        N,
    ]
    np.random.seed(0)
    np_inp = np.random.uniform(-0.1, 0.1, shape[:2]).astype(np.float32)

    inp = torch.tensor(np_inp, dtype=dtype, device=flag_gems.device)
    eps = 1e-5

    ref_inp = utils.to_reference(inp)

    def _torch_rms_norm_no_weight(x, eps):
        upcast_x = x.to(torch.float32)
        variance = upcast_x.pow(2).mean(-1, keepdim=True)
        inv_rms = torch.rsqrt(variance + eps)
        hidden_states = upcast_x * inv_rms.to(torch.float32)
        hidden_states = hidden_states.to(x.dtype)
        return hidden_states, inv_rms.squeeze(-1)

    ref_out, ref_inv_rms = _torch_rms_norm_no_weight(ref_inp, eps=eps)
    res_out, res_inv_rms = flag_gems._fused_rms_norm(
        inp, list(layer_shape), weight=None, eps=eps
    )

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(res_inv_rms, ref_inv_rms, torch.float32)


@pytest.mark.fused_rms_norm
def test_fused_rms_norm_aten_dispatch():
    """Ensure torch.ops.aten._fused_rms_norm dispatches to the FlagGems kernel."""
    N = 512
    inp = torch.randn(128, N, dtype=torch.float32, device=flag_gems.device)
    weight = torch.randn(N, dtype=torch.float32, device=flag_gems.device)
    eps = 1e-5

    ref_inp = utils.to_reference(inp)
    ref_weight = utils.to_reference(weight)

    upcast_x = ref_inp.to(torch.float32)
    variance = upcast_x.pow(2).mean(-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + eps)
    ref_out = ref_weight * (upcast_x * inv_rms).to(ref_inp.dtype)
    ref_inv_rms = inv_rms.squeeze(-1)

    with flag_gems.use_gems():
        res_out, res_inv_rms = torch.ops.aten._fused_rms_norm(inp, [N], weight, eps)

    utils.gems_assert_close(res_out, ref_out, torch.float32)
    utils.gems_assert_close(res_inv_rms, ref_inv_rms, torch.float32)
