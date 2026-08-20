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


def _torch_rms_norm_backward(grad_out, x, normalized_shape, rstd, weight, output_mask):
    """Reference RMS-norm backward computed in float32."""
    ndim_batch = x.ndim - len(normalized_shape)
    reduce_dims = tuple(range(ndim_batch, x.ndim))
    upcast_x = x.to(torch.float32)
    # rstd has shape (M,); broadcast it back over the normalized dims.
    inv_rms = rstd.to(torch.float32).reshape(
        x.shape[:ndim_batch] + (1,) * len(normalized_shape)
    )
    normalized = upcast_x * inv_rms
    numel = int(np.prod(normalized_shape))

    dx = None
    if output_mask[0]:
        g = grad_out.to(torch.float32)
        if weight is not None:
            g = g * weight.to(torch.float32)
        row_sum = (normalized * g).sum(dim=reduce_dims, keepdim=True)
        dx = ((g - normalized / numel * row_sum) * inv_rms).to(x.dtype)

    dw = None
    if output_mask[1] and weight is not None:
        reduce_batch = tuple(range(ndim_batch))
        dw = (grad_out.to(torch.float32) * normalized).sum(dim=reduce_batch)
        dw = dw.reshape(normalized_shape).to(x.dtype)

    return dx, dw


@pytest.mark.fused_rms_norm_backward
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("output_mask", [(True, True), (True, False), (False, True)])
def test_fused_rms_norm_backward(shape, dtype, output_mask):
    """Directly exercise aten._fused_rms_norm_backward for each output_mask."""
    N = shape[1]
    normalized_shape = [N]
    eps = 1e-5

    np.random.seed(0)
    np_inp = np.random.uniform(-0.1, 0.1, shape[:2]).astype(np.float32)
    np_grad = np.random.uniform(-0.01, 0.01, shape[:2]).astype(np.float32)
    np_weight = np.random.uniform(-0.1, 0.1, normalized_shape).astype(np.float32)

    inp = torch.tensor(np_inp, dtype=dtype, device=flag_gems.device)
    grad = torch.tensor(np_grad, dtype=dtype, device=flag_gems.device)
    weight = torch.tensor(np_weight, dtype=dtype, device=flag_gems.device)
    rstd = torch.rsqrt(inp.to(torch.float32).pow(2).mean(dim=-1) + eps)

    ref_dx, ref_dw = _torch_rms_norm_backward(
        utils.to_reference(grad),
        utils.to_reference(inp),
        normalized_shape,
        utils.to_reference(rstd),
        utils.to_reference(weight),
        output_mask,
    )

    with flag_gems.use_gems():
        res_dx, res_dw = torch.ops.aten._fused_rms_norm_backward(
            grad, inp, normalized_shape, rstd, weight, list(output_mask)
        )

    if output_mask[0]:
        utils.gems_assert_close(res_dx, ref_dx, dtype)
    else:
        assert res_dx is None
    if output_mask[1]:
        utils.gems_assert_close(res_dw, ref_dw, dtype, reduce_dim=N)
    else:
        assert res_dw is None


@pytest.mark.fused_rms_norm_backward
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_fused_rms_norm_backward_no_weight(shape, dtype):
    """aten._fused_rms_norm_backward without a weight tensor returns dw=None."""
    N = shape[1]
    normalized_shape = [N]
    eps = 1e-5

    np.random.seed(0)
    np_inp = np.random.uniform(-0.1, 0.1, shape[:2]).astype(np.float32)
    np_grad = np.random.uniform(-0.01, 0.01, shape[:2]).astype(np.float32)

    inp = torch.tensor(np_inp, dtype=dtype, device=flag_gems.device)
    grad = torch.tensor(np_grad, dtype=dtype, device=flag_gems.device)
    rstd = torch.rsqrt(inp.to(torch.float32).pow(2).mean(dim=-1) + eps)

    ref_dx, _ = _torch_rms_norm_backward(
        utils.to_reference(grad),
        utils.to_reference(inp),
        normalized_shape,
        utils.to_reference(rstd),
        None,
        (True, False),
    )

    with flag_gems.use_gems():
        res_dx, res_dw = torch.ops.aten._fused_rms_norm_backward(
            grad, inp, normalized_shape, rstd, None, [True, False]
        )

    assert res_dw is None
    utils.gems_assert_close(res_dx, ref_dx, dtype)
