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


def _reference_backward(shape, dtype, affine, seed=0, eps=1e-5):
    """Native tensors plus the float64 CPU ATen backward of the same inputs."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    channel_count = shape[1]
    inp = torch.randn(shape, generator=generator, dtype=torch.float32).to(dtype)
    weight = (
        torch.randn(channel_count, generator=generator, dtype=torch.float32).to(dtype)
        if affine
        else None
    )
    bias = (
        torch.randn(channel_count, generator=generator, dtype=torch.float32).to(dtype)
        if affine
        else None
    )
    running_mean = torch.randn(
        channel_count, generator=generator, dtype=torch.float32
    ).to(dtype)
    running_var = (
        torch.rand(channel_count, generator=generator, dtype=torch.float32) + 0.1
    ).to(dtype)
    grad_out = torch.randn(shape, generator=generator, dtype=torch.float32).to(dtype)
    return inp, weight, bias, running_mean, running_var, grad_out


def _upcast(tensor):
    if tensor is None:
        return None
    return tensor.detach().to("cpu", torch.float64)


# Shapes reuse the sibling batch_norm test coverage: 2D (C last of 2), 3D,
# 4D with small and large spatial dims, 5D, plus a batch dim above the 1024
# tile boundary.
BN_NO_UPDATE_BACKWARD_SHAPES = [
    (16, 3),
    (32, 32, 32),
    (8, 32, 224, 224),
    (2050, 16, 32, 32),
    (8, 16, 3, 224, 224),
]


@pytest.mark.batch_norm_no_update_backward
@pytest.mark.parametrize("shape", BN_NO_UPDATE_BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("affine", [True, False])
def test_batch_norm_no_update_backward(shape, dtype, affine):
    """Direct backward entry against ATen's engine-derived gradients.

    The reference evaluates ``native_batch_norm_backward`` with ``train=False``
    and empty saved statistics in float64 on CPU: exactly the formula the
    autograd engine derives for ``aten::_batch_norm_no_update``.
    """
    inp, weight, bias, running_mean, running_var, grad_out = _reference_backward(
        shape, dtype, affine
    )
    device = flag_gems.device

    res_grad_input, res_grad_weight, res_grad_bias = (
        flag_gems.batch_norm_no_update_backward(
            grad_out.to(device),
            inp.to(device),
            weight.to(device) if affine else None,
            running_mean.to(device),
            running_var.to(device),
            1e-5,
            output_mask=(True, True, True),
        )
    )

    ref_grads = torch.ops.aten.native_batch_norm_backward.default(
        _upcast(grad_out),
        _upcast(inp),
        _upcast(weight),
        _upcast(running_mean),
        _upcast(running_var),
        None,
        None,
        False,
        1e-5,
        [True, affine, affine],
    )

    reduce_dim = 1
    for dim, size in enumerate(shape):
        if dim != 1:
            reduce_dim *= size
    for ref_grad, res_grad in zip(
        ref_grads, (res_grad_input, res_grad_weight, res_grad_bias)
    ):
        if ref_grad is None:
            # ATen ignores the mask for the parameter gradients when
            # weight=None; our kernel still fills them. Skip the reference
            # comparison and only verify the input gradient below.
            continue
        utils.gems_assert_close(res_grad, ref_grad, dtype, reduce_dim=reduce_dim)


@pytest.mark.batch_norm_no_update_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_batch_norm_no_update_backward_output_mask(dtype):
    """output_mask entries select which gradients are computed, like ATen."""
    # Small plane: exercises the reduction path with a few spatial elements.
    shape = (5, 7, 9)
    inp, weight, _, running_mean, running_var, grad_out = _reference_backward(
        shape, dtype, True
    )
    device = flag_gems.device

    # Only the input gradient.
    gi, gw, gb = flag_gems.batch_norm_no_update_backward(
        grad_out.to(device),
        inp.to(device),
        weight.to(device),
        running_mean.to(device),
        running_var.to(device),
        1e-5,
        output_mask=(True, False, False),
    )
    assert torch.is_tensor(gi)
    assert gw is None
    assert gb is None
    ref_gi, _, _ = torch.ops.aten.native_batch_norm_backward.default(
        _upcast(grad_out),
        _upcast(inp),
        _upcast(weight),
        _upcast(running_mean),
        _upcast(running_var),
        None,
        None,
        False,
        1e-5,
        [True, False, False],
    )
    utils.gems_assert_close(gi, ref_gi, dtype, reduce_dim=shape[0] * shape[2])

    # Only the parameter gradients.
    gi, gw, gb = flag_gems.batch_norm_no_update_backward(
        grad_out.to(device),
        inp.to(device),
        weight.to(device),
        running_mean.to(device),
        running_var.to(device),
        1e-5,
        output_mask=(False, True, True),
    )
    assert gi is None
    assert torch.is_tensor(gw)
    assert torch.is_tensor(gb)
    _, ref_gw, ref_gb = torch.ops.aten.native_batch_norm_backward.default(
        _upcast(grad_out),
        _upcast(inp),
        _upcast(weight),
        _upcast(running_mean),
        _upcast(running_var),
        None,
        None,
        False,
        1e-5,
        [False, True, True],
    )
    utils.gems_assert_close(gw, ref_gw, dtype, reduce_dim=shape[0] * shape[2])
    utils.gems_assert_close(gb, ref_gb, dtype, reduce_dim=shape[0] * shape[2])


@pytest.mark.batch_norm_no_update_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_batch_norm_no_update_backward_noncontiguous(dtype):
    """A channel-major input keeps its layout contract through the backward."""
    # Small plane: exercises the reduction path with a few spatial elements.
    shape = (6, 8, 5)
    inp, weight, _, running_mean, running_var, grad_out = _reference_backward(
        shape, dtype, True
    )
    device = flag_gems.device
    # Channel-major layout: channels are the fastest of the leading pair, and
    # the spatial dim stays contiguous.
    nc_inp = inp.to(device).transpose(0, 1).contiguous().transpose(0, 1)
    assert not nc_inp.is_contiguous()
    assert nc_inp.stride()[2] == 1

    res_grads = flag_gems.batch_norm_no_update_backward(
        grad_out.to(device),
        nc_inp,
        weight.to(device),
        running_mean.to(device),
        running_var.to(device),
        1e-5,
        output_mask=(True, True, True),
    )
    ref_grads = torch.ops.aten.native_batch_norm_backward.default(
        _upcast(grad_out),
        _upcast(inp),
        _upcast(weight),
        _upcast(running_mean),
        _upcast(running_var),
        None,
        None,
        False,
        1e-5,
        [True, True, True],
    )
    reduce_dim = shape[0] * shape[2]
    for ref_grad, res_grad in zip(ref_grads, res_grads):
        utils.gems_assert_close(res_grad, ref_grad, dtype, reduce_dim=reduce_dim)


@pytest.mark.batch_norm_no_update_backward
def test_batch_norm_no_update_backward_requires_running_stats():
    """Missing running statistics is rejected like ATen's eval-mode backward."""
    # Small plane: exercises the reduction path with a few spatial elements.
    shape = (4, 3, 5)
    inp, weight, _, running_mean, _, grad_out = _reference_backward(
        shape, torch.float32, True
    )
    device = flag_gems.device
    with pytest.raises(RuntimeError, match="running_var"):
        flag_gems.batch_norm_no_update_backward(
            grad_out.to(device),
            inp.to(device),
            weight.to(device),
            running_mean.to(device),
            None,
            1e-5,
        )


@pytest.mark.batch_norm_no_update_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_batch_norm_no_update_backward_via_autograd(dtype):
    """End-to-end autograd chain: forward + autograd.grad through FlagGems ops.

    Wraps the FlagGems eval-mode forward in an autograd.Function whose backward
    launches this Triton kernel, then checks the gradients of
    ``torch.autograd.grad`` against ATen's engine-derived formula.
    """
    # Small plane: exercises the reduction path with a few spatial elements.
    shape = (8, 16, 33)
    inp, weight, bias, running_mean, running_var, grad_out = _reference_backward(
        shape, dtype, True
    )
    device = flag_gems.device

    class _BNNoUpdateFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inp, weight, bias, running_mean, running_var):
            out = torch.nn.functional.batch_norm(
                inp,
                running_mean,
                running_var,
                weight,
                bias,
                training=False,
                eps=1e-5,
            )
            ctx.save_for_backward(inp, running_mean, running_var)
            ctx.weight = weight
            ctx.has_weight = weight is not None
            return out

        @staticmethod
        def backward(ctx, grad_output):
            inp, running_mean, running_var = ctx.saved_tensors
            needs = ctx.needs_input_grad
            output_mask = [
                bool(needs[0]),
                bool(ctx.has_weight and needs[1]),
                bool(ctx.has_weight and needs[2]),
            ]
            grads = flag_gems.batch_norm_no_update_backward(
                grad_output,
                inp,
                ctx.weight,
                running_mean,
                running_var,
                1e-5,
                output_mask=output_mask,
            )
            return grads + (None, None)

    res_inp = inp.to(device).requires_grad_(True)
    res_weight = weight.to(device).requires_grad_(True)
    res_bias = bias.to(device).requires_grad_(True)

    out = _BNNoUpdateFunction.apply(
        res_inp, res_weight, res_bias, running_mean.to(device), running_var.to(device)
    )
    res_grads = torch.autograd.grad(
        (out,),
        (res_inp, res_weight, res_bias),
        (grad_out.to(device),),
        retain_graph=True,
    )

    ref_grads = torch.ops.aten.native_batch_norm_backward.default(
        _upcast(grad_out),
        _upcast(inp),
        _upcast(weight),
        _upcast(running_mean),
        _upcast(running_var),
        None,
        None,
        False,
        1e-5,
        [True, True, True],
    )
    reduce_dim = shape[0] * shape[2]
    for ref_grad, res_grad in zip(ref_grads, res_grads):
        utils.gems_assert_close(res_grad, ref_grad, dtype, reduce_dim=reduce_dim)
