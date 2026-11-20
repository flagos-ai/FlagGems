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


@pytest.mark.native_batch_norm_legit_no_training
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
def test_native_batch_norm_legit_no_training(shape, dtype, affine):
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

    eps = 1e-5

    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias, True)
    ref_running_mean = utils.to_reference(running_mean, True)
    ref_running_var = utils.to_reference(running_var, True)

    (
        ref_out,
        ref_save_mean,
        ref_save_var,
    ) = torch.ops.aten._native_batch_norm_legit_no_training.default(
        ref_inp,
        ref_weight,
        ref_bias,
        ref_running_mean,
        ref_running_var,
        0.1,
        eps,
    )

    (
        res_out,
        res_save_mean,
        res_save_var,
    ) = flag_gems._native_batch_norm_legit_no_training(
        inp,
        weight,
        bias,
        running_mean,
        running_var,
        0.1,
        eps,
    )

    utils.gems_assert_close(res_out, ref_out, dtype)


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


@pytest.mark.native_batch_norm_legit_no_training
@pytest.mark.native_batch_norm_legit_no_training_backward
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
def test_native_batch_norm_legit_no_training_backward(shape, dtype, affine):
    """Autograd backward of the inference-only batch normalization.

    The candidate entry is ``flag_gems._native_batch_norm_legit_no_training``,
    which wraps the forward in an ``autograd.Function`` and launches
    ``batch_norm_backward`` (train=False) from its ``backward``. The reference is
    ATen's backward evaluated in float64 on CPU, so the fp32 Triton accumulation
    is checked against the exact formula instead of ATen's own low-precision
    accumulation.
    """
    inp, weight, bias, running_mean, running_var, grad_out = _reference_backward(
        shape, dtype, affine
    )
    device = flag_gems.device
    res_inp = inp.to(device).requires_grad_(True)
    res_weight = weight.to(device).requires_grad_(True) if affine else None
    res_bias = bias.to(device).requires_grad_(True) if affine else None
    res_inputs = [res_inp] + ([res_weight, res_bias] if affine else [])

    res_out = flag_gems._native_batch_norm_legit_no_training(
        res_inp,
        res_weight,
        res_bias,
        running_mean.to(device),
        running_var.to(device),
        0.1,
        1e-5,
    )[0]
    res_grads = torch.autograd.grad(
        (res_out,), tuple(res_inputs), (grad_out.to(device),), retain_graph=True
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
    for ref_grad, res_grad in zip(ref_grads, res_grads):
        if ref_grad is None:
            assert res_grad is None
            continue
        utils.gems_assert_close(res_grad, ref_grad, dtype, reduce_dim=reduce_dim)


@pytest.mark.native_batch_norm_legit_no_training
@pytest.mark.native_batch_norm_legit_no_training_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_native_batch_norm_legit_no_training_backward_noncontiguous(dtype):
    """A non-contiguous input keeps its layout contract and still differentiates."""
    shape = (6, 8, 5)
    inp, weight, bias, running_mean, running_var, grad_out = _reference_backward(
        shape, dtype, True
    )
    device = flag_gems.device
    # Channel-major layout: channels are the fastest of the leading pair, and
    # the spatial dim stays contiguous.
    res_inp = inp.to(device).transpose(0, 1).contiguous().transpose(0, 1)
    assert not res_inp.is_contiguous()
    assert res_inp.stride()[2] == 1
    res_inp.requires_grad_(True)
    res_weight = weight.to(device).requires_grad_(True)
    res_bias = bias.to(device).requires_grad_(True)

    res_out = flag_gems._native_batch_norm_legit_no_training(
        res_inp,
        res_weight,
        res_bias,
        running_mean.to(device),
        running_var.to(device),
        0.1,
        1e-5,
    )[0]
    res_grads = torch.autograd.grad(
        (res_out,),
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


@pytest.mark.native_batch_norm_legit_no_training
@pytest.mark.native_batch_norm_legit_no_training_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_native_batch_norm_legit_no_training_backward_input_only(dtype):
    """weight=None differentiates the input only, matching ATen's output_mask."""
    shape = (5, 7, 9)
    inp, _, _, running_mean, running_var, grad_out = _reference_backward(
        shape, dtype, False
    )
    device = flag_gems.device
    res_inp = inp.to(device).requires_grad_(True)
    res_out = flag_gems._native_batch_norm_legit_no_training(
        res_inp,
        None,
        None,
        running_mean.to(device),
        running_var.to(device),
        0.1,
        1e-5,
    )[0]
    (res_grad,) = torch.autograd.grad(
        (res_out,), (res_inp,), (grad_out.to(device),), retain_graph=True
    )
    ref_grad, _, _ = torch.ops.aten.native_batch_norm_backward.default(
        _upcast(grad_out),
        _upcast(inp),
        None,
        _upcast(running_mean),
        _upcast(running_var),
        None,
        None,
        False,
        1e-5,
        [True, False, False],
    )
    utils.gems_assert_close(res_grad, ref_grad, dtype, reduce_dim=shape[0] * shape[2])


@pytest.mark.native_batch_norm_legit_no_training
@pytest.mark.native_batch_norm_legit_no_training_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_native_batch_norm_legit_no_training_backward_saved_stats_dtype(dtype):
    """save_mean/save_var use ATen's accumulation dtype (float32 for fp16/bf16).

    The engine's generated node validates that dtype before dispatching the
    backward through ``native_batch_norm_backward``, so returning the input dtype
    here made the fp16/bf16 backward fail with
    ``Expected save_mean to have type Float but got Half``.
    """
    channel_count = 4
    inp = torch.randn((3, channel_count, 5), dtype=dtype, device=flag_gems.device)
    weight = torch.randn(channel_count, dtype=dtype, device=flag_gems.device)
    bias = torch.randn(channel_count, dtype=dtype, device=flag_gems.device)
    running_mean = torch.randn(channel_count, dtype=dtype, device=flag_gems.device)
    running_var = torch.rand(channel_count, dtype=dtype, device=flag_gems.device) + 0.1
    _, save_mean, save_var = flag_gems._native_batch_norm_legit_no_training(
        inp, weight, bias, running_mean, running_var, 0.1, 1e-5
    )
    ref_save_mean, ref_save_var = (
        torch.ops.aten._native_batch_norm_legit_no_training.default(
            inp, weight, bias, running_mean, running_var, 0.1, 1e-5
        )[1:]
    )
    assert save_mean.dtype == ref_save_mean.dtype
    assert save_var.dtype == ref_save_var.dtype

    # The dtype must also be accepted by the engine's generated backward node.
    res_inp = inp.detach().clone().requires_grad_(True)
    res_weight = weight.detach().clone().requires_grad_(True)
    res_bias = bias.detach().clone().requires_grad_(True)
    out = flag_gems._native_batch_norm_legit_no_training(
        res_inp, res_weight, res_bias, running_mean, running_var, 0.1, 1e-5
    )[0]
    grads = torch.autograd.grad(
        (out,),
        (res_inp, res_weight, res_bias),
        (torch.randn_like(out),),
        retain_graph=True,
    )
    assert all(torch.is_tensor(g) for g in grads)
