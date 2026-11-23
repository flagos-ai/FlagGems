# Copyright 2026, The FlagOS Contributors.
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
    # Quick mode keeps one tiny 2-D inner case for the smoke run.
    SAFE_SOFTMAX_BACKWARD_SHAPES = [((1, 8), 1)]
else:
    # Representative shapes covering small and reduction-heavy 2-D/3-D cases
    # along both an inner (dim=-1) and a non-inner reduction axis.
    SAFE_SOFTMAX_BACKWARD_SHAPES = [
        ((1, 8), 1),
        ((4096, 256), 1),
        ((8, 16, 32), 2),
        ((32, 64, 64), 1),
        ((128, 256), 0),
    ]


def _reference_safe_softmax_backward(grad_output, output, dim, input_dtype):
    # ATen has no dedicated aten::_safe_softmax_backward: the autograd formula
    # of _safe_softmax routes through _softmax_backward_data, so the reference
    # below is that op itself -- an FP64 upcast oracle keeps the comparison
    # independent of either Triton implementation.
    return torch.ops.aten._softmax_backward_data(grad_output, output, dim, input_dtype)


def _make_consistent_pair(shape, dim, dtype, device):
    """Forward output and upstream gradient from a real safe_softmax pass.

    Rows of all ``-inf`` (the ``safe`` degenerate rows) produce an output of
    exactly zero, and their gradient must come out exactly zero too -- the
    worst trap of softmax backward kernels is letting the degenerate rows leak the invalid value.
    """
    inp = torch.rand(shape, dtype=dtype, device=device) * 12.0 - 6.0
    flat = inp.reshape(-1, shape[-1]) if inp.ndim > 1 else inp.reshape(1, -1)
    if flat.shape[0] >= 8:
        for i in range(flat.shape[0] // 8):
            r = (i * 7) % flat.shape[0]
            flat[r, :] = float("-inf")
    inp = flat.reshape(shape).contiguous()
    output = torch.ops.aten._safe_softmax(inp, dim)
    grad_output = torch.randn(shape, dtype=dtype, device=device)
    return grad_output, output


@pytest.mark.safe_softmax_backward
@pytest.mark.parametrize("shape_dim", SAFE_SOFTMAX_BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_safe_softmax_backward(shape_dim, dtype):
    shape, dim = shape_dim
    grad_output, output = _make_consistent_pair(shape, dim, dtype, flag_gems.device)

    ref_grad_output = utils.to_reference(grad_output, True)
    ref_output = utils.to_reference(output, True)
    ref_grad_input = _reference_safe_softmax_backward(
        ref_grad_output, ref_output, dim, ref_grad_output.dtype
    )

    res_grad_input = flag_gems.safe_softmax_backward(
        grad_output, output, dim, output.dtype
    )

    utils.gems_assert_close(
        res_grad_input, ref_grad_input, dtype, reduce_dim=shape[dim]
    )


@pytest.mark.safe_softmax_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_safe_softmax_backward_all_neg_inf_rows(dtype):
    """Rows of all -inf forward to exactly 0, and their gradient must be
    exactly 0 -- never the invalid value (the "safe" contract of safe_softmax)."""
    shape, dim = (64, 32), 1
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 12.0 - 6.0
    inp[3, :] = float("-inf")
    inp[17, :] = float("-inf")
    output = torch.ops.aten._safe_softmax(inp, dim)
    assert torch.equal(output[3], torch.zeros(32, dtype=dtype, device=flag_gems.device))
    grad_output = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    res_grad_input = flag_gems.safe_softmax_backward(grad_output, output, dim, dtype)
    assert torch.equal(
        res_grad_input[3], torch.zeros(32, dtype=dtype, device=flag_gems.device)
    )
    assert torch.equal(
        res_grad_input[17], torch.zeros(32, dtype=dtype, device=flag_gems.device)
    )

    ref_grad_output = utils.to_reference(grad_output, True)
    ref_output = utils.to_reference(output, True)
    ref_grad_input = _reference_safe_softmax_backward(
        ref_grad_output, ref_output, dim, ref_grad_output.dtype
    )
    utils.gems_assert_close(
        res_grad_input, ref_grad_input, dtype, reduce_dim=shape[dim]
    )


@pytest.mark.safe_softmax_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_safe_softmax_backward_non_contiguous(dtype):
    """Non-contiguous inputs must be normalised by the wrapper, matching
    the wrapper contract rather than the kernel's contiguous fast path."""
    shape, dim = (64, 32), 1
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 12.0 - 6.0
    output = torch.ops.aten._safe_softmax(inp, dim)
    grad_output = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    strided_grad = grad_output.transpose(0, 1).contiguous().transpose(0, 1)
    assert not strided_grad.is_contiguous()
    strided_out = output.transpose(0, 1).contiguous().transpose(0, 1)
    assert not strided_out.is_contiguous()

    ref_grad_output = utils.to_reference(strided_grad, True)
    ref_output = utils.to_reference(strided_out, True)
    ref_grad_input = _reference_safe_softmax_backward(
        ref_grad_output, ref_output, dim, ref_grad_output.dtype
    )

    res_grad_input = flag_gems.safe_softmax_backward(
        strided_grad, strided_out, dim, dtype
    )

    utils.gems_assert_close(
        res_grad_input, ref_grad_input, dtype, reduce_dim=shape[dim]
    )


@pytest.mark.safe_softmax_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_safe_softmax_backward_empty(dtype):
    grad_output = torch.empty((0, 4), dtype=dtype, device=flag_gems.device)
    output = torch.empty((0, 4), dtype=dtype, device=flag_gems.device)
    res_grad_input = flag_gems.safe_softmax_backward(grad_output, output, 1, dtype)
    assert res_grad_input.shape == (0, 4)
    assert res_grad_input.dtype == dtype


@pytest.mark.safe_softmax_backward
@pytest.mark.parametrize("shape_dim", SAFE_SOFTMAX_BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_safe_softmax_backward_autograd(shape_dim, dtype):
    """The bridge is reachable from the dispatched FlagGems ``_safe_softmax``
    forward: the gradient of the FlagGems forward must flow through the
    FlagGems backward kernel instead of the native chain."""
    shape, dim = shape_dim
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 12.0 - 6.0
    flat = inp.reshape(-1, shape[-1]) if inp.ndim > 1 else inp.reshape(1, -1)
    if flat.shape[0] >= 8:
        flat[0, :] = float("-inf")
    inp = flat.reshape(shape).contiguous()

    ref_inp = utils.to_reference(inp, True).requires_grad_(True)
    ref_out = torch.ops.aten._safe_softmax(ref_inp, dim)
    # The reference graph lives on the reference device (CPU under --ref=cpu,
    # upcast to fp64); the FlagGems bridge runs on CUDA and needs the upstream
    # gradient on CUDA, so keep separate copies instead of reusing one tensor.
    cuda_grad_output = torch.randn(inp.shape, dtype=dtype, device=flag_gems.device)
    ref_grad_output = utils.to_reference(cuda_grad_output, True)
    (ref_grad_input,) = torch.autograd.grad(ref_out, (ref_inp,), (ref_grad_output,))

    from flag_gems.ops.safe_softmax_backward import _SafeSoftmaxFunction

    res_inp = inp.clone().requires_grad_(True)
    res_out = _SafeSoftmaxFunction.apply(res_inp, dim)
    (res_grad_input,) = torch.autograd.grad(res_out, (res_inp,), (cuda_grad_output,))

    utils.gems_assert_close(
        res_grad_input, ref_grad_input, dtype, reduce_dim=shape[dim]
    )


@pytest.mark.safe_softmax_backward
def test_safe_softmax_backward_dispatch():
    """The bridge is reachable from the dispatched FlagGems ``_safe_softmax``
    forward.  Unlike the global ``flag_gems.use_gems()`` context manager, a
    private Library tears the override down in ``finally`` and leaves the
    native chain measured above untouched."""
    from flag_gems.ops.safe_softmax_backward import _SafeSoftmaxFunction

    shape, dim = (64, 32), 1
    inp = torch.rand(shape, dtype=torch.float32, device=flag_gems.device) * 12.0 - 6.0
    inp[5, :] = float("-inf")
    grad_output = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, True).requires_grad_(True)
    ref_out = torch.ops.aten._safe_softmax(ref_inp, dim)
    ref_grad_output = utils.to_reference(grad_output, True)
    (ref_grad_input,) = torch.autograd.grad(ref_out, (ref_inp,), (ref_grad_output,))

    lib = torch.library.Library("aten", "IMPL")
    try:
        flag_gems.only_enable(lib=lib, include=["_safe_softmax"])
        x = inp.clone().requires_grad_(True)
        out = _SafeSoftmaxFunction.apply(x, dim)
        (res_grad_input,) = torch.autograd.grad(out, (x,), (grad_output,))
    finally:
        lib._destroy()

    utils.gems_assert_close(
        res_grad_input, ref_grad_input, torch.float32, reduce_dim=shape[dim]
    )
