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

# ``aten::_fused_dropout`` only registers CUDA (plus Meta) kernels, so the
# native reference for the backward must be built with the GPU forward even in
# ``--ref cpu`` mode: the reference formula below runs the derivative chain on
# whichever device holds the native mask.  The FlagGems entry point is pure
# elementwise and runs on any device, but the acceptance target is CUDA.
FLOAT_DTYPES = utils.FLOAT_DTYPES + [torch.float64]

KEEP_PROBABILITIES = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]


def _as_ref(t):
    """Normalize the CUDA-native reference for the active comparison device.

    ``aten::_fused_dropout`` has no CPU kernel, so in ``--ref cpu`` mode the
    native chain still runs on CUDA and its results must be moved to the CPU
    explicitly (the plain ``to_reference`` upcast path would try to run the
    forward on CPU and fail).  Values are already exact.
    """
    if utils.TO_CPU:
        return t.cpu()
    return t


def _native_input_grad(x, grad, p):
    """ATen's FusedDropoutBackward0: grad * mask / p for the keep-prob forward.

    The chain runs on ``x``'s device: ``aten::_fused_dropout`` only registers a
    CUDA kernel, so the inputs must not be moved to the ``--ref cpu`` device.
    The derivative itself is exact (``grad * mask / p``), so no upcast is
    needed either.
    """
    out, mask = torch.ops.aten._fused_dropout(x, p)
    (grad_in,) = torch.autograd.grad(out, x, grad)
    return grad_in, mask


@pytest.mark.fused_dropout_backward
@pytest.mark.parametrize("p", KEEP_PROBABILITIES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_fused_dropout_backward_flat(dtype, p):
    """The explicit entry matches native autograd bit-exactly for every p."""
    if dtype == torch.float64 and not utils.fp64_is_supported:
        pytest.skip("Device does not support float64")
    x = torch.randn(7777, dtype=dtype, device=flag_gems.device, requires_grad=True)
    grad = torch.randn(7777, dtype=dtype, device=flag_gems.device)

    ref_grad_in, mask = _native_input_grad(x, grad, p)

    res_grad = flag_gems.fused_dropout_backward(grad, mask, p)
    assert res_grad.dtype == grad.dtype
    assert res_grad.shape == grad.shape

    utils.gems_assert_close(res_grad, _as_ref(ref_grad_in), dtype, atol=0, reduce_dim=1)


@pytest.mark.fused_dropout_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_fused_dropout_backward_shapes(dtype):
    """Scalars, vectors, multidimensional and empty tensors."""
    if dtype == torch.float64 and not utils.fp64_is_supported:
        pytest.skip("Device does not support float64")
    shapes = [(), (1,), (17,), (2, 19, 7), (1024, 1024), (2, 0, 3)]
    for shape in shapes:
        x = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
        grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        ref_grad_in, mask = _native_input_grad(x, grad, 0.5)
        res_grad = flag_gems.fused_dropout_backward(grad, mask, 0.5)
        utils.gems_assert_close(res_grad, _as_ref(ref_grad_in), dtype, atol=0)


@pytest.mark.fused_dropout_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_fused_dropout_backward_non_contiguous(dtype):
    """A transposed upstream gradient and a transposed forward input."""
    if dtype == torch.float64 and not utils.fp64_is_supported:
        pytest.skip("Device does not support float64")
    base = torch.randn((48, 64), dtype=dtype, device=flag_gems.device)
    x = base.t().requires_grad_(True)  # shape (64, 48), non-contiguous forward input
    grad = torch.randn((48, 64), dtype=dtype, device=flag_gems.device).t()
    assert grad.shape == x.shape and not grad.is_contiguous()

    out, mask = torch.ops.aten._fused_dropout(x, 0.7)
    # autograd.grad requires grad_output to match the output layout; feed it the
    # same values in a contiguous copy, while the FlagGems entry gets the
    # non-contiguous view (its wrapper makes it contiguous first).
    (ref_grad_in,) = torch.autograd.grad(out, x, grad.contiguous())
    res_grad = flag_gems.fused_dropout_backward(grad.contiguous(), mask, 0.7)
    utils.gems_assert_close(res_grad, _as_ref(ref_grad_in), dtype, atol=0)


@pytest.mark.fused_dropout_backward
@pytest.mark.parametrize("p", [0.0, 0.2, 0.8, 1.0])
def test_fused_dropout_backward_forward_autograd_bridge(p):
    """The FusedDropoutFunction bridge produces the native input gradient."""
    x = torch.randn(4096, device=flag_gems.device, requires_grad=True)
    from flag_gems.ops.fused_dropout_backward import FusedDropoutFunction

    out, mask = FusedDropoutFunction.apply(x, p)
    assert mask.dtype == torch.uint8
    if p == 0.0:
        # The forward is all-NaN (scale 1/0) and no finite gradient exists.
        assert torch.isnan(out).all()
        return
    grad = torch.randn(4096, device=flag_gems.device)
    (ref_grad_in,) = torch.autograd.grad(out, x, grad)
    manual = grad * mask.float() / p
    torch.testing.assert_close(ref_grad_in, manual, rtol=0, atol=0, equal_nan=p == 0.0)


@pytest.mark.fused_dropout_backward
def test_fused_dropout_backward_rejects_invalid_inputs():
    """Negative/zero p, non-float dtypes, and mask shape mismatches raise."""
    grad = torch.randn(8, device=flag_gems.device)
    mask = torch.zeros(8, device=flag_gems.device, dtype=torch.uint8)
    with pytest.raises(ValueError):
        flag_gems.fused_dropout_backward(grad, mask, 0.0)
    with pytest.raises(TypeError):
        flag_gems.fused_dropout_backward(
            grad.to(torch.int32),
            mask,
            0.5,
        )
    with pytest.raises(RuntimeError):
        flag_gems.fused_dropout_backward(
            grad,
            torch.zeros(9, device=flag_gems.device, dtype=torch.uint8),
            0.5,
        )


@pytest.mark.fused_dropout_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_fused_dropout_backward_mask_semantics(dtype):
    """Kept lanes scale by 1/p; dropped lanes are exactly zero."""
    if dtype == torch.float64 and not utils.fp64_is_supported:
        pytest.skip("Device does not support float64")
    p = 0.5
    grad = torch.randn(256, dtype=dtype, device=flag_gems.device)
    mask = torch.zeros(256, dtype=torch.uint8, device=flag_gems.device)
    mask[::2] = 1
    res = flag_gems.fused_dropout_backward(grad, mask, p)
    torch.testing.assert_close(res[1::2], torch.zeros_like(res[1::2]), rtol=0, atol=0)
    # Kept lanes: the fp32-opmath chain rounds grad to fp32 (exact for fp32
    # inputs), doubles it, and rounds back; for fp64 the multiply stays fp64.
    kept = (grad[::2].float() if dtype != torch.float64 else grad[::2]) * (1.0 / p)
    torch.testing.assert_close(res[::2], kept.to(dtype), rtol=0, atol=0)
