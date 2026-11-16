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
    # --quick keeps the smoke run on a single dtype to stay fast; the full run
    # covers every float dtype through the shared constant.
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES


# One case per layout the reverse scan can see: 1-D (row scan), reduce axis at
# the front / middle / end, negative dim, non-power-of-two lengths (2637, 1025)
# and a 4-D tensor with a short middle axis.
CUMSUM_BACKWARD_SHAPE_DIMS = (
    [((2, 32), -1)]
    if cfg.QUICK_MODE
    else [
        ((1024,), 0),
        ((128, 64), 0),
        ((128, 64), 1),
        ((128, 64), -1),
        ((16, 32, 64), 1),
        ((8, 3, 7, 5), 2),
        ((2637,), -1),
        ((64, 1025), 1),
    ]
)


def _cumsum_backward_reference(grad, dim):
    """float64 suffix sum of the upstream gradient along ``dim``.

    ``cumsum``'s gradient is ``sum_{j >= i} g_j``; computing it in float64 keeps
    the reference independent of the kernel's own accumulation order.
    """
    grad = utils.to_reference(grad, True)
    return torch.flip(torch.cumsum(torch.flip(grad, [dim]), dim), [dim]).to("cpu")


@pytest.mark.cumsum_backward
@pytest.mark.parametrize("shape, dim", CUMSUM_BACKWARD_SHAPE_DIMS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_cumsum_backward(shape, dim, dtype):
    """Gradient of ``flag_gems.cumsum`` against the float64 suffix sum."""
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad_out = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    res_inp = inp.detach().requires_grad_(True)
    res_out = flag_gems.cumsum(res_inp, dim)
    (res_grad,) = torch.autograd.grad(res_out, res_inp, grad_out)

    assert res_out.grad_fn is not None
    assert "CumsumFunction" in type(res_out.grad_fn).__name__

    ref_grad = _cumsum_backward_reference(grad_out, dim).to(dtype)

    # The forward output of the autograd.Function uses the FlagGems kernel (fp32
    # accumulate in fp16/bf16), so compare it against the fp64 reference with a
    # reduce-dim-scaled atol that absorbs the accumulation-order difference
    # over a long row, matching the existing forward test.
    utils.gems_assert_close(
        res_out,
        torch.cumsum(utils.to_reference(inp, True), dim),
        dtype,
        reduce_dim=shape[dim],
    )
    atol = 1e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-4
    utils.gems_assert_close(res_grad, ref_grad, dtype, reduce_dim=shape[dim], atol=atol)


@pytest.mark.cumsum_backward
@pytest.mark.parametrize("dim", [1, -1])
def test_cumsum_backward_autograd_chain(dim):
    """End-to-end float32 autograd chain against the float64 torch chain."""
    shape, dtype = (128, 257), torch.float32
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad_out = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp.detach(), True).requires_grad_(True)
    ref_grad_out = utils.to_reference(grad_out, True)
    (ref_grad,) = torch.autograd.grad(torch.cumsum(ref_inp, dim), ref_inp, ref_grad_out)

    res_inp = inp.detach().requires_grad_(True)
    res_out = flag_gems.cumsum(res_inp, dim)
    (res_grad,) = torch.autograd.grad(res_out, res_inp, grad_out)

    utils.gems_assert_close(res_grad, ref_grad.to(dtype), dtype, reduce_dim=shape[dim])


@pytest.mark.cumsum_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_cumsum_backward_non_contiguous(dtype):
    base = torch.randn((48, 64), dtype=dtype, device=flag_gems.device)
    inp = base.t()
    dim = -1
    grad_out = torch.randn(inp.shape, dtype=dtype, device=flag_gems.device)

    res_inp = inp.detach().requires_grad_(True)
    res_out = flag_gems.cumsum(res_inp, dim)
    (res_grad,) = torch.autograd.grad(res_out, res_inp, grad_out)

    ref_grad = _cumsum_backward_reference(grad_out, dim).to(dtype)
    utils.gems_assert_close(
        res_grad,
        ref_grad,
        dtype,
        reduce_dim=inp.shape[dim],
        atol=1e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-4,
    )


@pytest.mark.cumsum_backward
@pytest.mark.parametrize("dim", [0, 1, -2])
def test_cumsum_backward_middle_and_negative_dim(dim):
    shape, dtype = (4, 8, 16), torch.float32
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad_out = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    res_inp = inp.detach().requires_grad_(True)
    res_out = flag_gems.cumsum(res_inp, dim)
    (res_grad,) = torch.autograd.grad(res_out, res_inp, grad_out)

    ref_grad = _cumsum_backward_reference(grad_out, dim).to(dtype)
    utils.gems_assert_close(res_grad, ref_grad, dtype, reduce_dim=shape[dim])


@pytest.mark.cumsum_backward
@pytest.mark.parametrize("shape, dim", [((0,), 0), ((0, 5), 1), ((3, 0, 4), 2)])
def test_cumsum_backward_empty(shape, dim):
    inp = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    grad_out = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)

    res_inp = inp.detach().requires_grad_(True)
    res_out = flag_gems.cumsum(res_inp, dim)
    (res_grad,) = torch.autograd.grad(res_out, res_inp, grad_out)

    assert res_grad.shape == inp.shape
    assert res_grad.dtype == inp.dtype
    assert res_grad.numel() == 0


@pytest.mark.cumsum_backward
def test_cumsum_backward_scalar_input():
    inp = torch.tensor(2.5, dtype=torch.float32, device=flag_gems.device)
    res_inp = inp.detach().requires_grad_(True)
    res_out = flag_gems.cumsum(res_inp, 0)
    (res_grad,) = torch.autograd.grad(res_out, res_inp, torch.ones_like(res_out))

    ref_inp = utils.to_reference(inp.detach(), True).requires_grad_(True)
    (ref_grad,) = torch.autograd.grad(
        torch.cumsum(ref_inp, 0), ref_inp, torch.ones((), dtype=torch.float64)
    )

    assert res_out.detach().item() == pytest.approx(float(ref_inp.item()))
    assert res_grad.item() == pytest.approx(float(ref_grad.item()))


@pytest.mark.cumsum_backward
def test_cumsum_backward_use_gems_entry():
    """The public ``torch.cumsum`` under ``use_gems()`` keeps ATen's node.

    ``aten::cumsum`` is a CompositeExplicitAutograd op, so the generated
    ``CumsumBackward0`` node stays above the device kernel and differentiates
    with the ATen formula. This pins that public path against the float64
    reference; ``flag_gems.cumsum`` is the entry that reaches the Triton kernel.
    """
    shape, dim, dtype = (64, 128), 1, torch.float32
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad_out = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp.detach(), True).requires_grad_(True)
    ref_grad_out = utils.to_reference(grad_out, True)
    (ref_grad,) = torch.autograd.grad(torch.cumsum(ref_inp, dim), ref_inp, ref_grad_out)

    with flag_gems.use_gems():
        res_inp = inp.detach().requires_grad_(True)
        res_out = torch.cumsum(res_inp, dim)
        (res_grad,) = torch.autograd.grad(res_out, res_inp, grad_out)

    utils.gems_assert_close(res_grad, ref_grad.to(dtype), dtype, reduce_dim=shape[dim])
