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


@pytest.mark.logaddexp2
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_logaddexp2(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    y = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out = torch.logaddexp2(ref_x, ref_y)

    res_out = flag_gems.logaddexp2(x, y)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.logaddexp2_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_logaddexp2_out(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    y = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out_buf = torch.empty(shape, dtype=ref_x.dtype, device=ref_x.device)
    ref_out = torch.ops.aten.logaddexp2.out(ref_x, ref_y, out=ref_out_buf)

    res_out_buf = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    res_out = flag_gems.logaddexp2_out(x, y, out=res_out_buf)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.logaddexp2
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_logaddexp2_non_finite(dtype):
    # inf/-inf/nan combinations, e.g. logaddexp2(inf, inf) = inf and
    # logaddexp2(-inf, -inf) = -inf (delta = inf - inf = nan must not leak).
    inf = float("inf")
    nan = float("nan")
    x = torch.tensor(
        [inf, -inf, inf, 1.0, nan, -inf], dtype=dtype, device=flag_gems.device
    )
    y = torch.tensor(
        [inf, -inf, 1.0, -inf, 1.0, inf], dtype=dtype, device=flag_gems.device
    )

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out = torch.logaddexp2(ref_x, ref_y)

    res_out = flag_gems.logaddexp2(x, y)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


# ---------------------------------------------------------------------------
# Backward: first-order gradients of logaddexp2 for both inputs.
#
# aten::logaddexp2 has no standalone backward schema; the engine generates
#   self:  grad / (1 + pow(2, other - self))
#   other: grad / (1 + pow(2, self - other))
# so every reference below is a plain native autograd call on torch.logaddexp2
# with the same upstream gradient.
# ---------------------------------------------------------------------------

BROADCAST_CASES = [
    ((4, 1, 8), (1, 5, 8)),
    ((1, 1, 6), (3, 4, 1)),
    ((5,), (2, 3, 5)),
]

INF = float("inf")
NAN = float("nan")


def _require_gems_autograd(res_out):
    """Fail on NVIDIA when the FlagGems autograd path is not the one used."""
    grad_fn_name = type(res_out.grad_fn).__name__ if res_out.grad_fn else ""
    if flag_gems.vendor_name == "nvidia":
        assert "Logaddexp2Function" in grad_fn_name, grad_fn_name
    elif not grad_fn_name:
        pytest.skip("vendor backend does not expose a FlagGems logaddexp2 backward")


def _native_grads(x, y, out_grad, needs=(True, True)):
    """Native autograd in the input dtype.

    The reference keeps the element dtype instead of upcasting to fp64: the
    generated ATen backward materialises ``other - self``, ``pow(2, .)`` and
    ``1 + .`` as low-precision tensors, so native fp16/bf16 gradients are
    themselves a few ULP away from the fp64 answer. Upcasting would therefore
    test a contract the native operator does not satisfy (verified: only 84%
    of fp16 and 50% of bf16 elements of the native gradient stay inside the
    fp64 tolerance on a broadcast case). With ``--ref cpu`` the reference runs
    on the CPU in the same dtype.
    """
    ref_x = utils.to_reference(x).detach().clone().requires_grad_(needs[0])
    ref_y = utils.to_reference(y).detach().clone().requires_grad_(needs[1])
    ref_grad = utils.to_reference(out_grad).detach().clone()
    ref_out = torch.logaddexp2(ref_x, ref_y)
    inputs = [t for t, need in ((ref_x, needs[0]), (ref_y, needs[1])) if need]
    return torch.autograd.grad(ref_out, tuple(inputs), ref_grad)


def _gems_grads(x, y, out_grad, needs=(True, True)):
    res_x = x.detach().requires_grad_(needs[0])
    res_y = y.detach().requires_grad_(needs[1])
    res_out = flag_gems.logaddexp2(res_x, res_y)
    _require_gems_autograd(res_out)
    inputs = [t for t, need in ((res_x, needs[0]), (res_y, needs[1])) if need]
    return torch.autograd.grad(res_out, tuple(inputs), out_grad)


@pytest.mark.logaddexp2_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_logaddexp2_backward(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device) * 1.5
    y = torch.randn(shape, dtype=dtype, device=flag_gems.device) * 1.5
    out_grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_grad_x, ref_grad_y = _native_grads(x, y, out_grad)
    res_grad_x, res_grad_y = _gems_grads(x, y, out_grad)

    utils.gems_assert_close(res_grad_x, ref_grad_x, dtype)
    utils.gems_assert_close(res_grad_y, ref_grad_y, dtype)


@pytest.mark.logaddexp2_backward
@pytest.mark.parametrize("x_shape, y_shape", BROADCAST_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_logaddexp2_backward_broadcast(x_shape, y_shape, dtype):
    out_shape = torch.broadcast_shapes(x_shape, y_shape)
    x = torch.randn(x_shape, dtype=dtype, device=flag_gems.device)
    y = torch.randn(y_shape, dtype=dtype, device=flag_gems.device)
    out_grad = torch.randn(out_shape, dtype=dtype, device=flag_gems.device)

    ref_grad_x, ref_grad_y = _native_grads(x, y, out_grad)
    res_grad_x, res_grad_y = _gems_grads(x, y, out_grad)

    assert res_grad_x.shape == x_shape
    assert res_grad_y.shape == y_shape
    utils.gems_assert_close(res_grad_x, ref_grad_x, dtype)
    utils.gems_assert_close(res_grad_y, ref_grad_y, dtype)


@pytest.mark.logaddexp2_backward
@pytest.mark.parametrize("needs", [(True, False), (False, True)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_logaddexp2_backward_single_sided(needs, dtype):
    shape = (19, 37)
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    y = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    out_grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    (ref_grad,) = _native_grads(x, y, out_grad, needs)
    (res_grad,) = _gems_grads(x, y, out_grad, needs)

    utils.gems_assert_close(res_grad, ref_grad, dtype)


@pytest.mark.logaddexp2_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_logaddexp2_backward_equals_and_steps(dtype):
    # Equal inputs split the upstream gradient in half; +-200 steps saturate the
    # base-2 weight at 0 and 1 through fp32 underflow/overflow of 2**delta.
    base = torch.tensor(
        [0.0, 1.0, -1.0, 17.5, -3.25, 100.0], dtype=dtype, device=flag_gems.device
    )
    x = torch.cat((base, base))
    y = torch.cat(
        (
            base,
            base
            + torch.tensor(
                [200.0] * 3 + [-200.0] * 3, dtype=dtype, device=flag_gems.device
            ),
        )
    )
    out_grad = torch.arange(1, x.numel() + 1, dtype=dtype, device=flag_gems.device)

    ref_grad_x, ref_grad_y = _native_grads(x, y, out_grad)
    res_grad_x, res_grad_y = _gems_grads(x, y, out_grad)

    utils.gems_assert_close(res_grad_x, ref_grad_x, dtype)
    utils.gems_assert_close(res_grad_y, ref_grad_y, dtype)


@pytest.mark.logaddexp2_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_logaddexp2_backward_non_finite(dtype):
    # inf - inf is NaN in the native formula as well, so the reference and the
    # kernel must agree on the same NaN entries.
    x = torch.tensor(
        [INF, -INF, INF, 0.0, -INF, INF, 1.0e4, NAN, 3.0],
        dtype=dtype,
        device=flag_gems.device,
    )
    y = torch.tensor(
        [0.0, 0.0, -INF, -INF, INF, INF, -1.0e4, 1.0, 3.0],
        dtype=dtype,
        device=flag_gems.device,
    )
    out_grad = torch.ones_like(x)

    ref_grad_x, ref_grad_y = _native_grads(x, y, out_grad)
    res_grad_x, res_grad_y = _gems_grads(x, y, out_grad)

    utils.gems_assert_close(res_grad_x, ref_grad_x, dtype, equal_nan=True)
    utils.gems_assert_close(res_grad_y, ref_grad_y, dtype, equal_nan=True)


@pytest.mark.logaddexp2_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_logaddexp2_backward_noncontiguous(dtype):
    x = torch.randn((4, 6), dtype=dtype, device=flag_gems.device).T
    y = torch.randn((4, 6), dtype=dtype, device=flag_gems.device).T
    out_grad = torch.randn((6, 4), dtype=dtype, device=flag_gems.device)
    assert not x.is_contiguous() and not y.is_contiguous()

    ref_grad_x, ref_grad_y = _native_grads(x, y, out_grad)
    res_grad_x, res_grad_y = _gems_grads(x, y, out_grad)

    utils.gems_assert_close(res_grad_x, ref_grad_x, dtype)
    utils.gems_assert_close(res_grad_y, ref_grad_y, dtype)


@pytest.mark.logaddexp2_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_logaddexp2_backward_registered_autograd_key(dtype):
    shape = (37, 41)
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    y = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    out_grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_grad_x, ref_grad_y = _native_grads(x, y, out_grad)

    res_x = x.detach().requires_grad_(True)
    res_y = y.detach().requires_grad_(True)
    res_out = flag_gems.logaddexp2(res_x, res_y)
    _require_gems_autograd(res_out)
    res_grad_x, res_grad_y = torch.autograd.grad(res_out, (res_x, res_y), out_grad)

    utils.gems_assert_close(res_grad_x, ref_grad_x, dtype)
    utils.gems_assert_close(res_grad_y, ref_grad_y, dtype)


@pytest.mark.logaddexp2_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_logaddexp2_backward_functional(dtype):
    # The explicit backward entry (the KGS catalog ABI) must agree with the
    # autograd path, including the broadcast reduction.
    x_shape, y_shape = (4, 1, 8), (1, 5, 8)
    x = torch.randn(x_shape, dtype=dtype, device=flag_gems.device)
    y = torch.randn(y_shape, dtype=dtype, device=flag_gems.device)
    out_grad = torch.randn(
        torch.broadcast_shapes(x_shape, y_shape),
        dtype=dtype,
        device=flag_gems.device,
    )

    ref_grad_x, ref_grad_y = _native_grads(x, y, out_grad)
    res_grad_x, res_grad_y = flag_gems.logaddexp2_backward(out_grad, x, y)

    assert res_grad_x.shape == x_shape and res_grad_y.shape == y_shape
    utils.gems_assert_close(res_grad_x, ref_grad_x, dtype)
    utils.gems_assert_close(res_grad_y, ref_grad_y, dtype)


@pytest.mark.logaddexp2_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_logaddexp2_backward_empty_and_scalar(dtype):
    # 0-dim inputs and zero-element tensors must not launch a malformed grid.
    x = torch.tensor(1.25, dtype=dtype, device=flag_gems.device)
    y = torch.tensor(-0.5, dtype=dtype, device=flag_gems.device)
    out_grad = torch.tensor(3.0, dtype=dtype, device=flag_gems.device)
    ref_grad_x, ref_grad_y = _native_grads(x, y, out_grad)
    res_grad_x, res_grad_y = _gems_grads(x, y, out_grad)
    utils.gems_assert_close(res_grad_x, ref_grad_x, dtype)
    utils.gems_assert_close(res_grad_y, ref_grad_y, dtype)

    empty_x = torch.empty(0, dtype=dtype, device=flag_gems.device)
    empty_y = torch.empty(0, dtype=dtype, device=flag_gems.device)
    empty_grad = torch.empty(0, dtype=dtype, device=flag_gems.device)
    res = flag_gems.logaddexp2_backward(empty_grad, empty_x, empty_y)
    assert res[0].numel() == 0 and res[1].numel() == 0


@pytest.mark.logaddexp2_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_logaddexp2_backward_accuracy_matches_native(dtype):
    # Against the exact fp64 answer the FlagGems gradient must be no worse than
    # native autograd, which itself carries the low-precision intermediate
    # rounding of the generated backward.
    shape = (6, 5)
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device) * 1.5
    y = torch.randn(shape, dtype=dtype, device=flag_gems.device) * 1.5
    out_grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = x.double().requires_grad_(True)
    ref_y = y.double().requires_grad_(True)
    exact = torch.autograd.grad(
        torch.logaddexp2(ref_x, ref_y), (ref_x, ref_y), out_grad.double()
    )

    native = _native_grads(x, y, out_grad)
    gems = _gems_grads(x, y, out_grad)

    eps = torch.finfo(dtype).eps
    for name, native_g, gems_g, exact_g in zip(
        ("grad_self", "grad_other"), native, gems, exact
    ):
        exact_dtype = exact_g.to(dtype).float().cpu()
        scale = exact_dtype.abs().max().item()
        native_err = (native_g.float().cpu() - exact_dtype).abs().max().item()
        gems_err = (gems_g.float().cpu() - exact_dtype).abs().max().item()
        assert gems_err <= native_err + 2 * eps * max(scale, 1.0), (
            f"{name}: FlagGems error {gems_err:.3e} exceeds native error "
            f"{native_err:.3e} for {dtype}"
        )


@pytest.mark.logaddexp2_backward
def test_logaddexp2_backward_rejects_complex():
    x = torch.ones(
        (4,), dtype=torch.complex64, device=flag_gems.device, requires_grad=True
    )
    y = torch.ones(
        (4,), dtype=torch.float32, device=flag_gems.device, requires_grad=True
    )
    with pytest.raises(NotImplementedError, match="complex"):
        flag_gems.logaddexp2(y, x)
    with pytest.raises(NotImplementedError, match="complex"):
        flag_gems.logaddexp2_backward(y, x, y)
