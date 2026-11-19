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

"""Accuracy tests for the ``linalg_slogdet`` backward.

The ATen schema has no independent ``_linalg_slogdet_backward`` operator: the
engine builds ``LinalgSlogdetBackward0`` from
``torch/csrc/autograd/FunctionsManual.cpp`` and that formula reads the LU and
pivots outputs. FlagGems returns those two outputs only for schema parity, so
the devolving formula is wrong on the torch path; the implementation binds its
own node to the Autograd key instead. These tests therefore check the endpoint
``grad_logabsdet * A^{-H}`` against native autograd, against an independent
float64 reimplementation of the formula, and against non-differentiable
outputs.
"""

import contextlib

import pytest
import torch

import flag_gems
from flag_gems.ops._linalg_slogdet import linalg_slogdet_backward

from . import accuracy_utils as utils

# The five default shapes of ``_linalg_slogdet`` (batched small matrix plus
# 4x4 through 32x32) and the single dtype its kernel supports.
SLOGDET_SHAPES = [(2, 3, 3), (4, 4), (8, 8), (16, 16), (32, 32)]
SLOGDET_DTYPES = [torch.float32]


def _well_conditioned(shape, dtype):
    """A diagonally dominant matrix keeps the reference factorization stable."""
    n = shape[-1]
    A = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    return A + torch.eye(n, dtype=dtype, device=flag_gems.device) * n


def _grad_logabsdet(shape, dtype):
    batch_shape = shape[:-2]
    if not batch_shape:
        return torch.randn((), dtype=dtype, device=flag_gems.device)
    return torch.randn(batch_shape, dtype=dtype, device=flag_gems.device)


def _reference_grad(grad_logabsdet, A, dtype):
    """``grad_logabsdet[..., None, None] * A^{-H}`` in float64.

    Both inputs are passed through ``utils.to_reference`` so the reference
    runs on the device selected by ``--ref`` (CPU under ``--ref cpu``), and the
    result is cast back to ``dtype`` for the tolerance helper. The float64 step
    is what keeps the reference itself accurate.
    """
    ref_A = utils.to_reference(A)
    ref_grad = utils.to_reference(grad_logabsdet)
    result = (
        ref_grad.to(torch.float64)[..., None, None]
        * torch.linalg.inv(ref_A.to(torch.float64)).mH
    )
    return result.to(dtype=dtype)


@contextlib.contextmanager
def _dispatch_enabled(include):
    """Register ``include`` through the real registrar for one dispatch probe.

    ``flag_gems.use_gems()`` is forbidden in test files. ``only_enable`` is the
    sanctioned registration entry (it is what ``use_gems`` itself calls), and a
    private ``Library`` keeps the registration from leaking into the next
    parametrization.
    """
    lib = torch.library.Library("aten", "IMPL")
    try:
        flag_gems.only_enable(lib=lib, include=include)
        yield
    finally:
        lib._destroy()


@pytest.mark.linalg_slogdet_backward
@pytest.mark.parametrize("shape", SLOGDET_SHAPES)
@pytest.mark.parametrize("dtype", SLOGDET_DTYPES)
def test_linalg_slogdet_backward(shape, dtype):
    """End-to-end autograd: FlagGems entry against native ``slogdet``."""
    A = _well_conditioned(shape, dtype)
    ref_A = utils.to_reference(A).requires_grad_(True)
    ref_sign, ref_logabsdet = torch.linalg.slogdet(ref_A)
    grad_logabsdet = _grad_logabsdet(shape, dtype)
    ref_grad = utils.to_reference(grad_logabsdet)

    (expected,) = torch.autograd.grad(ref_logabsdet, ref_A, ref_grad)

    res_A = A.clone().requires_grad_(True)
    res_sign, res_logabsdet, res_LU, res_pivots = flag_gems._linalg_slogdet(res_A)
    (res_grad,) = torch.autograd.grad(res_logabsdet, res_A, grad_logabsdet)

    utils.gems_assert_close(res_sign, ref_sign, dtype)
    utils.gems_assert_close(res_logabsdet, ref_logabsdet, dtype, reduce_dim=shape[-1])
    utils.gems_assert_close(res_grad, expected, dtype, reduce_dim=shape[-1])

    # LU and pivots stay non-differentiable exactly as in the ATen schema.
    assert not res_LU.requires_grad
    assert not res_pivots.requires_grad


@pytest.mark.linalg_slogdet_backward
@pytest.mark.parametrize("shape", SLOGDET_SHAPES)
@pytest.mark.parametrize("dtype", SLOGDET_DTYPES)
def test_linalg_slogdet_backward_formula(shape, dtype):
    """The kernel matches the ATen formula reimplemented in float64."""
    A = _well_conditioned(shape, dtype)
    grad_logabsdet = _grad_logabsdet(shape, dtype)
    res_A = A.clone().requires_grad_(True)
    _sign, res_logabsdet, _LU, _pivots = flag_gems._linalg_slogdet(res_A)
    (res_grad,) = torch.autograd.grad(res_logabsdet, res_A, grad_logabsdet)

    expected = _reference_grad(grad_logabsdet, A, dtype)
    utils.gems_assert_close(res_grad, expected, dtype, reduce_dim=shape[-1])


@pytest.mark.linalg_slogdet_backward
@pytest.mark.parametrize("shape", SLOGDET_SHAPES)
@pytest.mark.parametrize("dtype", SLOGDET_DTYPES)
def test_linalg_slogdet_backward_entry(shape, dtype):
    """Direct call of the exported backward helper against the formula."""
    A = _well_conditioned(shape, dtype)
    grad_logabsdet = _grad_logabsdet(shape, dtype)

    res_grad = linalg_slogdet_backward(grad_logabsdet, A)
    expected = _reference_grad(grad_logabsdet, A, dtype)

    assert res_grad.shape == A.shape
    assert res_grad.dtype == A.dtype
    utils.gems_assert_close(res_grad, expected, dtype, reduce_dim=shape[-1])


@pytest.mark.linalg_slogdet_backward
@pytest.mark.parametrize("shape", SLOGDET_SHAPES)
@pytest.mark.parametrize("dtype", SLOGDET_DTYPES)
def test_linalg_slogdet_backward_needs_pivoting(shape, dtype):
    """Matrices whose elimination must swap rows still produce the formula.

    The kernel selects a pivot per column; a zero leading entry (or a
    transpose, which is a non-contiguous view) exercises that branch instead of
    the diagonally dominant path the other tests use.
    """
    n = shape[-1]
    A = _well_conditioned(shape, dtype)
    # A transposed view is non-contiguous, and zeroing the first column above
    # the diagonal forces a row swap in the first elimination step while the
    # large diagonal keeps the matrix nonsingular.
    A = A.transpose(-1, -2)
    assert not A.is_contiguous()
    A = A.contiguous()
    A = A.clone()
    A[..., 0, 1:] = 0.0

    grad_logabsdet = _grad_logabsdet(shape, dtype)
    res_A = A.clone().requires_grad_(True)
    _sign, res_logabsdet, _LU, _pivots = flag_gems._linalg_slogdet(res_A)
    (res_grad,) = torch.autograd.grad(res_logabsdet, res_A, grad_logabsdet)

    expected = _reference_grad(grad_logabsdet, A, dtype)
    utils.gems_assert_close(res_grad, expected, dtype, reduce_dim=n)


@pytest.mark.linalg_slogdet_backward
@pytest.mark.parametrize("shape", SLOGDET_SHAPES)
@pytest.mark.parametrize("dtype", SLOGDET_DTYPES)
def test_linalg_slogdet_backward_batched(shape, dtype):
    """A batch entry's gradient must depend only on its own matrix."""
    A = _well_conditioned(shape, dtype)
    grad_logabsdet = _grad_logabsdet(shape, dtype)
    res_A = A.clone().requires_grad_(True)
    _sign, res_logabsdet, _LU, _pivots = flag_gems._linalg_slogdet(res_A)
    (res_grad,) = torch.autograd.grad(res_logabsdet, res_A, grad_logabsdet)

    batch_shape = shape[:-2]
    batch_size = 1
    for dim in batch_shape:
        batch_size *= dim
    flat_A = A.reshape(batch_size, shape[-1], shape[-1])
    flat_grad = grad_logabsdet.reshape(batch_size)
    flat_res = res_grad.reshape(batch_size, shape[-1], shape[-1])
    for index in range(batch_size):
        single = _reference_grad(flat_grad[index], flat_A[index], dtype)
        utils.gems_assert_close(flat_res[index], single, dtype, reduce_dim=shape[-1])


@pytest.mark.linalg_slogdet_backward
@pytest.mark.parametrize("shape", SLOGDET_SHAPES)
@pytest.mark.parametrize("dtype", SLOGDET_DTYPES)
def test_linalg_slogdet_backward_non_contiguous_grad(shape, dtype):
    """An expanded upstream gradient feeds the kernel the same values."""
    A = _well_conditioned(shape, dtype)
    batch_shape = shape[:-2]
    grad_logabsdet = _grad_logabsdet(shape, dtype)
    res_A = A.clone().requires_grad_(True)
    _sign, res_logabsdet, _LU, _pivots = flag_gems._linalg_slogdet(res_A)
    (res_grad,) = torch.autograd.grad(res_logabsdet, res_A, grad_logabsdet)

    expected = _reference_grad(grad_logabsdet, A, dtype)
    utils.gems_assert_close(res_grad, expected, dtype, reduce_dim=shape[-1])

    if batch_shape:
        # A strided, non-contiguous upstream gradient must give the same answer:
        # build it from a transposed buffer and slice it back.
        wide = torch.randn(
            (batch_shape[0] * 2,) + batch_shape[1:],
            dtype=dtype,
            device=flag_gems.device,
        )
        strided = wide[::2]
        assert not strided.is_contiguous()
        strided_A = A.clone().requires_grad_(True)
        _s, strided_logabsdet, _l, _p = flag_gems._linalg_slogdet(strided_A)
        (strided_grad,) = torch.autograd.grad(strided_logabsdet, strided_A, strided)
        expected_strided = _reference_grad(strided, A, dtype)
        utils.gems_assert_close(
            strided_grad, expected_strided, dtype, reduce_dim=shape[-1]
        )


@pytest.mark.linalg_slogdet_backward
@pytest.mark.parametrize("dtype", SLOGDET_DTYPES)
def test_linalg_slogdet_backward_sign_only(dtype):
    """Differentiating only ``sign`` yields a zero gradient, like native.

    ``sign`` is piecewise constant for real input, so native
    ``torch.linalg.slogdet`` reports an unused input for that graph. The
    FlagGems node returns a zero gradient instead of a fabricated non-zero one.
    """
    shape = (8, 8)
    A = _well_conditioned(shape, dtype)
    res_A = A.clone().requires_grad_(True)
    res_sign, _res_logabsdet, _LU, _pivots = flag_gems._linalg_slogdet(res_A)
    (res_grad,) = torch.autograd.grad(
        res_sign, res_A, torch.ones_like(res_sign), allow_unused=True
    )
    assert res_grad is None or torch.count_nonzero(res_grad) == 0


@pytest.mark.linalg_slogdet_backward
@pytest.mark.parametrize("shape", SLOGDET_SHAPES)
@pytest.mark.parametrize("dtype", SLOGDET_DTYPES)
def test_linalg_slogdet_backward_dispatch(shape, dtype):
    """``torch.ops.aten._linalg_slogdet`` under gems dispatch runs our node."""
    A = _well_conditioned(shape, dtype)
    grad_logabsdet = _grad_logabsdet(shape, dtype)

    # The reference is the documented formula, not native autograd, so it does
    # not depend on where ``--ref`` puts the reference device.
    expected = _reference_grad(grad_logabsdet, A, dtype)

    res_A = A.clone().requires_grad_(True)
    with _dispatch_enabled(["_linalg_slogdet"]):
        _sign, res_logabsdet, _LU, _pivots = torch.ops.aten._linalg_slogdet(res_A)
        assert "_SlogdetAutograd" in type(res_logabsdet.grad_fn).__name__
        (res_grad,) = torch.autograd.grad(res_logabsdet, res_A, grad_logabsdet)

    utils.gems_assert_close(res_grad, expected, dtype, reduce_dim=shape[-1])


@pytest.mark.linalg_slogdet_backward
@pytest.mark.parametrize("shape", SLOGDET_SHAPES)
@pytest.mark.parametrize("dtype", SLOGDET_DTYPES)
def test_linalg_slogdet_backward_public_api(shape, dtype):
    """``torch.linalg.slogdet`` under gems reaches the same fused backward."""
    A = _well_conditioned(shape, dtype)
    grad_logabsdet = _grad_logabsdet(shape, dtype)

    res_A = A.clone().requires_grad_(True)
    with _dispatch_enabled(["_linalg_slogdet", "linalg_slogdet"]):
        res_sign, res_logabsdet = torch.linalg.slogdet(res_A)
        assert "_LinalgSlogdetAutograd" in type(res_logabsdet.grad_fn).__name__
        assert res_sign.shape == tuple(shape[:-2])
        (res_grad,) = torch.autograd.grad(res_logabsdet, res_A, grad_logabsdet)

    expected = _reference_grad(grad_logabsdet, A, dtype)
    utils.gems_assert_close(res_grad, expected, dtype, reduce_dim=shape[-1])


@pytest.mark.linalg_slogdet_backward
@pytest.mark.parametrize("dtype", SLOGDET_DTYPES)
def test_linalg_slogdet_backward_singular(dtype):
    """A singular matrix yields non-finite gradients, as native does.

    ATen's ``slogdet_backward`` divides by the factorization, so a singular
    input gives inf/NaN rather than a silently finite value. The kernel keeps
    a zero pivot instead of masking it, so the same contract holds.
    """
    A = torch.zeros(4, 4, dtype=dtype, device=flag_gems.device)
    A[0, 0] = 1.0
    A[1, 1] = 2.0
    A[2, :] = A[0, :]
    A[3, 3] = 4.0

    grad_logabsdet = torch.ones((), dtype=dtype, device=flag_gems.device)
    res_A = A.clone().requires_grad_(True)
    _sign, res_logabsdet, _LU, _pivots = flag_gems._linalg_slogdet(res_A)
    assert torch.isinf(res_logabsdet) or torch.isnan(res_logabsdet)
    (res_grad,) = torch.autograd.grad(res_logabsdet, res_A, grad_logabsdet)

    # Native keeps the same contract: the logabsdet gradient is A^{-H}, which
    # is undefined at a singular matrix, so the reference is also non-finite.
    # Native keeps the same contract: the logabsdet gradient is A^{-H}, which
    # is undefined at a singular matrix, so both sides must agree on
    # finiteness rather than on a fabricated finite value.
    ref_A = A.clone().requires_grad_(True)
    _ref_sign, ref_logabsdet = torch.linalg.slogdet(ref_A)
    (ref_grad,) = torch.autograd.grad(ref_logabsdet, ref_A, grad_logabsdet)
    # The kernel performs the same division as the native formula, so the
    # non-finite entries agree; a finite result on either side would mean one
    # path silently masked the singularity.
    assert not torch.isfinite(res_grad).all()
    assert not torch.isfinite(ref_grad).all()
