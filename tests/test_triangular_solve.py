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

import itertools
import logging

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import flag_gems

from .accuracy_utils import to_reference

FP64_VENDORS = ("nvidia", "hygon", "metax")
DTYPES = [
    torch.float32,
    pytest.param(
        torch.float64,
        marks=pytest.mark.skipif(
            flag_gems.vendor_name not in FP64_VENDORS,
            reason="triangular_solve float64 contract covers NVIDIA, Hygon and MetaX",
        ),
    ),
]
FLAGS = list(itertools.product((False, True), repeat=3))
# (A batch, B batch, matrix order, number of RHS columns).
SYSTEMS = [
    ((), (), 1, 1),
    ((), (), 7, 3),
    ((), (), 16, 1),
    ((), (), 33, 17),
    ((), (), 64, 4),
    ((), (), 65, 33),
    ((), (), 128, 8),
    ((), (), 257, 3),
    ((), (), 512, 8),
    ((), (), 1024, 1),
    ((), (), 1024, 64),
    ((4,), (4,), 16, 3),
    ((2, 3), (2, 3), 33, 4),
    ((2, 1), (3,), 7, 3),
    ((), (2, 3), 16, 4),
    ((2, 3), (), 16, 4),
    ((1, 3), (2, 1), 33, 3),
]


class _RejectNativeSolve(TorchDispatchMode):
    """Direct entry points must never delegate a solve to native ATen."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = func._schema.name
        assert name not in {
            "aten::triangular_solve",
            "aten::linalg_solve_triangular",
            "aten::_linalg_solve_ex",
            "aten::linalg_solve",
            "aten::linalg_lu_solve",
            "aten::cholesky_solve",
            "aten::linalg_inv",
            "aten::linalg_inv_ex",
        }, f"FlagGems called native solve: {func}"
        return func(*args, **(kwargs or {}))


def _inputs(a_batch, b_batch, n, nrhs, dtype, unitriangular=False):
    # Bounded off-diagonal row sums keep every system well conditioned,
    # including the unit-diagonal cases and large orders.
    A = torch.empty(*a_batch, n, n, dtype=dtype, device=flag_gems.device)
    A.uniform_(-1, 1)
    A.mul_(0.125 / max(n, 1))
    A.diagonal(0, -2, -1).fill_(7.0 if unitriangular else 2.0)
    B = torch.randn(*b_batch, n, nrhs, dtype=dtype, device=flag_gems.device)
    return B, A


def _native_reference(B, A, upper, transpose, unitriangular):
    # Some vendor Torch builds provide device TRSM but do not link CPU BLAS.
    # Skip only this missing reference capability, never a FlagGems failure.
    try:
        return torch.ops.aten.triangular_solve.default(
            B, A, upper, transpose, unitriangular
        )
    except RuntimeError as error:
        if (
            B.device.type == A.device.type == "cpu"
            and "Calling torch.triangular_solve on a CPU tensor requires "
            "compiling PyTorch with BLAS" in str(error)
        ):
            pytest.skip(
                "CPU triangular_solve reference requires PyTorch built with BLAS"
            )
        raise


def _reference(B, A, upper, transpose, unitriangular):
    outputs = _native_reference(
        to_reference(B), to_reference(A), upper, transpose, unitriangular
    )
    return tuple(output.cpu() for output in outputs)


def _close(actual, expected, dtype, equal_nan=False):
    tolerance = 1e-8 if dtype == torch.float64 else 1e-4
    torch.testing.assert_close(
        actual.cpu(),
        expected.cpu(),
        atol=tolerance,
        rtol=tolerance,
        equal_nan=equal_nan,
    )


def _functional(B, A, flags, entry):
    if entry == "direct":
        with _RejectNativeSolve():
            return flag_gems.triangular_solve(B, A, *flags)
    with flag_gems.use_gems(include=["triangular_solve"]):
        return torch.ops.aten.triangular_solve.default(B, A, *flags)


def _out(B, A, flags, X, M, entry):
    if entry == "direct":
        with _RejectNativeSolve():
            return flag_gems.triangular_solve_out(B, A, *flags, X=X, M=M)
    with flag_gems.use_gems(include=["triangular_solve_out"]):
        return torch.ops.aten.triangular_solve.X(B, A, *flags, X=X, M=M)


def _assert_result(result, reference, B, A, functional=True):
    X, M = result
    ref_X, ref_M = reference
    assert X.shape == ref_X.shape
    assert M.shape == ref_M.shape
    assert X.dtype == M.dtype == A.dtype
    assert X.device == M.device == A.device
    _close(X, ref_X, A.dtype)
    torch.testing.assert_close(M.cpu(), ref_M, atol=0, rtol=0, equal_nan=True)
    if functional:
        assert X.stride() == ref_X.stride()
        assert M.stride() == ref_M.stride()
        if M.numel():
            assert M.untyped_storage().data_ptr() != A.untyped_storage().data_ptr()
        if X.numel():
            assert X.untyped_storage().data_ptr() != B.untyped_storage().data_ptr()


@pytest.mark.triangular_solve
@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("flags", FLAGS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("entry", ["direct", "dispatcher"])
def test_triangular_solve(system, flags, dtype, entry, caplog):
    B, A = _inputs(*system, dtype, unitriangular=flags[2])
    before_A, before_B = A.clone(), B.clone()
    reference = _reference(B, A, *flags)
    with caplog.at_level(logging.DEBUG):
        result = _functional(B, A, flags, entry)
    assert any(
        record.name.endswith(".triangular_solve")
        and record.message.endswith(" TRIANGULAR_SOLVE")
        for record in caplog.records
    ), "triangular_solve did not execute a FlagGems implementation"
    _assert_result(result, reference, B, A)
    torch.testing.assert_close(A, before_A, atol=0, rtol=0)
    torch.testing.assert_close(B, before_B, atol=0, rtol=0)


@pytest.mark.triangular_solve
@pytest.mark.parametrize("layout", ["transpose", "slice", "expand"])
@pytest.mark.parametrize("flags", FLAGS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("entry", ["direct", "dispatcher"])
def test_triangular_solve_noncontiguous(layout, flags, dtype, entry):
    B, A = _inputs((2, 1), (3,), 17, 5, dtype, unitriangular=flags[2])
    if layout == "transpose":
        A = A.transpose(-2, -1).contiguous().transpose(-2, -1)
        B = B.transpose(-2, -1).contiguous().transpose(-2, -1)
    elif layout == "slice":
        a_holder = torch.empty(*A.shape[:-1], 34, dtype=dtype, device=A.device)
        b_holder = torch.empty(*B.shape[:-1], 10, dtype=dtype, device=B.device)
        a_holder[..., ::2].copy_(A)
        b_holder[..., ::2].copy_(B)
        A, B = a_holder[..., ::2], b_holder[..., ::2]
    else:
        A = A.expand(2, 3, 17, 17)
        B = B.expand(2, 3, 17, 5)
    assert not A.is_contiguous() and not B.is_contiguous()
    reference = _reference(B, A, *flags)
    _assert_result(_functional(B, A, flags, entry), reference, B, A)


@pytest.mark.triangular_solve
@pytest.mark.parametrize("flags", FLAGS)
@pytest.mark.parametrize("poison", [float("nan"), float("inf")])
@pytest.mark.parametrize("entry", ["direct", "dispatcher"])
def test_triangular_solve_ignored_entries(flags, poison, entry):
    B, A = _inputs((2, 1), (3,), 7, 3, torch.float32, flags[2])
    unused = torch.ones(7, 7, dtype=torch.bool, device=A.device)
    unused = unused.tril(-1) if flags[0] else unused.triu(1)
    A.masked_fill_(unused, poison)
    if flags[2]:
        A.diagonal(0, -2, -1).fill_(poison)
    reference = _reference(B, A, *flags)
    result = _functional(B, A, flags, entry)
    _assert_result(result, reference, B, A)
    assert torch.isfinite(result[0]).all()


@pytest.mark.triangular_solve
@pytest.mark.parametrize("diagonal", [0.0, 1e-12])
@pytest.mark.parametrize(
    "upper,transpose", list(itertools.product((False, True), repeat=2))
)
@pytest.mark.parametrize("dtype", DTYPES)
def test_triangular_solve_singular_and_near_singular(diagonal, upper, transpose, dtype):
    A = torch.eye(4, dtype=dtype, device=flag_gems.device)
    A[0, 0] = diagonal
    B = torch.ones(4, 2, dtype=dtype, device=flag_gems.device)
    flags = (upper, transpose, False)
    reference = _reference(B, A, *flags)
    X, M = _functional(B, A, flags, "direct")
    # Singular systems have nonfinite outputs; compare their locations and
    # signs independently of the ordinary conditioned accuracy matrix.
    ref_X = reference[0]
    assert torch.equal(torch.isnan(X.cpu()), torch.isnan(ref_X))
    assert torch.equal(torch.isposinf(X.cpu()), torch.isposinf(ref_X))
    assert torch.equal(torch.isneginf(X.cpu()), torch.isneginf(ref_X))
    finite = torch.isfinite(ref_X)
    _close(X.cpu()[finite], ref_X[finite], dtype)
    torch.testing.assert_close(M.cpu(), reference[1], atol=0, rtol=0)


EMPTY_SYSTEMS = [
    ((), (), 0, 3),
    ((), (), 4, 0),
    ((0,), (1,), 4, 2),
    ((2, 0), (1, 0), 0, 2),
]


@pytest.mark.triangular_solve
@pytest.mark.parametrize("system", EMPTY_SYSTEMS)
@pytest.mark.parametrize("entry", ["direct", "dispatcher"])
def test_triangular_solve_empty(system, entry):
    B, A = _inputs(*system, torch.float32)
    flags = (True, False, False)
    _assert_result(_functional(B, A, flags, entry), _reference(B, A, *flags), B, A)


@pytest.mark.triangular_solve_out
@pytest.mark.parametrize("flags", FLAGS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", ["contiguous", "transpose", "slice", "resize"])
@pytest.mark.parametrize("entry", ["direct", "dispatcher"])
def test_triangular_solve_out(flags, dtype, layout, entry, caplog):
    B, A = _inputs((2, 1), (3,), 17, 5, dtype, flags[2])
    reference = _reference(B, A, *flags)
    if layout == "resize":
        X = torch.empty(0, dtype=dtype, device=A.device)
        M = torch.empty(0, dtype=dtype, device=A.device)
    else:
        outputs = []
        for ref in reference:
            shape = ref.shape
            if layout == "transpose":
                out = torch.empty(
                    *shape[:-2], shape[-1], shape[-2], dtype=dtype, device=A.device
                )
                out = out.transpose(-2, -1)
            elif layout == "slice":
                out = torch.empty(
                    *shape[:-1], shape[-1] * 2, dtype=dtype, device=A.device
                )[..., ::2]
            else:
                out = torch.empty(shape, dtype=dtype, device=A.device)
            outputs.append(out)
        X, M = outputs
    pointers, strides = (X.data_ptr(), M.data_ptr()), (X.stride(), M.stride())
    with caplog.at_level(logging.DEBUG):
        result = _out(B, A, flags, X, M, entry)
    assert any(
        record.name.endswith(".triangular_solve")
        and record.message.endswith(" TRIANGULAR_SOLVE_OUT")
        for record in caplog.records
    ), "triangular_solve.X did not execute a FlagGems implementation"
    assert result[0] is X and result[1] is M
    if layout == "resize":
        assert X.stride() == reference[0].stride()
        assert M.stride() == reference[1].stride()
    else:
        assert (X.data_ptr(), M.data_ptr()) == pointers
        assert (X.stride(), M.stride()) == strides
    _assert_result(result, reference, B, A, functional=False)


@pytest.mark.triangular_solve_out
@pytest.mark.parametrize("entry", ["direct", "dispatcher"])
def test_triangular_solve_out_nonempty_resize(entry):
    B, A = _inputs((), (), 7, 3, torch.float32)
    X = torch.empty(1, dtype=A.dtype, device=A.device)
    M = torch.empty(1, dtype=A.dtype, device=A.device)
    flags = (True, False, False)
    with pytest.warns(UserWarning, match="output.*resized"):
        result = _out(B, A, flags, X, M, entry)
    assert result[0] is X and result[1] is M
    _assert_result(result, _reference(B, A, *flags), B, A, functional=False)


@pytest.mark.triangular_solve_out
@pytest.mark.parametrize("system", EMPTY_SYSTEMS)
def test_triangular_solve_out_empty(system):
    B, A = _inputs(*system, torch.float32)
    X = torch.empty(0, device=A.device)
    M = torch.empty(0, device=A.device)
    flags = (True, False, False)
    result = _out(B, A, flags, X, M, "direct")
    assert result[0] is X and result[1] is M
    _assert_result(result, _reference(B, A, *flags), B, A, functional=False)


@pytest.mark.triangular_solve_out
@pytest.mark.parametrize("entry", ["direct", "dispatcher"])
def test_triangular_solve_out_aliases_inputs(entry):
    B, A = _inputs((2,), (2,), 7, 3, torch.float32)
    flags = (False, True, False)
    reference = _reference(B, A, *flags)
    result = _out(B, A, flags, B, A, entry)
    assert result[0] is B and result[1] is A
    _assert_result(result, reference, B, A, functional=False)


INVALID_SHAPES = [
    ((), (4, 4)),
    ((4, 2), ()),
    ((4,), (4, 4)),
    ((4, 2), (4,)),
    ((4, 2), (4, 5)),
    ((3, 2), (4, 4)),
    ((2, 4, 2), (3, 4, 4)),
]


@pytest.mark.triangular_solve
@pytest.mark.parametrize("b_shape,a_shape", INVALID_SHAPES)
@pytest.mark.parametrize("entry", ["direct", "dispatcher"])
def test_triangular_solve_invalid_shapes(b_shape, a_shape, entry):
    B = torch.empty(b_shape, device=flag_gems.device)
    A = torch.empty(a_shape, device=flag_gems.device)
    with pytest.raises((RuntimeError, ValueError)):
        _functional(B, A, (True, False, False), entry)


@pytest.mark.triangular_solve_out
@pytest.mark.parametrize("b_shape,a_shape", INVALID_SHAPES)
def test_triangular_solve_out_invalid_shapes(b_shape, a_shape):
    B = torch.empty(b_shape, device=flag_gems.device)
    A = torch.empty(a_shape, device=flag_gems.device)
    X, M = torch.empty_like(B), torch.empty_like(A)
    with pytest.raises((RuntimeError, ValueError)):
        _out(B, A, (True, False, False), X, M, "direct")


REJECTED_DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.int32,
    torch.int64,
    torch.bool,
    torch.complex64,
    torch.complex128,
]


@pytest.mark.triangular_solve
@pytest.mark.parametrize("dtype", REJECTED_DTYPES)
@pytest.mark.parametrize("entry", ["direct", "dispatcher"])
def test_triangular_solve_rejected_dtype(dtype, entry):
    # Empty inputs avoid native initialization kernels for unsupported dtypes;
    # dtype validation still has to happen before the empty result shortcut.
    B = torch.empty(0, 2, dtype=dtype, device=flag_gems.device)
    A = torch.empty(0, 0, dtype=dtype, device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError, NotImplementedError)):
        _functional(B, A, (True, False, False), entry)


@pytest.mark.triangular_solve_out
@pytest.mark.parametrize("dtype", REJECTED_DTYPES)
def test_triangular_solve_out_rejected_dtype(dtype):
    B = torch.empty(0, 2, dtype=dtype, device=flag_gems.device)
    A = torch.empty(0, 0, dtype=dtype, device=flag_gems.device)
    X, M = torch.empty_like(B), torch.empty_like(A)
    with pytest.raises((RuntimeError, TypeError, NotImplementedError)):
        _out(B, A, (True, False, False), X, M, "direct")


@pytest.mark.triangular_solve
@pytest.mark.skipif(
    flag_gems.vendor_name in FP64_VENDORS,
    reason="float64 is supported by the triangular_solve contract on this backend",
)
def test_triangular_solve_rejected_float64():
    B = torch.empty(0, 2, dtype=torch.float64, device=flag_gems.device)
    A = torch.empty(0, 0, dtype=torch.float64, device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError, NotImplementedError)):
        _functional(B, A, (True, False, False), "direct")


@pytest.mark.triangular_solve
@pytest.mark.parametrize("mismatch", ["dtype", "device"])
def test_triangular_solve_mismatched_inputs(mismatch):
    B, A = _inputs((), (), 4, 2, torch.float32)
    B = B.to(torch.float16) if mismatch == "dtype" else B.cpu()
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        _functional(B, A, (True, False, False), "direct")


@pytest.mark.triangular_solve_out
@pytest.mark.parametrize("output", ["X", "M"])
@pytest.mark.parametrize("mismatch", ["dtype", "device"])
@pytest.mark.parametrize("entry", ["direct", "dispatcher"])
def test_triangular_solve_out_invalid_outputs(output, mismatch, entry):
    B, A = _inputs((), (), 4, 2, torch.float32)
    outputs = {"X": torch.empty_like(B), "M": torch.empty_like(A)}
    outputs[output] = (
        outputs[output].to(torch.float16)
        if mismatch == "dtype"
        else outputs[output].cpu()
    )
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        _out(B, A, (True, False, False), outputs["X"], outputs["M"], entry)


@pytest.mark.triangular_solve_out
@pytest.mark.parametrize("output", ["X", "M"])
@pytest.mark.parametrize("entry", ["direct", "dispatcher"])
def test_triangular_solve_out_rejects_expanded_output(output, entry):
    B, A = _inputs((), (), 3, 2, torch.float32)
    outputs = {"X": torch.empty_like(B), "M": torch.empty_like(A)}
    matrix = B if output == "X" else A
    outputs[output] = torch.empty(
        1, matrix.shape[-1], dtype=matrix.dtype, device=matrix.device
    ).expand(matrix.shape)
    with pytest.raises(RuntimeError, match="single memory location"):
        _out(B, A, (True, False, False), outputs["X"], outputs["M"], entry)


@pytest.mark.triangular_solve_out
@pytest.mark.parametrize("requires_grad", ["A", "B", "X", "M"])
@pytest.mark.parametrize("entry", ["direct", "dispatcher"])
def test_triangular_solve_out_rejects_autograd(requires_grad, entry):
    B, A = _inputs((), (), 4, 2, torch.float32)
    tensors = {"B": B, "A": A, "X": torch.empty_like(B), "M": torch.empty_like(A)}
    tensors[requires_grad].requires_grad_()
    with pytest.raises(RuntimeError, match="autograd|automatic differentiation"):
        _out(B, A, (True, False, False), tensors["X"], tensors["M"], entry)


@pytest.mark.triangular_solve
@pytest.mark.parametrize("flags", FLAGS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("loss_output", ["solution", "clone", "both"])
def test_triangular_solve_autograd_broadcast(flags, dtype, loss_output):
    B, A = _inputs((2, 1), (3,), 7, 3, dtype, flags[2])
    ref_A = to_reference(A).detach().clone().requires_grad_()
    ref_B = to_reference(B).detach().clone().requires_grad_()
    A.requires_grad_()
    B.requires_grad_()
    weights_X = torch.randn(2, 3, 7, 3, dtype=dtype, device=ref_A.device)
    weights_M = torch.randn(2, 3, 7, 7, dtype=dtype, device=ref_A.device)
    ref_X, ref_M = _native_reference(ref_B, ref_A, *flags)
    ref_loss = (ref_X * weights_X).sum() if loss_output != "clone" else 0
    if loss_output != "solution":
        ref_loss = ref_loss + (ref_M * weights_M).sum()
    ref_grad = torch.autograd.grad(ref_loss, (ref_B, ref_A), allow_unused=True)
    with flag_gems.use_gems(include=["triangular_solve"]):
        X, M = torch.ops.aten.triangular_solve.default(B, A, *flags)
        loss = (X * weights_X.to(X.device)).sum() if loss_output != "clone" else 0
        if loss_output != "solution":
            loss = loss + (M * weights_M.to(M.device)).sum()
        gradients = torch.autograd.grad(loss, (B, A), allow_unused=True)
    for gradient, ref_gradient in zip(gradients, ref_grad):
        if ref_gradient is None:
            assert gradient is None
        else:
            assert gradient.shape == ref_gradient.shape
            _close(gradient, ref_gradient, dtype)


@pytest.mark.triangular_solve
@pytest.mark.parametrize("flags", FLAGS)
@pytest.mark.parametrize("n,nrhs", [(7, 1), (65, 8), (129, 3)])
def test_triangular_solve_scipy_reference(flags, n, nrhs):
    # Independent CPU oracle, including vendor Torch builds without CPU BLAS.
    import numpy as np
    from scipy.linalg import solve_triangular

    B, A = _inputs((2, 1), (3,), n, nrhs, torch.float32, flags[2])
    a = np.broadcast_to(A.cpu().numpy(), (2, 3, n, n))
    b = np.broadcast_to(B.cpu().numpy(), (2, 3, n, nrhs))
    expected = np.empty_like(b)
    for index in np.ndindex(2, 3):
        expected[index] = solve_triangular(
            a[index],
            b[index],
            lower=not flags[0],
            trans=int(flags[1]),
            unit_diagonal=flags[2],
            check_finite=False,
        )
    result, coefficient = _functional(B, A, flags, "direct")
    _close(result, torch.from_numpy(expected), torch.float32)
    torch.testing.assert_close(
        coefficient.cpu(), A.cpu().expand(2, 3, n, n), atol=0, rtol=0
    )


@pytest.mark.triangular_solve
@pytest.mark.triangular_solve_out
@pytest.mark.parametrize("use_out", [False, True])
def test_triangular_solve_wide_rhs(use_out):
    # A modest allocation can exceed grid.y's 65535-program limit when
    # kernels assign a program to one or a few RHS columns.
    n, nrhs = 8, 262144
    A = torch.eye(n, device=flag_gems.device, dtype=torch.float32) * 2
    B = torch.ones(n, nrhs, device=flag_gems.device, dtype=torch.float32)
    flags = (False, False, False)
    if use_out:
        X = torch.empty(nrhs, n, device=B.device, dtype=B.dtype).T
        M = torch.empty(n, n, device=A.device, dtype=A.dtype).T
        result = _out(B, A, flags, X, M, "direct")
        assert result[0] is X and result[1] is M
    else:
        result = _functional(B, A, flags, "direct")
    _close(result[0], torch.full((n, nrhs), 0.5), torch.float32)
    torch.testing.assert_close(result[1].cpu(), A.cpu(), atol=0, rtol=0)
    assert result[0].mT.is_contiguous()
    assert result[1].mT.is_contiguous()
