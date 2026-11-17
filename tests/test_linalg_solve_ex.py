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

SOLVE_EX_DTYPES = [torch.float32]
if flag_gems.runtime.device.support_fp64:
    SOLVE_EX_DTYPES.append(torch.float64)


def _make_inputs(batch_shape, n, nrhs, dtype, *, requires_grad=False):
    torch.manual_seed(0)
    base = torch.randn(
        batch_shape + (n, n), dtype=dtype, device=flag_gems.device
    )
    eye = torch.eye(n, dtype=dtype, device=flag_gems.device)
    matrix = (base @ base.mT + eye * n).detach().requires_grad_(requires_grad)
    rhs = torch.randn(
        batch_shape + (n, nrhs), dtype=dtype, device=flag_gems.device
    )
    rhs.requires_grad_(requires_grad)
    return matrix, rhs


def _reference_solve_ex(matrix, rhs):
    if flag_gems.vendor_name == "mthreads" and matrix.device.type == "cpu":
        # Fixtures calling this path are nonsingular or empty. The singular
        # error contract is checked separately through the candidate.
        import numpy as np

        values = np.linalg.solve(matrix.detach().numpy(), rhs.detach().numpy())
        result = torch.from_numpy(values)
        info = torch.zeros(matrix.shape[:-2], dtype=torch.int32, device=matrix.device)
        return result, info
    return torch.linalg.solve_ex(matrix, rhs)


def _assert_outputs(actual, expected, dtype, device):
    result, info = actual
    assert result.device == info.device == device
    expected_result, expected_info = expected
    utils.gems_assert_close(result, expected_result, dtype)
    utils.gems_assert_equal(info, expected_info)
    assert result.shape == expected_result.shape
    assert result.dtype == dtype
    assert info.shape == expected_info.shape
    assert info.dtype == torch.int32


@pytest.mark.linalg_solve_ex
@pytest.mark.parametrize("batch_shape,n,nrhs", [((), 4, 1), ((), 8, 4), ((3,), 4, 2), ((2, 4), 8, 1)])
@pytest.mark.parametrize("dtype", SOLVE_EX_DTYPES)
def test_linalg_solve_ex(batch_shape, n, nrhs, dtype):
    matrix, rhs = _make_inputs(batch_shape, n, nrhs, dtype)
    matrix_before = matrix.clone()
    rhs_before = rhs.clone()
    ref_matrix = utils.to_reference(matrix, True)
    ref_rhs = utils.to_reference(rhs, True)
    expected = _reference_solve_ex(ref_matrix, ref_rhs)

    actual = flag_gems.linalg_solve_ex(matrix, rhs)

    _assert_outputs(actual, expected, dtype, matrix.device)
    utils.gems_assert_equal(matrix, utils.to_reference(matrix_before))
    utils.gems_assert_equal(rhs, utils.to_reference(rhs_before))
    assert actual[0] is not matrix
    assert actual[0] is not rhs
    input_storage_pointers = {
        matrix.untyped_storage().data_ptr(),
        rhs.untyped_storage().data_ptr(),
    }
    assert actual[0].untyped_storage().data_ptr() not in input_storage_pointers
    assert actual[1].untyped_storage().data_ptr() not in input_storage_pointers


@pytest.mark.linalg_solve_ex
@pytest.mark.parametrize("dtype", SOLVE_EX_DTYPES)
def test_linalg_solve_ex_broadcast(dtype):
    matrix, _ = _make_inputs((2, 1), 4, 2, dtype)
    torch.manual_seed(1)
    rhs = torch.randn((1, 3, 4, 2), dtype=dtype, device=flag_gems.device)
    ref_matrix = utils.to_reference(matrix, True)
    ref_rhs = utils.to_reference(rhs, True)
    expected = _reference_solve_ex(ref_matrix, ref_rhs)

    actual = flag_gems.linalg_solve_ex(matrix, rhs)

    expected_result, expected_info = expected
    _assert_outputs(actual, (expected_result, expected_info.expand(2, 3)), dtype, matrix.device)
    assert actual[0].shape == (2, 3, 4, 2)
    assert actual[1].shape == (2, 3)


@pytest.mark.linalg_solve_ex
def test_linalg_solve_ex_empty_batch():
    matrix = torch.empty((0, 4, 4), dtype=torch.float32, device=flag_gems.device)
    rhs = torch.empty((0, 4, 2), dtype=torch.float32, device=flag_gems.device)
    expected = _reference_solve_ex(
        utils.to_reference(matrix, True), utils.to_reference(rhs, True)
    )

    actual = flag_gems.linalg_solve_ex(matrix, rhs)

    _assert_outputs(actual, expected, torch.float32, matrix.device)


@pytest.mark.linalg_solve_ex
def test_linalg_solve_ex_check_errors_raises_for_singular_matrix():
    matrix = torch.zeros((4, 4), dtype=torch.float32, device=flag_gems.device)
    rhs = torch.ones((4, 1), dtype=torch.float32, device=flag_gems.device)

    with pytest.raises(torch.linalg.LinAlgError):
        flag_gems.linalg_solve_ex(matrix, rhs, check_errors=True)


@pytest.mark.linalg_solve_ex
def test_linalg_solve_ex_autograd_matches_reference():
    matrix, rhs = _make_inputs((), 4, 2, torch.float32, requires_grad=True)
    ref_matrix = utils.to_reference(matrix, True).detach().requires_grad_(True)
    ref_rhs = utils.to_reference(rhs, True).detach().requires_grad_(True)
    expected, expected_info = _reference_solve_ex(ref_matrix, ref_rhs)
    if flag_gems.vendor_name == "mthreads" and ref_matrix.device.type == "cpu":
        # For X = A⁻¹B and loss ||X||², dB = A⁻ᵀ(2X) and dA = -dB Xᵀ.
        rhs_grad, _ = _reference_solve_ex(ref_matrix.mT, 2 * expected)
        expected_grads = (-rhs_grad @ expected.mT, rhs_grad)
    else:
        expected_grads = torch.autograd.grad(
            expected.square().sum(), (ref_matrix, ref_rhs)
        )

    result, info = flag_gems.linalg_solve_ex(matrix, rhs)
    assert result.device == info.device == matrix.device
    actual_grads = torch.autograd.grad(result.square().sum(), (matrix, rhs))

    assert actual_grads[0].device == matrix.device
    assert actual_grads[1].device == rhs.device
    assert result.requires_grad
    assert not info.requires_grad
    utils.gems_assert_equal(info, expected_info)
    utils.gems_assert_close(actual_grads[0], expected_grads[0], torch.float32)
    utils.gems_assert_close(actual_grads[1], expected_grads[1], torch.float32)


@pytest.mark.linalg_solve_ex
@pytest.mark.parametrize(
    "matrix_shape,rhs_shape",
    [((4, 3), (4, 1)), ((4, 4), (3, 1)), ((4,), (4, 1))],
)
def test_linalg_solve_ex_invalid_shapes(matrix_shape, rhs_shape):
    torch.manual_seed(0)
    matrix = torch.randn(matrix_shape, device=flag_gems.device)
    rhs = torch.randn(rhs_shape, device=flag_gems.device)

    with pytest.raises((ValueError, RuntimeError)):
        flag_gems.linalg_solve_ex(matrix, rhs)


@pytest.mark.linalg_solve_ex
@pytest.mark.parametrize("dtype", [torch.float16, torch.int32])
def test_linalg_solve_ex_rejects_unsupported_dtype(dtype):
    matrix = torch.eye(4, dtype=dtype, device=flag_gems.device)
    rhs = torch.ones((4, 1), dtype=dtype, device=flag_gems.device)

    with pytest.raises((AssertionError, RuntimeError)):
        flag_gems.linalg_solve_ex(matrix, rhs)


@pytest.mark.linalg_solve_ex
def test_linalg_solve_ex_right_side_is_rejected_by_current_kernel():
    matrix, rhs = _make_inputs((), 4, 4, torch.float32)

    with pytest.raises(NotImplementedError, match="right=True"):
        flag_gems.linalg_solve_ex(matrix, rhs, left=False)


SOLVE_EX_SHAPES = [(4, 4), (8, 8), (16, 16), (32, 32)]


def _make_solve_ex_inputs(shape, k, dtype, device, left=True):
    """Create well-conditioned A and random B for solve_ex test."""
    n = shape[-1]
    A = torch.randn(*shape, dtype=dtype, device=device)
    eye = torch.eye(n, dtype=dtype, device=device)
    batch_dims = [1] * (A.ndim - 2)
    if batch_dims:
        eye = eye.view(*batch_dims, n, n)
    A = A @ A.mH + eye * n
    B_shape = shape[:-1] + (k,) if left else shape[:-2] + (k, n)
    B = torch.randn(*B_shape, dtype=dtype, device=device)
    return A, B


@pytest.mark.linalg_solve_ex
@pytest.mark.parametrize("shape", SOLVE_EX_SHAPES)
@pytest.mark.parametrize("dtype", SOLVE_EX_DTYPES)
def test_linalg_solve_ex_square_rhs(shape, dtype):
    n, k = shape
    A, B = _make_solve_ex_inputs((n, n), k, dtype, flag_gems.device)

    ref_A = utils.to_reference(A)
    ref_B = utils.to_reference(B)
    ref_out, ref_info = _reference_solve_ex(ref_A, ref_B)
    res_out, res_info = flag_gems.linalg_solve_ex(A, B)

    _assert_outputs((res_out, res_info), (ref_out, ref_info), dtype, A.device)


@pytest.mark.linalg_solve_ex
@pytest.mark.parametrize("batch_shape", [(1,), (3,), (4,), (2, 4)])
@pytest.mark.parametrize("n", [4, 8, 16])
@pytest.mark.parametrize("k", [1, 4])
@pytest.mark.parametrize("dtype", SOLVE_EX_DTYPES)
def test_linalg_solve_ex_batched(batch_shape, n, k, dtype):
    shape_A = batch_shape + (n, n)
    A, B = _make_solve_ex_inputs(shape_A, k, dtype, flag_gems.device)

    ref_A = utils.to_reference(A)
    ref_B = utils.to_reference(B)
    ref_out, ref_info = _reference_solve_ex(ref_A, ref_B)
    res_out, res_info = flag_gems.linalg_solve_ex(A, B)

    _assert_outputs((res_out, res_info), (ref_out, ref_info), dtype, A.device)
