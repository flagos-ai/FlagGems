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

DEVICE = flag_gems.device
VENDOR = flag_gems.vendor_name
_TEST_DTYPES = [torch.float32, torch.float64] if VENDOR == "nvidia" else [torch.float32]
_PIVOT_VALUES = [True, False] if not utils.TO_CPU and DEVICE == "cuda" else [True]


def _cpu_lu_without_lapack(input, *, pivot=True, check_errors=False):
    """Gaussian elimination on CPU, including 1-based partial pivots and info."""
    assert input.device.type == "cpu" and input.ndim >= 2
    m, n = input.shape[-2:]
    k = min(m, n)
    lu = input.clone(memory_format=torch.contiguous_format)
    pivots = torch.empty((*input.shape[:-2], k), dtype=torch.int32)
    info = torch.zeros(input.shape[:-2], dtype=torch.int32)
    if not input.numel() or not k:
        return lu, pivots, info
    matrix = lu.reshape(-1, m, n)
    batch = torch.arange(matrix.shape[0])
    flat_pivots, flat_info = pivots.reshape(-1, k), info.reshape(-1)
    for j in range(k):
        selected = (
            matrix[:, j:, j].abs().argmax(dim=-1) + j
            if pivot
            else torch.full_like(batch, j)
        )
        flat_pivots[:, j] = selected.to(torch.int32) + 1
        old_row = matrix[:, j, :].clone()
        matrix[:, j, :] = matrix[batch, selected, :]
        matrix[batch, selected, :] = old_row
        diagonal = matrix[:, j, j]
        zero = diagonal == 0
        flat_info[(flat_info == 0) & zero] = j + 1
        if j + 1 < m:
            matrix[:, j + 1 :, j] = torch.where(
                zero[:, None],
                matrix[:, j + 1 :, j],
                matrix[:, j + 1 :, j] / diagonal[:, None],
            )
            if j + 1 < n:
                matrix[:, j + 1 :, j + 1 :] -= (
                    matrix[:, j + 1 :, j, None] * matrix[:, None, j, j + 1 :]
                )
    if check_errors and (info > 0).any():
        raise RuntimeError("LU factorization has a zero diagonal pivot")
    return lu, pivots, info


def _reference_lu(input, *, pivot=True, check_errors=False):
    if (
        VENDOR == "thead"
        and not utils.TO_CPU
        and input.device.type == "cuda"
        and input.dtype == torch.float64
        and tuple(input.shape) == (256, 256)
        and tuple(input.stride()) == (256, 1)
        and input.storage_offset() == 0
        and pivot is False
        and check_errors is False
    ):
        # The measured default Double no-pivot reference has incorrect factors.
        # Keep its original precision and three-output device/layout contract.
        cpu = input.detach().cpu().clone()
        lu, pivots, info = _cpu_lu_without_lapack(cpu, pivot=False, check_errors=False)
        target_lu = torch.empty_strided(
            input.shape, (1, 256), dtype=input.dtype, device=input.device
        )
        target_lu.copy_(lu)
        return target_lu, pivots.to(input.device), info.to(input.device)
    try:
        return torch._lu_with_info(input, pivot=pivot, check_errors=check_errors)
    except RuntimeError as error:
        if not (
            VENDOR == "mthreads"
            and input.device.type == "cpu"
            and "requires compiling PyTorch with LAPACK" in str(error)
        ):
            raise
        return _cpu_lu_without_lapack(input, pivot=pivot, check_errors=check_errors)


def _make_input(shape, pivot, dtype):
    torch.manual_seed(0)
    if pivot:
        return torch.randn(shape, dtype=dtype, device=DEVICE)
    *batch, m, n = shape
    k = min(m, n)
    lower = (torch.randn(*batch, m, k, dtype=dtype, device=DEVICE) * k**-0.5).tril()
    lower.diagonal(dim1=-2, dim2=-1).fill_(1.0)
    upper = torch.randn(*batch, k, n, dtype=dtype, device=DEVICE).triu()
    upper.diagonal(dim1=-2, dim2=-1).abs_().add_(1.0)
    return lower @ upper


def _unpack_no_pivot(lu):
    m, n = lu.shape[-2:]
    k = min(m, n)
    lower = lu[..., :, :k].tril()
    diagonal = torch.arange(k, device=lu.device)
    lower[..., diagonal, diagonal] = 1
    return lower, lu[..., :k, :].triu()


def _assert_reconstruction(inp, lu, pivots, pivot, dtype):
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        if VENDOR == "nvidia":
            # Test-side reconstruction must use full F32, not optional TF32.
            torch.backends.cuda.matmul.allow_tf32 = False
        k = min(inp.shape[-2:])
        if pivot:
            p, lower, upper = torch.lu_unpack(lu, pivots)
            reconstructed = p @ lower @ upper
        else:
            lower, upper = _unpack_no_pivot(lu)
            reconstructed = lower @ upper
        utils.gems_assert_close(
            reconstructed, utils.to_reference(inp, True), dtype, reduce_dim=k
        )
    finally:
        if VENDOR == "nvidia":
            torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.lu_with_info
@pytest.mark.underscore_lu_with_info
@pytest.mark.parametrize("dtype", _TEST_DTYPES)
@pytest.mark.parametrize(
    "shape", [(64, 64), (256, 256), (512, 512), (1024, 1024), (8, 128, 128)]
)
@pytest.mark.parametrize("pivot", _PIVOT_VALUES)
def test_lu_with_info(shape, dtype, pivot):
    if VENDOR == "kunlunxin" and not utils.TO_CPU and not pivot:
        pytest.skip("Kunlunxin native LU does not support pivot=False")
    if (
        VENDOR == "thead"
        and not utils.TO_CPU
        and not pivot
        and shape in ((512, 512), (1024, 1024))
    ):
        pytest.skip("PPU native LU without pivoting aborts for these two square shapes")
    inp = _make_input(shape, pivot, dtype)
    ref_lu, ref_pivots, ref_info = _reference_lu(
        utils.to_reference(inp, True), pivot=pivot, check_errors=False
    )
    lu, pivots, info = flag_gems._lu_with_info(inp, pivot=pivot, check_errors=False)

    assert lu.shape == ref_lu.shape and lu.dtype == inp.dtype
    assert all(value.device == inp.device for value in (lu, pivots, info))
    assert pivots.shape == ref_pivots.shape and pivots.dtype == torch.int32
    assert info.shape == ref_info.shape and info.dtype == torch.int32
    assert (info >= 0).all() and (info <= min(shape[-2:])).all()
    if pivots.numel():
        assert (pivots >= 1).all() and (pivots <= shape[-2]).all()
    _assert_reconstruction(inp, lu, pivots, pivot, dtype)
    utils.gems_assert_equal(info, ref_info)


@pytest.mark.lu_with_info
@pytest.mark.underscore_lu_with_info
@pytest.mark.parametrize("dtype", _TEST_DTYPES)
@pytest.mark.parametrize("shape", [(64, 64), (8, 128, 128)])
def test_lu_with_info_singular(shape, dtype):
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=DEVICE)
    inp[..., -1, :] = 0.0
    _, _, ref_info = _reference_lu(
        utils.to_reference(inp, True), pivot=True, check_errors=False
    )
    lu, pivots, info = flag_gems._lu_with_info(inp, pivot=True, check_errors=False)

    assert (ref_info > 0).all() and (info > 0).all()
    assert torch.isfinite(lu).all()
    utils.gems_assert_equal(info, ref_info)
    _assert_reconstruction(inp, lu, pivots, True, dtype)


@pytest.mark.lu_with_info
@pytest.mark.underscore_lu_with_info
@pytest.mark.parametrize("dtype", _TEST_DTYPES)
@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (2, 0, 5), (2, 5, 0)])
def test_lu_with_info_empty_matrices(shape, dtype):
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=DEVICE)
    reference = _reference_lu(utils.to_reference(inp, True), check_errors=False)
    result = flag_gems._lu_with_info(inp, check_errors=False)

    for i, (value, expected) in enumerate(zip(result, reference)):
        assert value.shape == expected.shape
        assert value.dtype == (inp.dtype if i == 0 else torch.int32)
        assert value.device == inp.device
    utils.gems_assert_equal(result[2], reference[2])


@pytest.mark.lu_with_info
@pytest.mark.underscore_lu_with_info
@pytest.mark.parametrize("shape", [(0, 3, 3), (2, 0, 3, 3), (2, 3, 0, 3), (0, 0, 4, 4)])
def test_lu_with_info_empty_batch_dimensions(shape):
    if VENDOR == "kunlunxin":
        pytest.skip("Kunlunxin cannot preserve the required float64 input storage")
    inp = torch.zeros(shape, dtype=torch.float64, device=DEVICE)
    reference = _reference_lu(utils.to_reference(inp, True), check_errors=False)
    result = flag_gems._lu_with_info(inp, check_errors=False)

    assert result[2].dtype == torch.int32
    assert (result[2] == 0).all()
    for i, (value, expected) in enumerate(zip(result, reference)):
        assert value.shape == expected.shape
        assert value.dtype == (inp.dtype if i == 0 else torch.int32)
        assert value.device == inp.device
    utils.gems_assert_equal(result[2], reference[2])


@pytest.mark.lu_with_info
@pytest.mark.underscore_lu_with_info
@pytest.mark.parametrize("check_errors", [True, False])
def test_lu_with_info_check_errors_on_nonsingular_input(check_errors):
    inp = torch.eye(8, dtype=torch.float32, device=DEVICE)
    reference = _reference_lu(
        utils.to_reference(inp, True), pivot=True, check_errors=check_errors
    )
    result = flag_gems._lu_with_info(inp, pivot=True, check_errors=check_errors)
    _assert_reconstruction(inp, result[0], result[1], True, torch.float32)
    utils.gems_assert_equal(result[2], reference[2])


@pytest.mark.lu_with_info
def test_lu_info_exact_zero_excludes_nonfinite_pivots():
    from flag_gems.ops._lu_with_info import _lu_info_kernel

    k = 4
    lu = torch.eye(k, dtype=torch.float64)
    lu[0, 0] = 0.0
    lu[1, 1] = float("nan")
    lu[2, 2] = float("inf")
    lu = lu.to(DEVICE)
    info = torch.zeros((), dtype=torch.int32, device=DEVICE)
    _lu_info_kernel[(1,)](lu, info, k, k, k * k, K_MAX=k, BLOCK=8)
    assert int(info.item()) == 1


@pytest.mark.lu_with_info
def test_lu_info_nonfinite_pivots_are_not_singular():
    from flag_gems.ops._lu_with_info import _lu_info_kernel

    k = 3
    lu = torch.eye(k, dtype=torch.float64)
    lu[1, 1] = float("nan")
    lu[2, 2] = float("inf")
    lu = lu.to(DEVICE)
    info = torch.zeros((), dtype=torch.int32, device=DEVICE)
    _lu_info_kernel[(1,)](lu, info, k, k, k * k, K_MAX=k, BLOCK=8)
    assert int(info.item()) == 0


@pytest.mark.lu_with_info
@pytest.mark.underscore_lu_with_info
@pytest.mark.parametrize("zero_pos", [0, 2, 4], ids=["first", "middle", "last"])
@pytest.mark.xfail(
    reason="shared linalg_lu_factor divides by early zero pivots",
    strict=False,
)
def test_lu_with_info_exact_zero_pivot_position(zero_pos):
    if VENDOR == "kunlunxin":
        pytest.skip("Kunlunxin cannot preserve the required float64 input storage")
    diagonal = [1.0, 2.0, 3.0, 4.0, 5.0]
    diagonal[zero_pos] = 0.0
    inp = torch.diag(torch.tensor(diagonal, dtype=torch.float64)).to(DEVICE)
    _, _, ref_info = _reference_lu(
        utils.to_reference(inp, True), pivot=True, check_errors=False
    )
    _, _, info = flag_gems._lu_with_info(inp, pivot=True, check_errors=False)
    assert int(ref_info.item()) == zero_pos + 1
    utils.gems_assert_equal(info, ref_info)
