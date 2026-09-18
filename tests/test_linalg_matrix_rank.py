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

VENDOR_NAME = getattr(flag_gems, "vendor_name", "")
IS_ASCEND = VENDOR_NAME == "ascend"
SUPPORT_FP64 = flag_gems.runtime.device.support_fp64

# torch.linalg.matrix_rank officially accepts these four dtypes. FlagGems
# supports both real dtypes; complex inputs are deliberately rejected instead
# of being silently skipped. float64 cases only run where the device backend
# actually supports fp64 (Ascend does not).
SUPPORTED_DTYPE_CASES = [
    pytest.param(torch.float32, id="float32"),
] + ([pytest.param(torch.float64, id="float64")] if SUPPORT_FP64 else [])

OFFICIAL_DTYPE_CASES = [
    pytest.param(torch.float32, True, id="float32-supported"),
    pytest.param(
        torch.float64,
        True,
        id="float64-supported",
        marks=pytest.mark.skipif(
            not SUPPORT_FP64, reason="float64 not supported on this device"
        ),
    ),
    # On Ascend complex tensors cannot even be constructed (aclnnEye has no
    # complex support), so the rejection contract is not exercisable there.
    pytest.param(
        torch.complex64,
        False,
        id="complex64-unsupported",
        marks=pytest.mark.skipif(
            IS_ASCEND, reason="complex tensors not constructible on Ascend"
        ),
    ),
    pytest.param(
        torch.complex128,
        False,
        id="complex128-unsupported",
        marks=pytest.mark.skipif(
            IS_ASCEND, reason="complex tensors not constructible on Ascend"
        ),
    ),
]

RANK_CASES = [
    pytest.param((1, 7), 1, id="rank1-wide"),
    pytest.param((7, 2), 2, id="rank2-tall"),
    pytest.param((3, 5), 3, id="single-wide"),
    pytest.param((5, 3), 2, id="single-tall"),
    pytest.param((4, 4), 3, id="single-square"),
    pytest.param((16, 16), 15, id="small-k16"),
    pytest.param((17, 17), 16, id="serial-medium-square"),
    pytest.param((33, 33), 32, id="square-k33"),
    pytest.param((2, 4, 4), 3, id="one-batch-dimension"),
    pytest.param((2, 3, 5, 3), 2, id="multiple-batch-dimensions"),
]

EMPTY_SHAPES = [
    pytest.param((0, 0), id="zero-by-zero"),
    pytest.param((0, 3), id="zero-by-n"),
    pytest.param((3, 0), id="m-by-zero"),
    pytest.param((2, 0, 3), id="batched-zero-by-n"),
    pytest.param((2, 3, 0), id="batched-m-by-zero"),
    pytest.param((0, 3, 3), id="empty-batch"),
]


def _make_matrix_with_rank(shape, rank, dtype=torch.float32):
    matrix = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    diagonal = torch.arange(rank, device=matrix.device)
    values = torch.arange(1, rank + 1, dtype=dtype, device=matrix.device)
    matrix[..., diagonal, diagonal] = values
    return matrix


_NO_TOL = object()


def _to_reference_value(value, device):
    if isinstance(value, torch.Tensor):
        return utils.to_reference(value).to(device=device)
    return value


def _reference_matrix_rank(matrix, tol=_NO_TOL, *, upcast=False, **kwargs):
    ref_matrix = utils.to_reference(matrix, upcast).cpu()
    ref_kwargs = {
        name: _to_reference_value(value, ref_matrix.device)
        for name, value in kwargs.items()
    }
    ref_args = () if tol is _NO_TOL else (_to_reference_value(tol, ref_matrix.device),)
    reference = torch.linalg.matrix_rank(ref_matrix, *ref_args, **ref_kwargs)
    return reference.to(matrix.device)


def _assert_equal(result, reference):
    utils.gems_assert_equal(result, utils.to_reference(reference))


def _assert_output_metadata(result, matrix):
    assert result.shape == matrix.shape[:-2]
    assert result.dtype == torch.int64
    assert result.device == matrix.device


def _assert_direct_matches_native(matrix, **kwargs):
    native = _reference_matrix_rank(matrix, **kwargs)
    direct = flag_gems.linalg_matrix_rank(matrix, **kwargs)
    _assert_output_metadata(direct, matrix)
    _assert_equal(direct, native)
    return direct


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
def test_accuracy_linalg_matrix_rank_default_identity(dtype):
    matrix = torch.eye(8, dtype=dtype, device=flag_gems.device)
    expected = torch.tensor(8, dtype=torch.int64, device=matrix.device)

    result = _assert_direct_matches_native(matrix)
    _assert_equal(result, expected)


@pytest.mark.linalg_matrix_rank
@pytest.mark.skipif(not SUPPORT_FP64, reason="float64 not supported on this device")
def test_accuracy_linalg_matrix_rank_float64_preserves_small_singular_value():
    matrix = torch.tensor(
        [[1.0, 1.0], [1.0, 1.0 + 1e-10]],
        dtype=torch.float64,
        device=flag_gems.device,
    )
    expected = torch.tensor(2, dtype=torch.int64, device=matrix.device)

    result = _assert_direct_matches_native(matrix)
    _assert_equal(result, expected)


@pytest.mark.linalg_matrix_rank
@pytest.mark.skipif(not SUPPORT_FP64, reason="float64 not supported on this device")
def test_accuracy_linalg_matrix_rank_float64_tolerance_precision():
    matrix = torch.diag(
        torch.tensor(
            [1.0, 0.50000000000001],
            dtype=torch.float64,
            device=flag_gems.device,
        )
    )
    atol = torch.tensor(0.50000000000005, dtype=torch.float64, device=matrix.device)
    expected = torch.tensor(1, dtype=torch.int64, device=matrix.device)

    result = _assert_direct_matches_native(matrix, atol=atol)
    _assert_equal(result, expected)


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize(
    "dtype,k,tiny,atol",
    [
        pytest.param(torch.float32, 16, 1e-6, 1e-3, id="float32-small"),
        pytest.param(
            torch.float64,
            17,
            1e-12,
            1e-9,
            id="float64-serial",
            marks=pytest.mark.skipif(
                not SUPPORT_FP64, reason="float64 not supported on this device"
            ),
        ),
    ],
)
def test_accuracy_linalg_matrix_rank_well_separated_spectrum(dtype, k, tiny, atol):
    generator = torch.Generator(device=flag_gems.device).manual_seed(20260807)
    orthogonal = torch.linalg.qr(
        torch.randn(
            (k, k),
            dtype=dtype,
            device=flag_gems.device,
            generator=generator,
        )
    ).Q
    spectrum = torch.ones(k, dtype=dtype, device=flag_gems.device)
    spectrum[-1] = tiny
    matrix = orthogonal @ torch.diag(spectrum) @ orthogonal.mT
    expected = torch.tensor(k - 1, dtype=torch.int64, device=matrix.device)

    result = _assert_direct_matches_native(matrix, atol=atol)
    _assert_equal(result, expected)


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
def test_accuracy_linalg_matrix_rank_rank_deficient(dtype):
    matrix = _make_matrix_with_rank((5, 5), 3, dtype)
    expected = torch.tensor(3, dtype=torch.int64, device=matrix.device)

    result = _assert_direct_matches_native(matrix, atol=5e-2, hermitian=False)
    _assert_equal(result, expected)


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((2, 4, 6), id="batched-small"),
        pytest.param((513, 513), id="square-k513"),
        pytest.param((1024, 1024), id="square-k1024"),
        pytest.param((2, 513, 513), id="batched-k513"),
    ],
)
def test_accuracy_linalg_matrix_rank_nonempty_zero_matrix(dtype, shape):
    # Cover both small and large zero matrices, including batched input.
    matrix = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    expected = torch.zeros(shape[:-2], dtype=torch.int64, device=matrix.device)

    if flag_gems.vendor_name in ("metax", "hygon"):
        # The MetaX and Hygon torch native references (matrix_rank via SVD)
        # do not converge on large all-zero matrices, so compare against the
        # analytic expectation using the direct FlagGems path.
        result = flag_gems.linalg_matrix_rank(matrix, hermitian=False)
        _assert_output_metadata(result, matrix)
    else:
        result = _assert_direct_matches_native(matrix, hermitian=False)
    _assert_equal(result, expected)


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
@pytest.mark.parametrize("shape,expected_rank", RANK_CASES)
def test_accuracy_linalg_matrix_rank_shapes(dtype, shape, expected_rank):
    matrix = _make_matrix_with_rank(shape, expected_rank, dtype)
    expected = torch.full(
        matrix.shape[:-2],
        expected_rank,
        dtype=torch.int64,
        device=matrix.device,
    )

    result = _assert_direct_matches_native(matrix, atol=5e-2, hermitian=False)
    _assert_equal(result, expected)


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
@pytest.mark.parametrize(
    "shape,expected_rank",
    [
        pytest.param((3, 5), 2, id="wide"),
        pytest.param((5, 3), 2, id="tall"),
        pytest.param((2, 3, 5, 3), 2, id="multi-batch"),
    ],
)
def test_accuracy_linalg_matrix_rank_matches_adjoint(dtype, shape, expected_rank):
    matrix = _make_matrix_with_rank(shape, expected_rank, dtype)

    rank = _assert_direct_matches_native(matrix, atol=5e-2, hermitian=False)
    adjoint_rank = _assert_direct_matches_native(matrix.mH, atol=5e-2, hermitian=False)
    _assert_equal(rank, adjoint_rank)


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
def test_accuracy_linalg_matrix_rank_aah_svd_matches_hermitian(dtype):
    matrix = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=dtype,
        device=flag_gems.device,
    )
    matrix = torch.stack((matrix, matrix.roll(1, dims=0)))
    aah = matrix @ matrix.mH
    expected = torch.full((2,), 3, dtype=torch.int64, device=matrix.device)

    svd_rank = _assert_direct_matches_native(aah, atol=5e-2, hermitian=False)
    hermitian_rank = _assert_direct_matches_native(aah, atol=5e-2, hermitian=True)

    _assert_equal(svd_rank, hermitian_rank)
    _assert_equal(svd_rank, expected)


@pytest.mark.linalg_matrix_rank
@pytest.mark.linalg_matrix_rank_atol_rtol_float
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
@pytest.mark.parametrize(
    "kwargs,expected_rank",
    [
        pytest.param({}, 4, id="default"),
        pytest.param({"rtol": 0.75}, 2, id="rtol-only"),
        pytest.param({"atol": 0.75}, 3, id="atol-only"),
        pytest.param({"atol": 0.75, "rtol": 0.75}, 2, id="atol-and-rtol"),
    ],
)
def test_accuracy_linalg_matrix_rank_tolerance_combinations(
    dtype, kwargs, expected_rank
):
    spectrum = torch.tensor(
        [1.5, 1.25, 0.8, 0.1],
        dtype=dtype,
        device=flag_gems.device,
    )
    matrix = torch.diag(spectrum)

    result = _assert_direct_matches_native(matrix, **kwargs)
    expected = torch.tensor(expected_rank, dtype=torch.int64, device=matrix.device)
    _assert_equal(result, expected)


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
@pytest.mark.parametrize(
    "kwargs,expected_rank",
    [
        pytest.param({"atol": 0.75}, 3, id="python-float"),
        pytest.param({"atol": torch.tensor(0.75)}, 3, id="zero-dim-atol-tensor"),
        pytest.param({"rtol": torch.tensor(0.75)}, 2, id="zero-dim-rtol-tensor"),
    ],
)
def test_accuracy_linalg_matrix_rank_scalar_tolerance_types(
    dtype, kwargs, expected_rank
):
    spectrum = torch.tensor(
        [1.5, 1.25, 0.8, 0.1],
        dtype=dtype,
        device=flag_gems.device,
    )
    matrix = torch.diag(spectrum)
    kwargs = {
        name: (
            value.to(device=matrix.device, dtype=dtype)
            if isinstance(value, torch.Tensor)
            else value
        )
        for name, value in kwargs.items()
    }

    result = _assert_direct_matches_native(matrix, **kwargs)
    expected = torch.tensor(expected_rank, dtype=torch.int64, device=matrix.device)
    _assert_equal(result, expected)


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
def test_accuracy_linalg_matrix_rank_legacy_float_tolerance(dtype):
    spectrum = torch.tensor(
        [1.5, 1.25, 0.8, 0.1],
        dtype=dtype,
        device=flag_gems.device,
    )
    matrix = torch.diag(spectrum)
    native = _reference_matrix_rank(matrix, 0.75)

    direct = flag_gems.linalg_matrix_rank_tol(matrix, 0.75)
    _assert_equal(direct, native)


@pytest.mark.linalg_matrix_rank
@pytest.mark.linalg_matrix_rank_tol_tensor
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
def test_accuracy_linalg_matrix_rank_legacy_tensor_tolerance(dtype):
    spectrum = torch.tensor(
        [1.5, 1.25, 0.8, 0.1],
        dtype=dtype,
        device=flag_gems.device,
    )
    matrix = torch.diag(spectrum)
    tolerance = torch.tensor(0.75, dtype=dtype, device=matrix.device)
    native = _reference_matrix_rank(matrix, tolerance)

    direct = flag_gems.linalg_matrix_rank_tol(matrix, tolerance)
    _assert_equal(direct, native)


@pytest.mark.linalg_matrix_rank
@pytest.mark.linalg_matrix_rank_atol_rtol_tensor
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
def test_accuracy_linalg_matrix_rank_per_batch_tolerance(dtype):
    spectrum = torch.tensor(
        [1.5, 1.25, 0.8, 0.1],
        dtype=dtype,
        device=flag_gems.device,
    )
    matrix = torch.stack((torch.diag(spectrum), torch.diag(spectrum)))
    atol = torch.tensor([0.75, 1.3], dtype=dtype, device=matrix.device)
    expected = torch.tensor([3, 1], dtype=torch.int64, device=matrix.device)

    result = _assert_direct_matches_native(matrix, atol=atol)
    _assert_equal(result, expected)


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
def test_accuracy_linalg_matrix_rank_broadcast_tolerance(dtype):
    spectrum = torch.tensor(
        [1.5, 1.25, 0.8, 0.1],
        dtype=dtype,
        device=flag_gems.device,
    )
    base = torch.diag(spectrum)
    matrix = base.expand(2, 3, 4, 4).clone()
    atol = torch.tensor([[0.75], [1.3]], dtype=dtype, device=matrix.device)
    expected = torch.tensor(
        [[3, 3, 3], [1, 1, 1]], dtype=torch.int64, device=matrix.device
    )

    result = _assert_direct_matches_native(matrix, atol=atol)
    _assert_equal(result, expected)


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
def test_accuracy_linalg_matrix_rank_hermitian_false(dtype):
    matrix = torch.tensor(
        [[2.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=dtype,
        device=flag_gems.device,
    )
    expected = torch.tensor(2, dtype=torch.int64, device=matrix.device)

    result = _assert_direct_matches_native(matrix, atol=5e-2, hermitian=False)
    _assert_equal(result, expected)


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
def test_accuracy_linalg_matrix_rank_hermitian_true(dtype):
    matrix = torch.tensor(
        [[2.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=dtype,
        device=flag_gems.device,
    )
    expected = torch.tensor(2, dtype=torch.int64, device=matrix.device)

    result = _assert_direct_matches_native(matrix, atol=5e-2, hermitian=True)
    _assert_equal(result, expected)


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
def test_accuracy_linalg_matrix_rank_hermitian_uses_lower_triangle(dtype):
    matrix = torch.tensor(
        [[4.0, 99.0], [2.0, 1.0]],
        dtype=dtype,
        device=flag_gems.device,
    )
    expected = torch.tensor(1, dtype=torch.int64, device=matrix.device)

    result = _assert_direct_matches_native(matrix, atol=5e-2, hermitian=True)
    _assert_equal(result, expected)


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
@pytest.mark.parametrize(
    "order,rank",
    [
        pytest.param(3, 2, id="order-3"),
        pytest.param(32, 28, id="order-32"),
        pytest.param(33, 29, id="order-33"),
        pytest.param(64, 60, id="order-64"),
    ],
)
def test_accuracy_linalg_matrix_rank_hermitian_ignores_strict_upper(dtype, order, rank):
    # torch hermitian semantics: only the LOWER triangle of the input is
    # read.  Filling the strict upper triangle with huge garbage must not
    # change the result.  EVERYTHING is built on the CPU -- the low-rank
    # product, the fp32 rounding, the indexed garbage write and the
    # reference: device-side fp32 GEMM can perturb the zero eigenspace by
    # more than atol, device-side advanced-indexing writes have been
    # observed to leak into the lower triangle, and the platform native
    # hermitian result is not a reliable arbitrator for this construction.
    generator = torch.Generator().manual_seed(7)
    basis = torch.randn(order, rank, dtype=torch.float64, generator=generator)
    clean_cpu = (basis @ basis.mT).to(dtype)
    garbage_cpu = clean_cpu.clone()
    upper_rows, upper_cols = torch.triu_indices(order, order, offset=1)
    garbage_cpu[upper_rows, upper_cols] = 1.0e6
    expected = _reference_matrix_rank(
        garbage_cpu, upcast=True, atol=5e-2, rtol=0.0, hermitian=True
    )

    clean = clean_cpu.to(flag_gems.device)
    garbage = garbage_cpu.to(flag_gems.device)
    for matrix in (clean, garbage):
        result = flag_gems.linalg_matrix_rank(matrix, atol=5e-2, hermitian=True)
        _assert_output_metadata(result, matrix)
        _assert_equal(result, expected.to(flag_gems.device))


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
def test_accuracy_linalg_matrix_rank_hermitian_order_33(dtype):
    matrix = _make_matrix_with_rank((33, 33), 32, dtype)
    expected = torch.tensor(32, dtype=torch.int64, device=matrix.device)

    result = _assert_direct_matches_native(matrix, atol=5e-2, hermitian=True)
    _assert_equal(result, expected)


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
@pytest.mark.parametrize(
    "shape,expected_rank",
    [
        pytest.param((256, 256), 200, id="square-k256"),
        pytest.param((257, 257), 250, id="square-k257"),
        pytest.param((2, 300, 300), 250, id="batched-k300"),
        pytest.param((32, 32), 30, id="square-k32"),
        pytest.param((33, 33), 30, id="square-k33"),
        pytest.param((64, 64), 60, id="square-k64"),
        pytest.param((128, 128), 120, id="square-k128"),
        pytest.param((4, 32, 32), 30, id="batched-k32"),
        pytest.param((1024, 1024), 1000, id="square-k1024"),
    ],
)
def test_accuracy_linalg_matrix_rank_hermitian_shapes(dtype, shape, expected_rank):
    matrix = _make_matrix_with_rank(shape, expected_rank, dtype)
    expected = torch.full(
        matrix.shape[:-2],
        expected_rank,
        dtype=torch.int64,
        device=matrix.device,
    )

    result = _assert_direct_matches_native(matrix, atol=5e-2, hermitian=True)
    _assert_equal(result, expected)


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
@pytest.mark.parametrize(
    "shape,expected_rank",
    [
        pytest.param((513, 513), 500, id="square-k513"),
        pytest.param((1024, 1024), 1000, id="square-k1024"),
        pytest.param((600, 700), 550, id="wide-600x700"),
        pytest.param((700, 600), 550, id="tall-700x600"),
        pytest.param((2, 513, 513), 500, id="batched-k513"),
        pytest.param((129, 2048), 100, id="wide-129x2048"),
        pytest.param((2048, 129), 100, id="tall-2048x129"),
    ],
)
def test_accuracy_linalg_matrix_rank_general_large_shapes(dtype, shape, expected_rank):
    matrix = _make_matrix_with_rank(shape, expected_rank, dtype)
    expected = torch.full(
        matrix.shape[:-2],
        expected_rank,
        dtype=torch.int64,
        device=matrix.device,
    )

    result = _assert_direct_matches_native(matrix, atol=5e-2)
    _assert_equal(result, expected)


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
def test_accuracy_linalg_matrix_rank_general_dense_low_rank(dtype):
    # Check a dense, large, non-Hermitian low-rank matrix with a clear
    # spectral gap at the tolerance. Construction stays on the CPU in
    # float64 with one final rounding so device-side construction does not
    # perturb the intended zero singular values.
    generator = torch.Generator().manual_seed(4321)
    n, rank = 1024, 1000
    left, _ = torch.linalg.qr(
        torch.randn(n, n, dtype=torch.float64, generator=generator)
    )
    right, _ = torch.linalg.qr(
        torch.randn(n, n, dtype=torch.float64, generator=generator)
    )
    values = torch.zeros(n, dtype=torch.float64)
    values[:rank] = torch.linspace(rank, 1, rank, dtype=torch.float64)
    matrix = (left @ torch.diag(values) @ right.mT).to(dtype)
    reference = _reference_matrix_rank(matrix, upcast=True, atol=5e-2, rtol=0.0)
    assert reference.item() == rank  # construction sanity: gap survives rounding
    matrix = matrix.to(flag_gems.device)

    result = flag_gems.linalg_matrix_rank(matrix, atol=5e-2)
    _assert_output_metadata(result, matrix)
    _assert_equal(result, reference.to(flag_gems.device))


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
def test_accuracy_linalg_matrix_rank_hermitian_dense_low_rank(dtype):
    # Check a dense symmetric low-rank matrix with a clear spectral gap at
    # the tolerance.
    generator = torch.Generator(device=flag_gems.device).manual_seed(1234)
    n, rank = 300, 250
    basis = torch.randn(
        n, rank, dtype=dtype, device=flag_gems.device, generator=generator
    )
    weights = torch.linspace(2.0, 1.0, rank, dtype=dtype, device=flag_gems.device)
    matrix = basis @ torch.diag(weights) @ basis.mT
    expected = torch.tensor(rank, dtype=torch.int64, device=matrix.device)

    result = _assert_direct_matches_native(matrix, atol=5e-2, hermitian=True)
    _assert_equal(result, expected)


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
@pytest.mark.parametrize("shape", EMPTY_SHAPES)
def test_accuracy_linalg_matrix_rank_empty(dtype, shape):
    matrix = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    expected = torch.zeros(shape[:-2], dtype=torch.int64, device=flag_gems.device)

    result = _assert_direct_matches_native(matrix, hermitian=False)
    _assert_equal(result, expected)


@pytest.mark.linalg_matrix_rank
@pytest.mark.linalg_matrix_rank_atol_rtol_float_out
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
def test_accuracy_linalg_matrix_rank_out(dtype):
    matrix = _make_matrix_with_rank((3, 5), 2, dtype).mT
    assert not matrix.is_contiguous()
    expected = torch.tensor(2, dtype=torch.int64, device=matrix.device)
    out = torch.empty((), dtype=torch.int64, device=matrix.device)

    result = flag_gems.linalg_matrix_rank_out(
        matrix, atol=5e-2, hermitian=False, out=out
    )
    assert result.data_ptr() == out.data_ptr()
    _assert_output_metadata(result, matrix)
    _assert_equal(result, expected)


@pytest.mark.linalg_matrix_rank
@pytest.mark.linalg_matrix_rank_atol_rtol_tensor_out
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
def test_accuracy_linalg_matrix_rank_tensor_tolerance_out(dtype):
    matrix = torch.diag(
        torch.tensor([1.5, 1.25, 0.8, 0.1], dtype=dtype, device=flag_gems.device)
    )
    tolerance = torch.tensor(0.75, dtype=dtype, device=matrix.device)
    expected = torch.tensor(3, dtype=torch.int64, device=matrix.device)
    out = torch.empty((), dtype=torch.int64, device=matrix.device)

    result = flag_gems.linalg_matrix_rank_out(matrix, atol=tolerance, out=out)

    assert result.data_ptr() == out.data_ptr()
    _assert_output_metadata(result, matrix)
    _assert_equal(result, expected)


@pytest.mark.linalg_matrix_rank
@pytest.mark.linalg_matrix_rank_out
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
def test_accuracy_linalg_matrix_rank_legacy_float_tolerance_out(dtype):
    matrix = torch.diag(
        torch.tensor([1.5, 1.25, 0.8, 0.1], dtype=dtype, device=flag_gems.device)
    )
    expected = torch.tensor(3, dtype=torch.int64, device=matrix.device)
    out = torch.empty((), dtype=torch.int64, device=matrix.device)

    result = flag_gems.linalg_matrix_rank_tol_out(matrix, 0.75, out=out)

    assert result.data_ptr() == out.data_ptr()
    _assert_output_metadata(result, matrix)
    _assert_equal(result, expected)


@pytest.mark.linalg_matrix_rank
@pytest.mark.linalg_matrix_rank_out_tol_tensor
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
def test_accuracy_linalg_matrix_rank_legacy_tensor_tolerance_out(dtype):
    matrix = torch.diag(
        torch.tensor([1.5, 1.25, 0.8, 0.1], dtype=dtype, device=flag_gems.device)
    )
    tolerance = torch.tensor(0.75, dtype=dtype, device=matrix.device)
    expected = torch.tensor(3, dtype=torch.int64, device=matrix.device)
    out = torch.empty((), dtype=torch.int64, device=matrix.device)

    result = flag_gems.linalg_matrix_rank_tol_out(matrix, tolerance, out=out)

    assert result.data_ptr() == out.data_ptr()
    _assert_output_metadata(result, matrix)
    _assert_equal(result, expected)


@pytest.mark.linalg_matrix_rank
def test_accuracy_linalg_matrix_rank_out_wrong_dtype():
    matrix = torch.eye(3, dtype=torch.float32, device=flag_gems.device)
    out = torch.empty(0, dtype=torch.bool, device=matrix.device)

    with pytest.raises(RuntimeError, match="safely castable"):
        flag_gems.linalg_matrix_rank_out(matrix, out=out)


@pytest.mark.linalg_matrix_rank
def test_accuracy_linalg_matrix_rank_out_wrong_device():
    matrix = torch.eye(3, dtype=torch.float32, device=flag_gems.device)
    if matrix.device.type == "cpu":
        pytest.skip("wrong-device out test requires an accelerator input")
    out = torch.empty(0, dtype=torch.int64, device="cpu")

    with pytest.raises(RuntimeError, match="same device"):
        flag_gems.linalg_matrix_rank_out(matrix, out=out)


@pytest.mark.linalg_matrix_rank
def test_accuracy_linalg_matrix_rank_out_wrong_shape_warns_and_resizes():
    matrix = torch.eye(3, dtype=torch.float32, device=flag_gems.device)
    out = torch.empty((3,), dtype=torch.int64, device=matrix.device)
    expected = torch.tensor(3, dtype=torch.int64, device=matrix.device)

    with pytest.warns(UserWarning, match="output.*was resized"):
        result = flag_gems.linalg_matrix_rank_out(matrix, out=out)

    assert result.data_ptr() == out.data_ptr()
    assert out.shape == torch.Size([])
    _assert_equal(result, expected)


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("dtype,is_supported", OFFICIAL_DTYPE_CASES)
def test_accuracy_linalg_matrix_rank_official_dtype_contract(dtype, is_supported):
    matrix = torch.eye(3, dtype=dtype, device=flag_gems.device)

    if is_supported:
        result = _assert_direct_matches_native(matrix)
        expected = torch.tensor(3, dtype=torch.int64, device=matrix.device)
        _assert_equal(result, expected)
    else:
        with pytest.raises(NotImplementedError, match="float32 and float64"):
            flag_gems.linalg_matrix_rank(matrix)


@pytest.mark.linalg_matrix_rank
@pytest.mark.skipif(SUPPORT_FP64, reason="device supports native float64")
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((5, 1), id="k1"),
        pytest.param((5, 2), id="k2"),
        pytest.param((5, 5), id="square-k5"),
        pytest.param((40, 40), id="square-k40"),
        pytest.param((600, 600), id="general"),
    ],
)
def test_accuracy_linalg_matrix_rank_fp64_rejected(shape):
    # Unsupported float64 inputs must fail through the public API for every shape.
    matrix = torch.randn(shape, dtype=torch.float64, device=flag_gems.device)

    with pytest.raises(NotImplementedError, match="float64"):
        flag_gems.linalg_matrix_rank(matrix)


@pytest.mark.linalg_matrix_rank
@pytest.mark.skipif(SUPPORT_FP64, reason="device supports native float64")
@pytest.mark.parametrize("hermitian", [False, True])
def test_accuracy_linalg_matrix_rank_no_native_fp64_k32(hermitian):
    # Public regression for k=32 on devices without native float64.
    k, rank = 32, 30
    generator = torch.Generator().manual_seed(29)
    spectrum = torch.arange(1, rank + 1, dtype=torch.float64)
    left = torch.linalg.qr(torch.randn(k, k, generator=generator, dtype=torch.float64))[
        0
    ]
    if hermitian:
        base = (left[:, :rank] * spectrum) @ left[:, :rank].mT
    else:
        right = torch.linalg.qr(
            torch.randn(k, k, generator=generator, dtype=torch.float64)
        )[0]
        base = (left[:, :rank] * spectrum) @ right[:, :rank].mT
    matrix = base.float().to(flag_gems.device)
    reference = _reference_matrix_rank(
        matrix, upcast=True, hermitian=hermitian, atol=5e-2
    )

    result = flag_gems.linalg_matrix_rank(matrix, hermitian=hermitian, atol=5e-2)
    _assert_equal(result, reference)


@pytest.mark.linalg_matrix_rank
def test_accuracy_linalg_matrix_rank_rejects_complex_tolerance():
    matrix = torch.eye(3, dtype=torch.float32, device=flag_gems.device)
    tolerance_device = torch.device("cpu") if IS_ASCEND else matrix.device
    complex_tol = torch.tensor(1 + 0j, device=tolerance_device)

    with pytest.raises(RuntimeError, match="complex type"):
        flag_gems.linalg_matrix_rank(matrix, atol=complex_tol)


@pytest.mark.linalg_matrix_rank
@pytest.mark.skipif(
    not IS_ASCEND,
    reason="Ascend-specific numerical coverage",
)
@pytest.mark.parametrize(
    "shape,rank,hermitian",
    [
        # Cover square, tall, wide and batched matrices across relevant sizes.
        pytest.param((33, 33), 16, False, id="square-k33"),
        pytest.param((256, 64), 32, False, id="tall-k64"),
        pytest.param((64, 512), 32, False, id="wide-k64"),
        pytest.param((1024, 8), 4, False, id="long-k8"),
        pytest.param((128, 128), 60, False, id="square-k128"),
        pytest.param((256, 512), 100, False, id="wide-k256"),
        pytest.param((2, 100, 100), 40, False, id="batched-k100"),
        pytest.param((200, 200), 80, True, id="hermitian-k200"),
    ],
)
def test_accuracy_linalg_matrix_rank_ascend_slow_decay_spectrum(shape, rank, hermitian):
    # Slowly decaying low-rank spectra must match the CPU fp64 oracle when
    # evaluated through the public interface with fp32 tolerance semantics.
    generator = torch.Generator().manual_seed(2026)
    *batch, m, n = shape
    if hermitian:
        basis = torch.linalg.qr(
            torch.randn(m, m, generator=generator, dtype=torch.float64)
        )[0]
        values = torch.cat(
            [
                torch.logspace(0, -4, rank, dtype=torch.float64),
                torch.zeros(m - rank, dtype=torch.float64),
            ]
        )
        matrix = ((basis * values) @ basis.mT).to(torch.float32)
    else:
        left = torch.linalg.qr(
            torch.randn(*batch, m, rank, generator=generator, dtype=torch.float64)
        )[0]
        right = torch.linalg.qr(
            torch.randn(*batch, n, rank, generator=generator, dtype=torch.float64)
        )[0]
        values = torch.logspace(0, -4, rank, dtype=torch.float64)
        matrix = ((left * values) @ right.mT).to(torch.float32)

    matrix = matrix.to(device=flag_gems.device)
    rtol = max(m, n) * torch.finfo(torch.float32).eps
    reference = _reference_matrix_rank(
        matrix, upcast=True, atol=0.0, rtol=rtol, hermitian=hermitian
    )

    result = flag_gems.linalg_matrix_rank(matrix, hermitian=hermitian)
    _assert_output_metadata(result, matrix)
    _assert_equal(result, reference.to(device=matrix.device))


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
@pytest.mark.parametrize(
    "shape,fill_row,fill_col",
    [
        pytest.param((16, 5), 15, 4, id="tall-small"),
        pytest.param((5, 16), 4, 15, id="wide-small"),
        pytest.param((50, 40), 49, 39, id="tall"),
        pytest.param((40, 50), 39, 49, id="wide"),
    ],
)
def test_accuracy_linalg_matrix_rank_nonsquare_tail_energy(
    dtype, shape, fill_row, fill_col
):
    # Put the final independent row or column outside the leading square
    # block. This ensures rectangular inputs do not lose rank information
    # carried only by their non-square tail.
    m, n = shape
    matrix = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    diagonal = torch.arange(min(m, n) - 1, device=matrix.device)
    matrix[diagonal, diagonal] = 1.0
    matrix[fill_row, fill_col] = 5.0
    expected = torch.tensor(min(m, n), dtype=torch.int64, device=matrix.device)

    result = _assert_direct_matches_native(matrix)
    _assert_equal(result, expected)


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
@pytest.mark.parametrize(
    "shape,rank",
    [
        pytest.param((16, 5), 3, id="tall-16x5"),
        pytest.param((5, 16), 3, id="wide-5x16"),
        pytest.param((32, 8), 4, id="tall-32x8"),
        pytest.param((8, 32), 4, id="wide-8x32"),
        pytest.param((33, 64), 16, id="wide"),
        pytest.param((64, 33), 16, id="tall"),
        pytest.param((48, 60), 24, id="wide-48x60"),
    ],
)
def test_accuracy_linalg_matrix_rank_nonsquare_lowrank(dtype, shape, rank):
    # Random non-square low-rank matrices with singular values from 1 down
    # to 1e-4. Construct in fp64 and round once so the fp64 reference with
    # fp32-semantics tolerance remains stable.
    generator = torch.Generator().manual_seed(17)
    m, n = shape
    left = torch.linalg.qr(
        torch.randn(m, rank, generator=generator, dtype=torch.float64)
    )[0]
    right = torch.linalg.qr(
        torch.randn(n, rank, generator=generator, dtype=torch.float64)
    )[0]
    values = torch.logspace(0, -4, rank, dtype=torch.float64)
    matrix = ((left * values) @ right.mT).to(dtype).to(flag_gems.device)

    rtol = max(m, n) * torch.finfo(torch.float32).eps
    reference = _reference_matrix_rank(matrix, upcast=True, atol=0.0, rtol=rtol)

    result = flag_gems.linalg_matrix_rank(matrix)
    _assert_output_metadata(result, matrix)
    _assert_equal(result, reference.to(device=matrix.device))


@pytest.mark.linalg_matrix_rank
# Include representative Hermitian orders on each supported backend.
@pytest.mark.parametrize(
    "k", [3, 33, 65, 128, 257] if IS_ASCEND else [33, 65, 128, 257]
)
def test_accuracy_linalg_matrix_rank_hermitian_strict_threshold(k):
    # PyTorch counts eigenvalues using a strict |lambda| > tolerance test.
    device = flag_gems.device

    def diag_case(values, atol, rtol):
        matrix = torch.diag(values).to(torch.float32).to(device)
        reference = _reference_matrix_rank(
            matrix, upcast=True, hermitian=True, atol=atol, rtol=rtol
        )
        result = flag_gems.linalg_matrix_rank(
            matrix, hermitian=True, atol=atol, rtol=rtol
        )
        _assert_equal(result, reference.to(device))

    # negative tie: lambda == -tol must NOT be counted
    diag_case(torch.tensor([1.0, -0.5] + [0.0] * (k - 2)), 0.5, 0.0)
    # one ULP below the tie: pred(-tol) MUST be counted.  An arithmetic
    # threshold shift (-tol*(1+2eps)) lands 2-3 ULP below -tol depending on
    # tol's mantissa (2 ULP for tol=0.5, 3 ULP for tol=0.75) and would
    # wrongly skip this eigenvalue; only the mirrored zero-pivot tie
    # convention (zero pivot -> tiny positive) counts it exactly.
    pred_half = torch.nextafter(
        torch.tensor(-0.5, dtype=torch.float32),
        torch.tensor(float("-inf"), dtype=torch.float32),
    ).item()
    diag_case(torch.tensor([1.0, pred_half] + [0.0] * (k - 2)), 0.5, 0.0)
    pred_3q = torch.nextafter(
        torch.tensor(-0.75, dtype=torch.float32),
        torch.tensor(float("-inf"), dtype=torch.float32),
    ).item()
    diag_case(torch.tensor([1.0, pred_3q] + [0.0] * (k - 2)), 0.75, 0.0)
    # smallest-NORMAL tolerance: an arithmetic threshold shift rounds back
    # onto -tol itself, wrongly skipping lambda = -2*tiny.
    tiny = torch.finfo(torch.float32).tiny
    diag_case(torch.tensor([1.0, -2.0 * tiny] + [0.0] * (k - 2)), tiny, 0.0)
    # The equivalent tie at the smallest subnormal is intentionally not
    # tested because subnormal support differs across device backends.
    # positive tie: lambda == +tol must NOT be counted
    diag_case(torch.tensor([0.5, -1.0] + [0.0] * (k - 2)), 0.5, 0.0)
    # atol == rtol == 0 on a nonzero rank-deficient spectrum: #{|lam| > 0}
    diag_case(torch.tensor([1.0, -2.0] + [0.0] * (k - 2)), 0.0, 0.0)
    # all-zero spectrum with atol == rtol == 0
    diag_case(torch.zeros(k), 0.0, 0.0)

    # dense (rotated) ties with margins above the fp32 noise floor
    generator = torch.Generator().manual_seed(k)
    basis = torch.linalg.qr(
        torch.randn(k, k, generator=generator, dtype=torch.float64)
    )[0]
    values = torch.zeros(k, dtype=torch.float64)
    values[:3] = torch.tensor([1.0, -0.5, 0.5])
    matrix = ((basis * values) @ basis.mT).float().to(device)
    for atol, expected_rank in [(0.49, 3), (0.51, 1)]:
        reference = _reference_matrix_rank(
            matrix, upcast=True, hermitian=True, atol=atol, rtol=0.0
        )
        assert reference.item() == expected_rank  # construction sanity
        result = flag_gems.linalg_matrix_rank(
            matrix, hermitian=True, atol=atol, rtol=0.0
        )
        _assert_equal(result, reference.to(device))

    # batch + per-batch tensor tolerance
    matrix = torch.stack(
        [
            torch.diag(torch.tensor([1.0, -0.5] + [0.0] * (k - 2))),
            torch.diag(torch.tensor([1.0, -0.5] + [0.0] * (k - 2))),
        ]
    ).float()
    atol = torch.tensor([0.5, 0.6], device=device)
    rtol = torch.zeros(2, device=device)
    reference = _reference_matrix_rank(
        matrix, upcast=True, hermitian=True, atol=atol.cpu(), rtol=rtol.cpu()
    )
    result = flag_gems.linalg_matrix_rank(
        matrix.to(device), hermitian=True, atol=atol, rtol=rtol
    )
    _assert_equal(result, reference.to(device))


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("k", [3, 33, 65, 257])
@pytest.mark.parametrize("hermitian", [False, True])
def test_accuracy_linalg_matrix_rank_negative_tolerances(k, hermitian):
    # torch does not clamp the tolerance: tol = max(atol, rtol*sigma_max).
    # tol < 0 is reachable only when BOTH atol < 0 and rtol < 0, and then
    # every singular value (>= 0) exceeds tol -> rank == k for a nonzero
    # matrix; a zero matrix still gives 0 because rtol*0 == 0 lifts tol to
    # max(atol, 0) == 0.  A negative atol alone is harmless (rtol*sigma_max
    # >= 0 dominates the max).
    device = flag_gems.device
    values = torch.zeros(k)
    values[:2] = torch.tensor([1.0, -0.5])
    matrix = torch.diag(values).to(torch.float32).to(device)
    zero = torch.zeros(k, k, dtype=torch.float32, device=device)

    def check(mat, atol, rtol):
        reference = _reference_matrix_rank(
            mat, upcast=True, hermitian=hermitian, atol=atol, rtol=rtol
        )
        result = flag_gems.linalg_matrix_rank(
            mat, hermitian=hermitian, atol=atol, rtol=rtol
        )
        _assert_equal(result, reference.to(device))

    # negative atol alone: behaves as atol = 0
    check(matrix, -1.0, 0.0)
    # negative rtol alone: behaves as rtol = 0
    check(matrix, 0.0, -1.0)
    # both negative: tol < 0 -> every singular value counts -> full rank
    check(matrix, -1.0, -1.0)
    # both negative on a zero matrix: tol == 0 -> rank 0
    check(zero, -1.0, -1.0)

    if hermitian:
        # hermitian reads ONLY the lower triangle: strict-upper garbage is
        # invisible, so the both-negative fixup must test the lower
        # triangle for "nonzero" -- torch returns 0 here, not k.
        upper_only = torch.zeros(k, k, dtype=torch.float32, device=device)
        upper_only[0, k - 1] = 1.0
        reference = _reference_matrix_rank(
            upper_only, upcast=True, hermitian=True, atol=-1.0, rtol=-1.0
        )
        assert reference.item() == 0  # construction sanity
        result = flag_gems.linalg_matrix_rank(
            upper_only, hermitian=True, atol=-1.0, rtol=-1.0
        )
        _assert_equal(result, reference.to(device))
        # ... and a lower-triangle-only nonzero DOES give full rank under
        # tol < 0 (eigenvalues +1/-1 of the symmetrized matrix).
        lower_only = torch.zeros(k, k, dtype=torch.float32, device=device)
        lower_only[k - 1, 0] = 1.0
        reference = _reference_matrix_rank(
            lower_only, upcast=True, hermitian=True, atol=-1.0, rtol=-1.0
        )
        assert reference.item() == k  # construction sanity
        result = flag_gems.linalg_matrix_rank(
            lower_only, hermitian=True, atol=-1.0, rtol=-1.0
        )
        _assert_equal(result, reference.to(device))

        # Repeat the same three regimes with tensor tolerances: strict-upper
        # garbage -> 0, lower-only nonzero -> k, true zero -> 0.
        mixed = torch.stack([upper_only, lower_only, zero])
        atol_t = torch.full((3,), -1.0, device=device)
        rtol_t = torch.full((3,), -1.0, device=device)
        reference = _reference_matrix_rank(
            mixed, upcast=True, hermitian=True, atol=atol_t.cpu(), rtol=rtol_t.cpu()
        )
        assert reference.tolist() == [0, k, 0]  # construction sanity
        result = flag_gems.linalg_matrix_rank(
            mixed, hermitian=True, atol=atol_t, rtol=rtol_t
        )
        _assert_equal(result, reference.to(device))

    # batch + per-batch tensor tolerances mixing all three regimes
    batch = torch.stack([matrix, zero, matrix])
    atol_t = torch.tensor([-1.0, -1.0, 0.0], device=device)
    rtol_t = torch.tensor([-1.0, -1.0, 0.0], device=device)
    reference = _reference_matrix_rank(
        batch, upcast=True, hermitian=hermitian, atol=atol_t.cpu(), rtol=rtol_t.cpu()
    )
    result = flag_gems.linalg_matrix_rank(
        batch, hermitian=hermitian, atol=atol_t, rtol=rtol_t
    )
    _assert_equal(result, reference.to(device))


@pytest.mark.linalg_matrix_rank
@pytest.mark.skipif(not IS_ASCEND, reason="Ascend-specific numerical coverage")
@pytest.mark.parametrize("shape", [(129, 64), (64, 129), (192, 64), (64, 192)])
def test_accuracy_linalg_matrix_rank_ascend_long_rectangular(shape):
    # Tall and wide non-power-of-two shapes must match the CPU reference.
    m, n = shape
    rank = 17
    generator = torch.Generator().manual_seed(2026)
    left = torch.linalg.qr(
        torch.randn(m, rank, generator=generator, dtype=torch.float64)
    )[0]
    right = torch.linalg.qr(
        torch.randn(n, rank, generator=generator, dtype=torch.float64)
    )[0]
    values = torch.logspace(0, -4, rank, dtype=torch.float64)
    matrix = ((left * values) @ right.mT).to(torch.float32).to(flag_gems.device)

    rtol = max(m, n) * torch.finfo(torch.float32).eps
    reference = _reference_matrix_rank(matrix, upcast=True, atol=0.0, rtol=rtol)
    result = flag_gems.linalg_matrix_rank(matrix)
    _assert_output_metadata(result, matrix)
    _assert_equal(result, reference.to(device=matrix.device))


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("k,expect_rank", [(65, 1), (128, 1), (257, 1), (513, 1)])
def test_accuracy_linalg_matrix_rank_hermitian_deflated_spectrum(k, expect_rank):
    # Strongly deflated Hermitian spectra must return a finite, exact rank.
    generator = torch.Generator().manual_seed(k)
    basis = torch.linalg.qr(
        torch.randn(k, k, generator=generator, dtype=torch.float64)
    )[0]
    values = torch.zeros(k, dtype=torch.float64)
    values[:4] = torch.tensor([1.0, -0.5, 0.5, -0.25])
    matrix = ((basis * values) @ basis.mT).float().to(flag_gems.device)

    reference = _reference_matrix_rank(
        matrix, upcast=True, hermitian=True, atol=0.51, rtol=0.0
    )
    assert reference.item() == expect_rank  # 1.0 only; +/-0.5/-0.25 excluded
    result = flag_gems.linalg_matrix_rank(matrix, hermitian=True, atol=0.51, rtol=0.0)
    _assert_equal(result, reference.to(flag_gems.device))


@pytest.mark.linalg_matrix_rank
@pytest.mark.skipif(not IS_ASCEND, reason="Ascend-specific numerical coverage")
@pytest.mark.parametrize(
    "shape,rank,hermitian,kind",
    [
        # General matrices around important size boundaries.
        pytest.param((128, 128), 60, False, "gapped", id="general-k128"),
        pytest.param((255, 255), 120, False, "gapped", id="general-k255"),
        pytest.param((256, 256), 120, False, "slowdecay", id="general-k256"),
        pytest.param((256, 512), 120, False, "slowdecay", id="general-k256-wide"),
        pytest.param((512, 256), 120, False, "slowdecay", id="general-k256-tall"),
        pytest.param((2, 256, 256), 120, False, "slowdecay", id="general-k256-batched"),
        # Hermitian matrices around the same boundaries.
        pytest.param((64, 64), 30, True, "slowdecay", id="hermitian-k64"),
        pytest.param((65, 65), 30, True, "gapped", id="hermitian-k65"),
        pytest.param((256, 256), 120, True, "slowdecay", id="hermitian-k256"),
    ],
)
def test_accuracy_linalg_matrix_rank_ascend_shape_boundaries(
    shape, rank, hermitian, kind
):
    # slowdecay uses singular values from 1 to 1e-4; gapped uses singular
    # values in {1, 0}. Compare both families with an fp64 CPU reference
    # using fp32 default-tolerance semantics.
    generator = torch.Generator().manual_seed(2026 + rank)
    *batch, m, n = shape
    if hermitian:
        basis = torch.linalg.qr(
            torch.randn(m, m, generator=generator, dtype=torch.float64)
        )[0]
        if kind == "gapped":
            nonzero = torch.ones(rank, dtype=torch.float64)
        else:
            nonzero = torch.logspace(0, -4, rank, dtype=torch.float64)
        full = torch.cat([nonzero, torch.zeros(m - rank, dtype=torch.float64)])
        matrix = ((basis * full) @ basis.mT).to(torch.float32)
    else:
        left = torch.linalg.qr(
            torch.randn(*batch, m, rank, generator=generator, dtype=torch.float64)
        )[0]
        right = torch.linalg.qr(
            torch.randn(*batch, n, rank, generator=generator, dtype=torch.float64)
        )[0]
        if kind == "gapped":
            values = torch.ones(rank, dtype=torch.float64)
        else:
            values = torch.logspace(0, -4, rank, dtype=torch.float64)
        matrix = ((left * values) @ right.mT).to(torch.float32)

    matrix = matrix.to(flag_gems.device)
    rtol = max(m, n) * torch.finfo(torch.float32).eps
    reference = _reference_matrix_rank(
        matrix, upcast=True, atol=0.0, rtol=rtol, hermitian=hermitian
    )
    result = flag_gems.linalg_matrix_rank(matrix, hermitian=hermitian)
    _assert_output_metadata(result, matrix)
    _assert_equal(result, reference.to(flag_gems.device))


@pytest.mark.linalg_matrix_rank
@pytest.mark.skipif(not IS_ASCEND, reason="Ascend-specific numerical coverage")
@pytest.mark.parametrize("k", [300, 513])
def test_accuracy_linalg_matrix_rank_hermitian_ignores_strict_upper_large(k):
    # torch hermitian semantics read only the lower triangle, including for
    # large matrices; strict-upper garbage must not affect the result.
    generator = torch.Generator().manual_seed(k)
    lower = torch.tril(torch.randn(k, k, generator=generator))
    matrix = lower.clone()
    matrix.masked_fill_(torch.triu(torch.ones(k, k, dtype=torch.bool), 1), 1e6)
    matrix = matrix.to(flag_gems.device)
    ref_matrix = utils.to_reference(matrix, True).cpu()
    ref_matrix = torch.tril(ref_matrix) + torch.tril(ref_matrix, -1).mT
    reference = _reference_matrix_rank(ref_matrix, hermitian=True)
    result = flag_gems.linalg_matrix_rank(matrix, hermitian=True)
    _assert_equal(result, reference.to(flag_gems.device))


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("log10_scale", [20, -30], ids=lambda s: f"1e{s}")
@pytest.mark.parametrize(
    "shape,hermitian",
    [
        pytest.param((16, 16), False, id="general-small"),
        pytest.param((65, 65), True, id="hermitian"),
        pytest.param((513, 513), False, id="general"),
    ],
)
def test_accuracy_linalg_matrix_rank_extreme_scales(shape, hermitian, log10_scale):
    # Exercise magnitudes whose squares overflow or underflow fp32. Matrix
    # rank is invariant under nonzero scalar multiplication, so both scales
    # must match the fp64 CPU reference with fp32 default-tolerance semantics.
    scale = 10.0**log10_scale
    generator = torch.Generator().manual_seed(sum(shape) + log10_scale)
    matrix = torch.randn(*shape, generator=generator)
    if hermitian:
        matrix = matrix + matrix.mT
    else:
        # A raw Gaussian's smallest singular value can sit within fp32
        # factorization noise of the rtol threshold (observed: the rank
        # flapping by one across backends at 1e20); diagonal dominance
        # makes the full-rank verdict unambiguous at any fp32 noise level.
        matrix = matrix + 3.0 * (shape[-1] ** 0.5) * torch.eye(shape[-1])
    matrix = (matrix * scale).to(flag_gems.device)
    rtol = max(shape[-2:]) * torch.finfo(torch.float32).eps
    reference = _reference_matrix_rank(
        matrix, upcast=True, hermitian=hermitian, rtol=rtol
    )
    result = flag_gems.linalg_matrix_rank(matrix, hermitian=hermitian)
    _assert_equal(result, reference.to(flag_gems.device))


@pytest.mark.linalg_matrix_rank
def test_accuracy_linalg_matrix_rank_mixed_magnitude_batch():
    # One batch mixing 1e20, 1e-30 and exactly-zero matrices: per-batch
    # scaling must keep each matrix independent of the others' magnitude.
    batch = torch.stack(
        [
            torch.eye(65) * 1e20,
            torch.eye(65) * 1e-30,
            torch.zeros(65, 65),
        ]
    ).to(flag_gems.device)
    expected = torch.tensor([65, 65, 0], dtype=torch.int64)
    result = flag_gems.linalg_matrix_rank(batch, hermitian=True)
    _assert_equal(result, expected.to(flag_gems.device))


@pytest.mark.linalg_matrix_rank
@pytest.mark.skipif(not SUPPORT_FP64, reason="float64 tolerances need native FP64")
@pytest.mark.skipif(IS_ASCEND, reason="Ascend backend has its own implementation")
def test_accuracy_linalg_matrix_rank_tolerance_precision():
    # A float64/Python-float tolerance next to a singular value of a
    # float32 matrix must decide the rank at ITS OWN precision, not after
    # rounding to float32: atol = 0.5 - 1e-16 rounds back to 0.5 in fp32
    # and would wrongly exclude the 0.5 eigenvalue (strict threshold).
    matrix = torch.diag(torch.tensor([1.0, 0.5, 0.0, 0.0])).float()
    matrix = matrix.to(flag_gems.device)
    for atol in (0.5, 0.5 - 1e-16, 0.5 + 1e-16):
        reference = _reference_matrix_rank(
            matrix, upcast=True, hermitian=True, atol=atol
        )
        result = flag_gems.linalg_matrix_rank(matrix, hermitian=True, atol=atol)
        _assert_equal(result, reference.to(flag_gems.device))

    # nextafter boundary around a non-exact fp32 singular value (0.1).
    sigma = torch.tensor(0.1, dtype=torch.float32).item()
    matrix = torch.diag(torch.tensor([1.0, sigma])).float().to(flag_gems.device)
    just_below = torch.nextafter(torch.tensor(sigma), torch.tensor(0.0)).item()
    for atol, expected_rank in ((just_below, 2), (sigma, 1)):
        result = flag_gems.linalg_matrix_rank(
            matrix, hermitian=True, atol=atol, rtol=0.0
        )
        reference = _reference_matrix_rank(
            matrix, upcast=True, hermitian=True, atol=atol, rtol=0.0
        )
        _assert_equal(result, reference)


@pytest.mark.linalg_matrix_rank
@pytest.mark.skipif(not SUPPORT_FP64, reason="float64 not supported on this device")
@pytest.mark.skipif(IS_ASCEND, reason="Ascend backend has its own implementation")
@pytest.mark.parametrize("k", [33, 65, 257])
def test_accuracy_linalg_matrix_rank_fp64_critical_spectrum(k):
    # Check fp64 eigenvalues within about 1e-12 of the relative-tolerance
    # threshold. A rotated spectrum prevents this from degenerating into a
    # diagonal-only case; the CPU fp64 result is the oracle.
    generator = torch.Generator().manual_seed(k)
    basis = torch.linalg.qr(
        torch.randn(k, k, generator=generator, dtype=torch.float64)
    )[0]
    values = torch.zeros(k, dtype=torch.float64)
    values[0] = 1.0
    values[1] = 0.5
    matrix = ((basis * values) @ basis.mT).to(flag_gems.device)
    for delta, expected_rank in ((1e-12, 1), (-1e-12, 2)):
        rtol = 0.5 * (1.0 + delta)  # threshold ~= 0.5 * (1 +/- 1e-12)
        reference = _reference_matrix_rank(matrix, hermitian=True, atol=0.0, rtol=rtol)
        assert reference.item() == expected_rank  # construction sanity
        result = flag_gems.linalg_matrix_rank(
            matrix, hermitian=True, atol=0.0, rtol=rtol
        )
        _assert_equal(result, reference.to(flag_gems.device))


def _make_rotated_hermitian_spectrum(k, values, seed):
    # Hermitian matrix with the given eigenvalues, rotated by a random
    # orthogonal basis. Build on CPU in fp64 and round to fp32 once because
    # device-side fp64 is not portable.
    generator = torch.Generator().manual_seed(seed)
    basis = torch.linalg.qr(
        torch.randn(k, k, generator=generator, dtype=torch.float64)
    )[0]
    spectrum = torch.zeros(k, dtype=torch.float64)
    spectrum[: len(values)] = torch.tensor(values, dtype=torch.float64)
    return ((basis * spectrum) @ basis.mT).float()


@pytest.mark.linalg_matrix_rank
@pytest.mark.skipif(IS_ASCEND, reason="Ascend backend has its own implementation")
def test_accuracy_linalg_matrix_rank_large_hermitian_ignores_strict_upper():
    # Garbage in the strict upper triangle, which torch does not read for
    # hermitian=True, must not change the result. Build both matrices on CPU
    # because device-side advanced indexing is not portable.
    k = 1024
    generator = torch.Generator().manual_seed(7)
    basis = torch.randn(k, 30, dtype=torch.float64, generator=generator)
    clean_cpu = ((basis @ basis.mT) / k).float()  # rank 30, spectrum O(1)
    garbage_cpu = clean_cpu.clone()
    rows, cols = torch.triu_indices(k, k, offset=1)
    garbage_cpu[rows, cols] = 1.0e6

    clean = clean_cpu.to(flag_gems.device)
    garbage = garbage_cpu.to(flag_gems.device)
    # atol sits far above the fp32 rounding noise of the zero eigenspace.
    reference = _reference_matrix_rank(
        garbage_cpu, upcast=True, hermitian=True, atol=5e-2
    )
    clean_rank = flag_gems.linalg_matrix_rank(clean, hermitian=True, atol=5e-2)
    garbage_rank = flag_gems.linalg_matrix_rank(garbage, hermitian=True, atol=5e-2)
    _assert_equal(garbage_rank, clean_rank)
    _assert_equal(garbage_rank, reference.to(flag_gems.device))


@pytest.mark.linalg_matrix_rank
@pytest.mark.skipif(IS_ASCEND, reason="Ascend backend has its own implementation")
@pytest.mark.parametrize("rank", [1, 7])
def test_accuracy_linalg_matrix_rank_large_hermitian_deflated(rank):
    # A large positive-semidefinite matrix with only a few nonzero
    # eigenvalues must preserve its analytic rank.
    k = 1024
    generator = torch.Generator().manual_seed(rank)
    factor = torch.randn(k, rank, dtype=torch.float64, generator=generator)
    matrix = ((factor @ factor.mT) / k).float().to(flag_gems.device)
    reference = _reference_matrix_rank(matrix, upcast=True, hermitian=True, atol=5e-2)
    assert reference.item() == rank  # construction sanity
    result = flag_gems.linalg_matrix_rank(matrix, hermitian=True, atol=5e-2)
    _assert_equal(result, reference.to(flag_gems.device))


@pytest.mark.linalg_matrix_rank
@pytest.mark.skipif(IS_ASCEND, reason="Ascend backend has its own implementation")
@pytest.mark.parametrize("log10_scale", [20, -30], ids=lambda s: f"1e{s}")
def test_accuracy_linalg_matrix_rank_large_hermitian_extreme_scales(log10_scale):
    # Large Hermitian inputs at 1e20 and 1e-30 must retain scale-invariant
    # rank semantics.
    k = 768
    scale = 10.0**log10_scale
    generator = torch.Generator().manual_seed(k + log10_scale)
    matrix = torch.randn(k, k, generator=generator, dtype=torch.float64)
    matrix = ((matrix + matrix.mT) * scale).float().to(flag_gems.device)
    rtol = k * torch.finfo(torch.float32).eps
    reference = _reference_matrix_rank(matrix, upcast=True, hermitian=True, rtol=rtol)
    result = flag_gems.linalg_matrix_rank(matrix, hermitian=True)
    _assert_equal(result, reference.to(flag_gems.device))


@pytest.mark.linalg_matrix_rank
@pytest.mark.skipif(IS_ASCEND, reason="Ascend backend has its own implementation")
def test_accuracy_linalg_matrix_rank_large_hermitian_critical_spectrum():
    # A dense rotated eigenvalue just above or below the threshold must be
    # classified consistently with the CPU fp64 reference.
    k = 1024
    for delta, expected_rank in ((1e-4, 2), (-1e-4, 1)):
        matrix = _make_rotated_hermitian_spectrum(
            k, [1.0, 0.5 * (1.0 + delta)], seed=int(delta * 1e8)
        ).to(flag_gems.device)
        reference = _reference_matrix_rank(
            matrix, upcast=True, hermitian=True, atol=0.5, rtol=0.0
        )
        assert reference.item() == expected_rank  # construction sanity
        result = flag_gems.linalg_matrix_rank(
            matrix, hermitian=True, atol=0.5, rtol=0.0
        )
        _assert_equal(result, reference.to(flag_gems.device))


@pytest.mark.linalg_matrix_rank
@pytest.mark.skipif(IS_ASCEND, reason="Ascend backend has its own implementation")
def test_accuracy_linalg_matrix_rank_large_hermitian_batched():
    # Batched large matrices with different ranks must be evaluated
    # independently (batch 0 rank 50, batch 1 full rank).
    k = 768
    low_rank = _make_rotated_hermitian_spectrum(k, list(range(1, 51)), seed=1)
    generator = torch.Generator().manual_seed(2)
    dense = torch.randn(k, k, generator=generator, dtype=torch.float64)
    dense = (dense + dense.mT).float()
    batch = torch.stack([low_rank, dense]).to(flag_gems.device)
    reference = _reference_matrix_rank(batch, upcast=True, hermitian=True, atol=5e-2)
    assert reference.tolist() == [50, k]  # construction sanity
    result = flag_gems.linalg_matrix_rank(batch, hermitian=True, atol=5e-2)
    _assert_equal(result, reference.to(flag_gems.device))


@pytest.mark.linalg_matrix_rank
@pytest.mark.skipif(IS_ASCEND, reason="Ascend backend has its own implementation")
def test_accuracy_linalg_matrix_rank_large_hermitian_repeatable():
    # A near-threshold large input must produce the same public result over
    # repeated calls. The eigenvalue is far enough from the threshold to
    # make the expected rank unambiguous.
    k = 1024
    matrix = _make_rotated_hermitian_spectrum(k, [1.0, 0.5 * 1.0001], seed=3).to(
        flag_gems.device
    )
    reference = _reference_matrix_rank(
        matrix, upcast=True, hermitian=True, atol=0.5, rtol=0.0
    )
    assert reference.item() == 2  # construction sanity
    for _ in range(20):
        result = flag_gems.linalg_matrix_rank(
            matrix, hermitian=True, atol=0.5, rtol=0.0
        )
        _assert_equal(result, reference.to(flag_gems.device))


@pytest.mark.linalg_matrix_rank
def test_accuracy_linalg_matrix_rank_empty_validates_tolerances():
    # Native torch runs its same-device / non-complex tolerance checks
    # BEFORE its empty-input return; FlagGems must match, so an empty
    # matrix still rejects invalid tensor tolerances.
    matrix = torch.empty(2, 0, 5, device=flag_gems.device)
    result = flag_gems.linalg_matrix_rank(matrix)
    reference = _reference_matrix_rank(matrix)
    _assert_equal(result, reference.to(flag_gems.device))

    # Complex tensors are not constructible on Ascend (torch_npu rejects
    # them); building the tolerance on CPU still exercises the complex
    # rejection because the dtype check precedes the device check.
    complex_device = torch.device("cpu") if IS_ASCEND else matrix.device
    complex_tol = torch.ones(2, dtype=torch.complex64, device=complex_device)
    with pytest.raises(RuntimeError, match="complex"):
        flag_gems.linalg_matrix_rank(matrix, atol=complex_tol)
    if matrix.device.type != "cpu":
        cpu_tol = torch.ones(2)
        with pytest.raises(RuntimeError, match="same device"):
            flag_gems.linalg_matrix_rank(matrix, rtol=cpu_tol)
