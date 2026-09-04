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

import importlib

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

VENDOR_NAME = getattr(flag_gems, "vendor_name", "")
IS_ASCEND = VENDOR_NAME == "ascend"
IS_THEAD = VENDOR_NAME == "thead"
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
    pytest.param((16, 16), 15, id="small-jacobi-boundary"),
    pytest.param((17, 17), 16, id="serial-medium-square"),
    pytest.param((33, 33), 32, id="blocked-square"),
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


def _to_reference_value(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    return value


def _assert_equal(result, reference):
    utils.gems_assert_equal(result, utils.to_reference(reference))


def _native_matrix_rank(matrix, **kwargs):
    ref_matrix = utils.to_reference(matrix, True)
    ref_kwargs = {
        name: _to_reference_value(value, ref_matrix.device)
        for name, value in kwargs.items()
    }

    # THead's device-side Hermitian solver is not a reliable correctness
    # oracle: batched inputs may hit an unsupported syev entry point, while
    # single rank-deficient matrices can miscount repeated zero eigenvalues.
    # Compute every Hermitian reference on CPU so the FlagGems device
    # implementation can still be exercised against native PyTorch semantics.
    if IS_THEAD and kwargs.get("hermitian", False) and ref_matrix.device.type != "cpu":
        cpu_kwargs = {
            name: _to_reference_value(value, torch.device("cpu"))
            for name, value in ref_kwargs.items()
        }
        return torch.linalg.matrix_rank(ref_matrix.cpu(), **cpu_kwargs).to(
            ref_matrix.device
        )
    return torch.linalg.matrix_rank(ref_matrix, **ref_kwargs)


def _assert_output_metadata(result, matrix):
    assert result.shape == matrix.shape[:-2]
    assert result.dtype == torch.int64
    assert result.device == matrix.device


def _assert_direct_matches_native(matrix, **kwargs):
    native = _native_matrix_rank(matrix, **kwargs)
    direct = flag_gems.linalg_matrix_rank(matrix, **kwargs)
    _assert_output_metadata(direct, matrix)
    _assert_equal(direct, native)
    return direct


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
def test_linalg_matrix_rank_default_identity(dtype):
    matrix = torch.eye(8, dtype=dtype, device=flag_gems.device)
    expected = torch.tensor(8, dtype=torch.int64, device=matrix.device)

    result = _assert_direct_matches_native(matrix)
    _assert_equal(result, expected)


@pytest.mark.linalg_matrix_rank
@pytest.mark.skipif(not SUPPORT_FP64, reason="float64 not supported on this device")
def test_linalg_matrix_rank_float64_preserves_small_singular_value():
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
def test_linalg_matrix_rank_float64_tolerance_precision():
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
def test_linalg_matrix_rank_well_separated_spectrum(dtype, k, tiny, atol):
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
def test_linalg_matrix_rank_does_not_call_torch_decomposition(dtype, monkeypatch):
    matrix = _make_matrix_with_rank((4, 4), 3, dtype)
    expected = _native_matrix_rank(matrix, atol=5e-2)

    def forbidden_decomposition(*args, **kwargs):
        raise AssertionError("FlagGems matrix_rank called a Torch decomposition")

    for name in ("svd", "svdvals", "eigh", "eigvalsh"):
        monkeypatch.setattr(
            torch.linalg,
            name,
            forbidden_decomposition,
        )

    result = flag_gems.linalg_matrix_rank(matrix, atol=5e-2)
    hermitian_result = flag_gems.linalg_matrix_rank(
        matrix,
        atol=5e-2,
        hermitian=True,
    )
    _assert_equal(result, expected)
    _assert_equal(hermitian_result, expected)


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
def test_linalg_matrix_rank_rank_deficient(dtype):
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
        pytest.param((513, 513), id="bidiag-k513"),
        pytest.param((1024, 1024), id="bidiag-k1024"),
        pytest.param((2, 513, 513), id="bidiag-batched"),
    ],
)
def test_linalg_matrix_rank_nonempty_zero_matrix(dtype, shape):
    # The k >= 513 shapes exercise the unblocked bidiagonalization path,
    # whose zero-matrix shortcut must still hand defined state to the
    # (unconditionally launched) final Sturm kernel.
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
def test_linalg_matrix_rank_shapes(dtype, shape, expected_rank):
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
def test_linalg_matrix_rank_matches_adjoint(dtype, shape, expected_rank):
    matrix = _make_matrix_with_rank(shape, expected_rank, dtype)

    rank = _assert_direct_matches_native(matrix, atol=5e-2, hermitian=False)
    adjoint_rank = _assert_direct_matches_native(matrix.mH, atol=5e-2, hermitian=False)
    _assert_equal(rank, adjoint_rank)


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
def test_linalg_matrix_rank_aah_svd_matches_hermitian(dtype):
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
def test_linalg_matrix_rank_tolerance_combinations(dtype, kwargs, expected_rank):
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
def test_linalg_matrix_rank_scalar_tolerance_types(dtype, kwargs, expected_rank):
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
def test_linalg_matrix_rank_legacy_float_tolerance(dtype):
    spectrum = torch.tensor(
        [1.5, 1.25, 0.8, 0.1],
        dtype=dtype,
        device=flag_gems.device,
    )
    matrix = torch.diag(spectrum)
    ref_matrix = utils.to_reference(matrix, True)
    native = torch.linalg.matrix_rank(ref_matrix, 0.75)

    direct = flag_gems.linalg_matrix_rank_tol(matrix, 0.75)
    _assert_equal(direct, native)


@pytest.mark.linalg_matrix_rank
@pytest.mark.linalg_matrix_rank_tol_tensor
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
def test_linalg_matrix_rank_legacy_tensor_tolerance(dtype):
    spectrum = torch.tensor(
        [1.5, 1.25, 0.8, 0.1],
        dtype=dtype,
        device=flag_gems.device,
    )
    matrix = torch.diag(spectrum)
    tolerance = torch.tensor(0.75, dtype=dtype, device=matrix.device)
    ref_matrix = utils.to_reference(matrix, True)
    native = torch.linalg.matrix_rank(
        ref_matrix, _to_reference_value(tolerance, ref_matrix.device)
    )

    direct = flag_gems.linalg_matrix_rank_tol(matrix, tolerance)
    _assert_equal(direct, native)


@pytest.mark.linalg_matrix_rank
@pytest.mark.linalg_matrix_rank_atol_rtol_tensor
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
def test_linalg_matrix_rank_per_batch_tolerance(dtype):
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
def test_linalg_matrix_rank_broadcast_tolerance(dtype):
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
def test_linalg_matrix_rank_hermitian_false(dtype):
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
def test_linalg_matrix_rank_hermitian_true(dtype):
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
def test_linalg_matrix_rank_hermitian_uses_lower_triangle(dtype):
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
        pytest.param(3, 2, id="fused-k3"),
        pytest.param(32, 28, id="fused-k32"),
        pytest.param(33, 29, id="padded-k33"),
        pytest.param(64, 60, id="padded-k64"),
    ],
)
def test_linalg_matrix_rank_hermitian_ignores_strict_upper(dtype, order, rank):
    # torch hermitian semantics: only the LOWER triangle of the input is
    # read.  Filling the strict upper triangle with huge garbage must not
    # change the result.  EVERYTHING is built on the CPU -- the low-rank
    # product, the fp32 rounding, the indexed garbage write and the
    # reference: device-side fp32 GEMM can perturb the zero eigenspace by
    # more than atol, device-side advanced-indexing writes have been
    # observed to leak into the lower triangle, and the platform native
    # hermitian path is not a reliable arbitrator.  Covers the fused
    # (k <= 32) and padded (33..64) tridiagonalization paths; the 2x2
    # closed form is above.
    generator = torch.Generator().manual_seed(7)
    basis = torch.randn(order, rank, dtype=torch.float64, generator=generator)
    clean_cpu = (basis @ basis.mT).to(dtype)
    garbage_cpu = clean_cpu.clone()
    upper_rows, upper_cols = torch.triu_indices(order, order, offset=1)
    garbage_cpu[upper_rows, upper_cols] = 1.0e6
    expected = torch.linalg.matrix_rank(
        garbage_cpu.double(), atol=5e-2, rtol=0.0, hermitian=True
    )

    clean = clean_cpu.to(flag_gems.device)
    garbage = garbage_cpu.to(flag_gems.device)
    for matrix in (clean, garbage):
        result = flag_gems.linalg_matrix_rank(matrix, atol=5e-2, hermitian=True)
        _assert_output_metadata(result, matrix)
        _assert_equal(result, expected.to(flag_gems.device))


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
def test_linalg_matrix_rank_hermitian_blocked(dtype):
    matrix = _make_matrix_with_rank((33, 33), 32, dtype)
    expected = torch.tensor(32, dtype=torch.int64, device=matrix.device)

    result = _assert_direct_matches_native(matrix, atol=5e-2, hermitian=True)
    _assert_equal(result, expected)


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
@pytest.mark.parametrize(
    "shape,expected_rank",
    [
        pytest.param((256, 256), 200, id="tridiag-square"),
        pytest.param((257, 257), 250, id="tridiag-odd-order"),
        pytest.param((2, 300, 300), 250, id="tridiag-batched"),
        pytest.param((32, 32), 30, id="tridiag-k32"),
        pytest.param((33, 33), 30, id="tridiag-k33"),
        pytest.param((64, 64), 60, id="tridiag-k64"),
        pytest.param((128, 128), 120, id="tridiag-k128"),
        pytest.param((4, 32, 32), 30, id="tridiag-batched-small"),
        pytest.param((1024, 1024), 1000, id="tridiag-k1024"),
    ],
)
def test_linalg_matrix_rank_hermitian_tridiag(dtype, shape, expected_rank):
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
        pytest.param((513, 513), 500, id="bidiag-k513"),
        pytest.param((1024, 1024), 1000, id="bidiag-k1024"),
        pytest.param((600, 700), 550, id="bidiag-wide"),
        pytest.param((700, 600), 550, id="bidiag-tall"),
        pytest.param((2, 513, 513), 500, id="bidiag-batched"),
        pytest.param((129, 2048), 100, id="bidiag-longrows-wide"),
        pytest.param((2048, 129), 100, id="bidiag-longrows-tall"),
    ],
)
def test_linalg_matrix_rank_bidiag(dtype, shape, expected_rank):
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
def test_linalg_matrix_rank_bidiag_dense(dtype):
    # Dense non-hermitian low-rank matrices exercise the two-sided
    # Householder bidiagonalization + Golub-Kahan Sturm-count path
    # (min(m, n) > 512) with a clear spectral gap at the tolerance.
    # Construction stays on the CPU in float64 with a single final
    # rounding: a device-side fp32 QR+GEMM chain can push the zero-space
    # noise above atol on some platforms.  The arbitrator is the CPU
    # float64 oracle, not the platform native SVD (which has been observed
    # to disagree with the analytic expectation on borderline spectra).
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
    reference = torch.linalg.matrix_rank(matrix.double(), atol=5e-2, rtol=0.0)
    assert reference.item() == rank  # construction sanity: gap survives rounding
    matrix = matrix.to(flag_gems.device)

    result = flag_gems.linalg_matrix_rank(matrix, atol=5e-2)
    _assert_output_metadata(result, matrix)
    _assert_equal(result, reference.to(flag_gems.device))


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
def test_linalg_matrix_rank_hermitian_tridiag_dense(dtype):
    # Dense symmetric low-rank matrices exercise the Householder
    # tridiagonalization + Sturm-count path (k >= 256) with a clear
    # spectral gap at the tolerance.
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
def test_linalg_matrix_rank_empty(dtype, shape):
    matrix = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    expected = torch.zeros(shape[:-2], dtype=torch.int64, device=flag_gems.device)

    result = _assert_direct_matches_native(matrix, hermitian=False)
    _assert_equal(result, expected)


@pytest.mark.linalg_matrix_rank
@pytest.mark.linalg_matrix_rank_atol_rtol_float_out
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
def test_linalg_matrix_rank_out(dtype):
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
def test_linalg_matrix_rank_tensor_tolerance_out(dtype):
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
def test_linalg_matrix_rank_legacy_float_tolerance_out(dtype):
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
def test_linalg_matrix_rank_legacy_tensor_tolerance_out(dtype):
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
def test_linalg_matrix_rank_out_wrong_dtype():
    matrix = torch.eye(3, dtype=torch.float32, device=flag_gems.device)
    out = torch.empty(0, dtype=torch.bool, device=matrix.device)

    with pytest.raises(RuntimeError, match="safely castable"):
        flag_gems.linalg_matrix_rank_out(matrix, out=out)


@pytest.mark.linalg_matrix_rank
def test_linalg_matrix_rank_out_wrong_device():
    matrix = torch.eye(3, dtype=torch.float32, device=flag_gems.device)
    if matrix.device.type == "cpu":
        pytest.skip("wrong-device out test requires an accelerator input")
    out = torch.empty(0, dtype=torch.int64, device="cpu")

    with pytest.raises(RuntimeError, match="same device"):
        flag_gems.linalg_matrix_rank_out(matrix, out=out)


@pytest.mark.linalg_matrix_rank
def test_linalg_matrix_rank_out_wrong_shape_warns_and_resizes():
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
def test_linalg_matrix_rank_official_dtype_contract(dtype, is_supported):
    matrix = torch.eye(3, dtype=dtype, device=flag_gems.device)

    if is_supported:
        result = _assert_direct_matches_native(matrix)
        expected = torch.tensor(3, dtype=torch.int64, device=matrix.device)
        _assert_equal(result, expected)
    else:
        with pytest.raises(NotImplementedError, match="float32 and float64"):
            flag_gems.linalg_matrix_rank(matrix)


@pytest.mark.linalg_matrix_rank
@pytest.mark.skipif(
    not IS_ASCEND, reason="fp64 rejection is specific to the Ascend backend"
)
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((5, 1), id="k1"),
        pytest.param((5, 2), id="k2"),
        pytest.param((5, 5), id="fused"),
        pytest.param((40, 40), id="padded"),
        pytest.param((600, 600), id="bidiag"),
    ],
)
def test_linalg_matrix_rank_fp64_rejected(shape):
    # fp64 must fail fast with a clear error for EVERY shape class, before
    # any shape dispatch (k=1/2 used to slip past the check and die inside
    # the Triton compiler with MLIRCompilationError).
    matrix = torch.randn(shape, dtype=torch.float64, device=flag_gems.device)

    with pytest.raises(NotImplementedError, match="float64"):
        flag_gems.linalg_matrix_rank(matrix)


@pytest.mark.linalg_matrix_rank
def test_linalg_matrix_rank_rejects_complex_tolerance():
    matrix = torch.eye(3, dtype=torch.float32, device=flag_gems.device)
    complex_tol = torch.tensor(1 + 0j, device=matrix.device)

    with pytest.raises(RuntimeError, match="complex type"):
        flag_gems.linalg_matrix_rank(matrix, atol=complex_tol)


@pytest.mark.linalg_matrix_rank
@pytest.mark.skipif(
    not IS_ASCEND,
    reason="exact-path coverage is specific to the Ascend backend",
)
@pytest.mark.parametrize(
    "shape,rank,hermitian",
    [
        # Gram band: non-hermitian 33..64 and long-dimension k <= 64
        pytest.param((33, 33), 16, False, id="gram-band-k33"),
        pytest.param((256, 64), 32, False, id="gram-band-tall"),
        pytest.param((64, 512), 32, False, id="gram-band-wide"),
        pytest.param((1024, 8), 4, False, id="gram-band-long-dim-k8"),
        # QR band: 64 < k <= 512
        pytest.param((128, 128), 60, False, id="qr-band-k128"),
        pytest.param((256, 512), 100, False, id="qr-band-wide"),
        pytest.param((2, 100, 100), 40, False, id="qr-band-batched"),
        pytest.param((200, 200), 80, True, id="qr-band-hermitian"),
    ],
)
def test_linalg_matrix_rank_exact_path(shape, rank, hermitian, monkeypatch):
    # Exact reference path coverage: the exact path is the DEFAULT dispatch
    # (the Gram/unpivoted-QR fast bands are opt-in via
    # FLAGGEMS_MR_FAST_PATH=1), so this test runs the SVD-accurate
    # Golub-Kahan bidiagonalization + df64 Sturm count on every band.
    # Slowly-decaying low-rank spectra (singular values from 1
    # geometrically down to 1e-4) are where the Gram path overestimates
    # rank (sigma^2 domain) and the unpivoted QR miscounts near the
    # tolerance (|R_ii| != sigma_i); the exact path must match an fp64
    # reference with fp32-semantics tolerance exactly.  Clear any FAST_PATH
    # leftover from the environment so the test really pins the exact
    # (default) dispatch.
    monkeypatch.delenv("FLAGGEMS_MR_FAST_PATH", raising=False)
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
    reference = torch.linalg.matrix_rank(
        matrix.cpu().to(torch.float64), atol=0.0, rtol=rtol, hermitian=hermitian
    )

    result = flag_gems.linalg_matrix_rank(matrix, hermitian=hermitian)
    _assert_output_metadata(result, matrix)
    _assert_equal(result, reference.to(device=matrix.device))


@pytest.mark.linalg_matrix_rank
@pytest.mark.skipif(not IS_ASCEND, reason="Ascend-specific dispatch coverage")
def test_linalg_matrix_rank_fast_path_dispatch(monkeypatch):
    # FLAGGEMS_MR_FAST_PATH=1 dispatch coverage: spy on the four band
    # launchers to record which path each shape takes.  Spying at the
    # launcher level (instead of relying on the fast path producing a
    # wrong result) keeps the test deterministic -- the fast paths are
    # exact on the clear random spectra used here.
    mod = importlib.import_module(flag_gems.linalg_matrix_rank.__module__)
    calls = []

    def _spy(name):
        orig = getattr(mod, name)

        def wrapper(*args, **kwargs):
            calls.append(name)
            return orig(*args, **kwargs)

        return wrapper

    for name in (
        "_launch_longdim_rank",  # exact long-dimension k <= 64 (QR compress)
        "_launch_rrqr_rank",  # fast 65..255 unpivoted QR
        "_launch_bidiag_rank",  # exact general k > 64
        "_launch_tridiag_big_rank",  # exact hermitian k > 64
    ):
        monkeypatch.setattr(mod, name, _spy(name))

    generator = torch.Generator().manual_seed(2027)
    dev = flag_gems.device

    def run(shape, hermitian=False, tensor_tol=False):
        calls.clear()
        a = torch.randn(*shape, generator=generator, dtype=torch.float32)
        if hermitian:
            a = a + a.mT
        a = a.to(dev)
        kwargs = {}
        if tensor_tol:
            # atol must broadcast to the batch shape; a non-batched input
            # needs a 0-D tensor.
            kwargs["atol"] = torch.zeros(a.shape[:-2], device=dev)
        got = flag_gems.linalg_matrix_rank(a, hermitian=hermitian, **kwargs)
        # Clear spectra: both modes must agree with the reference; this
        # test pins the DISPATCH, not a fast-path inaccuracy.
        ref = torch.linalg.matrix_rank(a.cpu(), hermitian=hermitian)
        assert torch.equal(got.cpu(), ref)
        return list(calls)

    # --- default (exact) dispatch ---
    monkeypatch.delenv("FLAGGEMS_MR_FAST_PATH", raising=False)
    assert run((256, 64)) == ["_launch_longdim_rank"]
    assert run((64, 64)) == []  # square k <= 64: bidiag64 inline, unaffected
    assert run((65, 65)) == ["_launch_bidiag_rank"]
    assert run((255, 255)) == ["_launch_bidiag_rank"]
    assert run((256, 256)) == ["_launch_bidiag_rank"]
    assert run((65, 65), hermitian=True) == ["_launch_tridiag_big_rank"]
    assert run((2, 65, 65)) == ["_launch_bidiag_rank"]  # batch

    # --- fast opt-in ---
    monkeypatch.setenv("FLAGGEMS_MR_FAST_PATH", "1")
    assert run((256, 64)) == []  # long-dim k <= 64: Gram inline
    assert run((64, 64)) == []  # square k <= 64: still unaffected
    assert run((65, 65)) == ["_launch_rrqr_rank"]
    assert run((255, 255)) == ["_launch_rrqr_rank"]
    assert run((256, 256)) == ["_launch_bidiag_rank"]  # k >= 256 stays exact
    assert run((65, 65), hermitian=True) == ["_launch_rrqr_rank"]
    assert run((65, 65), tensor_tol=True) == ["_launch_rrqr_rank"]

    # --- runtime switching in one process (the env is read per call) ---
    monkeypatch.delenv("FLAGGEMS_MR_FAST_PATH", raising=False)
    assert run((65, 65)) == ["_launch_bidiag_rank"]
    monkeypatch.setenv("FLAGGEMS_MR_FAST_PATH", "1")
    assert run((65, 65)) == ["_launch_rrqr_rank"]
    monkeypatch.delenv("FLAGGEMS_MR_FAST_PATH", raising=False)
    assert run((65, 65)) == ["_launch_bidiag_rank"]


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
@pytest.mark.parametrize(
    "shape,fill_row,fill_col",
    [
        pytest.param((16, 5), 15, 4, id="tall-fused-band"),
        pytest.param((5, 16), 4, 15, id="wide-fused-band"),
        pytest.param((50, 40), 49, 39, id="tall-bidiag64-band"),
        pytest.param((40, 50), 39, 49, id="wide-bidiag64-band"),
    ],
)
def test_linalg_matrix_rank_nonsquare_tail_energy(dtype, shape, fill_row, fill_col):
    # GK bidiagonalization of a tall matrix needs K left reflections (the
    # last one folds the column K-1 tail into d[K-1]); a wide matrix is
    # handled by transposing to the tall form.  Putting the last column's
    # (resp. row's) energy beyond the diagonal band exposes a missing final
    # reflection: the rank comes out one short.  The fused (k <= 32) kernel
    # had exactly this latent bug -- random full-rank inputs mask it
    # because the lost energy stays below the tolerance.
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
        pytest.param((16, 5), 3, id="tall-fused-band"),
        pytest.param((5, 16), 3, id="wide-fused-band"),
        pytest.param((32, 8), 4, id="tall-fused-band-k8"),
        pytest.param((8, 32), 4, id="wide-fused-band-k8"),
        pytest.param((33, 64), 16, id="wide-bidiag64-band"),
        pytest.param((64, 33), 16, id="tall-bidiag64-band"),
        pytest.param((48, 60), 24, id="wide-bidiag64-band-2"),
    ],
)
def test_linalg_matrix_rank_nonsquare_lowrank(dtype, shape, rank):
    # Random non-square low-rank matrices (slowly-decaying spectrum, sigma
    # from 1 down to 1e-4) across the fused and bidiag64 bands.  Constructed
    # in fp64 and rounded once, so the fp64 reference with fp32-semantics
    # tolerance is exact.
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
    reference = torch.linalg.matrix_rank(
        matrix.cpu().to(torch.float64), atol=0.0, rtol=rtol
    )

    result = flag_gems.linalg_matrix_rank(matrix)
    _assert_output_metadata(result, matrix)
    _assert_equal(result, reference.to(device=matrix.device))


@pytest.mark.linalg_matrix_rank
# k <= 32 is only exercised on Ascend: there the fused kernel counts with a
# Sturm qd chain (whose tie convention this test targets), while the generic
# path uses one-sided Jacobi whose column-norm comparison is directly strict
# (and its fp32 sum-of-squares cannot represent the subnormal-tie case).
@pytest.mark.parametrize(
    "k", [3, 33, 65, 128, 257] if IS_ASCEND else [33, 65, 128, 257]
)
def test_linalg_matrix_rank_hermitian_strict_threshold(k, monkeypatch):
    # The exact herm paths are the default dispatch (the 65..255 fast
    # unpivoted-QR band is opt-in via FLAGGEMS_MR_FAST_PATH=1); clear any
    # FAST_PATH leftover so the strict-threshold cases really run the
    # fused/padded/tridiag Sturm counters exercised here.
    monkeypatch.delenv("FLAGGEMS_MR_FAST_PATH", raising=False)
    # torch's hermitian semantics are STRICT: rank = #{|lambda| > tol}
    # = #{lambda > tol} + #{lambda < -tol}.  The Sturm qd zero-pivot guard
    # counts #{lambda <= x}, so the positive side K - #{<= tol} is already
    # strict, while the negative side uses the mirrored tie convention
    # (zero pivot -> tiny POSITIVE) which counts #{lambda < -tol} exactly
    # (_sturm_count_posneg2 / _sturm_count_strict*); otherwise an
    # eigenvalue exactly equal to -tol is wrongly counted, and with
    # atol == rtol == 0 a nonzero rank-deficient spectrum reports full rank.
    # Diagonal inputs keep the factorization exact, so these ties are
    # deterministic.  k covers every herm path: fused (<=32), padded
    # tridiag (33..64), and the large one-sided tridiagonalization.
    device = flag_gems.device

    def diag_case(values, atol, rtol):
        matrix = torch.diag(values).to(torch.float32).to(device)
        reference = torch.linalg.matrix_rank(
            matrix.cpu().double(), hermitian=True, atol=atol, rtol=rtol
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
    # NOTE: the equivalent tie at the smallest SUBNORMAL is intentionally
    # not tested here.  The Sturm count runs in Triton kernels, and some
    # backends flush subnormals to zero there even when the platform's
    # native PyTorch elementwise kernels preserve them -- so a host-side
    # torch probe cannot certify the capability.  Re-enable only behind a
    # runtime-provided "Triton FP32 denormal" capability bit.
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
        reference = torch.linalg.matrix_rank(
            matrix.cpu().double(), hermitian=True, atol=atol, rtol=0.0
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
    reference = torch.linalg.matrix_rank(
        matrix.double(), hermitian=True, atol=atol.cpu(), rtol=rtol.cpu()
    )
    result = flag_gems.linalg_matrix_rank(
        matrix.to(device), hermitian=True, atol=atol, rtol=rtol
    )
    _assert_equal(result, reference.to(device))


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("k", [3, 33, 65, 257])
@pytest.mark.parametrize("hermitian", [False, True])
def test_linalg_matrix_rank_negative_tolerances(k, hermitian):
    # torch does not clamp the tolerance: tol = max(atol, rtol*sigma_max).
    # tol < 0 is reachable only when BOTH atol < 0 and rtol < 0, and then
    # every singular value (>= 0) exceeds tol -> rank == k for a nonzero
    # matrix; a zero matrix still gives 0 because rtol*0 == 0 lifts tol to
    # max(atol, 0) == 0.  A negative atol alone is harmless (rtol*sigma_max
    # >= 0 dominates the max).  The backend fixes the both-negative corner
    # up host-side; without it the hermitian split #{|lam|>tol} =
    # #{lam>tol} + #{lam<-tol} double-counts the overlap (rank > k) and the
    # sigma^2-domain paths square tol.
    device = flag_gems.device
    values = torch.zeros(k)
    values[:2] = torch.tensor([1.0, -0.5])
    matrix = torch.diag(values).to(torch.float32).to(device)
    zero = torch.zeros(k, k, dtype=torch.float32, device=device)

    def check(mat, atol, rtol):
        reference = torch.linalg.matrix_rank(
            mat.cpu().double(), hermitian=hermitian, atol=atol, rtol=rtol
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
        reference = torch.linalg.matrix_rank(
            upper_only.cpu().double(), hermitian=True, atol=-1.0, rtol=-1.0
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
        reference = torch.linalg.matrix_rank(
            lower_only.cpu().double(), hermitian=True, atol=-1.0, rtol=-1.0
        )
        assert reference.item() == k  # construction sanity
        result = flag_gems.linalg_matrix_rank(
            lower_only, hermitian=True, atol=-1.0, rtol=-1.0
        )
        _assert_equal(result, reference.to(device))

        # Same three regimes through the TENSOR-tolerance branch (the async
        # early-exit fixup kernel): strict-upper garbage -> 0, lower-only
        # nonzero -> k, true zero -> 0.
        mixed = torch.stack([upper_only, lower_only, zero])
        atol_t = torch.full((3,), -1.0, device=device)
        rtol_t = torch.full((3,), -1.0, device=device)
        reference = torch.linalg.matrix_rank(
            mixed.cpu().double(), hermitian=True, atol=atol_t.cpu(), rtol=rtol_t.cpu()
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
    reference = torch.linalg.matrix_rank(
        batch.cpu().double(), hermitian=hermitian, atol=atol_t.cpu(), rtol=rtol_t.cpu()
    )
    result = flag_gems.linalg_matrix_rank(
        batch, hermitian=hermitian, atol=atol_t, rtol=rtol_t
    )
    _assert_equal(result, reference.to(device))


@pytest.mark.linalg_matrix_rank
@pytest.mark.skipif(not IS_ASCEND, reason="Ascend-specific path coverage")
@pytest.mark.parametrize("shape", [(129, 64), (64, 129), (192, 64), (64, 192)])
def test_linalg_matrix_rank_longdim_exact_power2_nb(shape, monkeypatch):
    # Long-dimension k <= 64 QR-compresses to the k x k R factor with the
    # register panel kernel (default since the exact-path flip; clear any
    # FAST_PATH leftover so the register panel is really exercised); for
    # these shapes
    # rs = round_up(max(m, n), 64) = 192, and a raw NB = rs // 64 = 3
    # specialization is a marginal UB allocation that flip-flops between
    # fitting and "ub overflow" across compiles.  The launcher must clamp NB
    # to {1, 2, 4} (same as the main QR launcher).
    monkeypatch.delenv("FLAGGEMS_MR_FAST_PATH", raising=False)
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
    reference = torch.linalg.matrix_rank(
        matrix.cpu().to(torch.float64), atol=0.0, rtol=rtol
    )
    result = flag_gems.linalg_matrix_rank(matrix)
    _assert_output_metadata(result, matrix)
    _assert_equal(result, reference.to(device=matrix.device))


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("k,expect_rank", [(65, 1), (128, 1), (257, 1), (513, 1)])
def test_linalg_matrix_rank_hermitian_deflated_spectrum(k, expect_rank, monkeypatch):
    # Exact paths are the default; clear any FAST_PATH leftover so the
    # large one-sided tridiagonalization path is really exercised.
    monkeypatch.delenv("FLAGGEMS_MR_FAST_PATH", raising=False)
    # Strongly deflated spectra (a few significant eigenvalues, the rest at
    # the fp32 noise floor) drive the trailing subdiagonal to ~1e-10, where
    # tau = 2/vnorm2 ~ 1e20 and a naively grouped (tau*tau)*(v'w)/2
    # coefficient overflows fp32 in the rank-2 update -- verified to corrupt
    # the trailing diagonal to -inf and then NaN via 0*inf in the next
    # float mask.  The apply kernel regroups the coefficient as
    # tau*(tau*cs); this test guards that regression on the large
    # one-sided tridiagonalization path.
    generator = torch.Generator().manual_seed(k)
    basis = torch.linalg.qr(
        torch.randn(k, k, generator=generator, dtype=torch.float64)
    )[0]
    values = torch.zeros(k, dtype=torch.float64)
    values[:4] = torch.tensor([1.0, -0.5, 0.5, -0.25])
    matrix = ((basis * values) @ basis.mT).float().to(flag_gems.device)

    reference = torch.linalg.matrix_rank(
        matrix.cpu().double(), hermitian=True, atol=0.51, rtol=0.0
    )
    assert reference.item() == expect_rank  # 1.0 only; +/-0.5/-0.25 excluded
    result = flag_gems.linalg_matrix_rank(matrix, hermitian=True, atol=0.51, rtol=0.0)
    assert torch.isfinite(result.cpu().double()).all()
    _assert_equal(result, reference.to(flag_gems.device))


@pytest.mark.linalg_matrix_rank
@pytest.mark.skipif(not IS_ASCEND, reason="Ascend-specific path coverage")
@pytest.mark.parametrize(
    "shape,rank,hermitian,kind",
    [
        # general: RRQR (65..128) vs default bidiag (>128) boundary.
        # The RRQR case uses an exactly-gapped spectrum: unpivoted QR's
        # |R_ii| undercounts slow-decay spectra even at 6x tolerance margin
        # (verified: sigma=1e-4 vs tol=1.5e-5 reports 59/60) -- that is the
        # documented exception-2 limitation, not a dispatch bug.
        pytest.param((128, 128), 60, False, "gapped", id="general-k128-rrqr"),
        pytest.param((255, 255), 120, False, "gapped", id="general-k255-rrqr"),
        pytest.param((256, 256), 120, False, "slowdecay", id="general-k256-bidiag"),
        pytest.param((256, 512), 120, False, "slowdecay", id="general-k256-wide"),
        pytest.param((512, 256), 120, False, "slowdecay", id="general-k256-tall"),
        pytest.param((2, 256, 256), 120, False, "slowdecay", id="general-k256-batched"),
        # hermitian: QR (65..255) vs one-sided big tridiag (>=256)
        pytest.param((64, 64), 30, True, "slowdecay", id="herm-k64-padded"),
        pytest.param((65, 65), 30, True, "gapped", id="herm-k65-rrqr"),
        pytest.param((256, 256), 120, True, "slowdecay", id="herm-k256-tridiag"),
    ],
)
def test_linalg_matrix_rank_dispatch_boundary(shape, rank, hermitian, kind):
    # Default-dispatch boundaries.  slowdecay = singular values 1 .. 1e-4
    # (exposes the Gram sigma^2 floor and QR near-tolerance miscount where
    # those paths are NOT expected); gapped = sigma in {1, 0} with an exact
    # gap (valid on every dispatch).  Reference: fp64 with fp32-semantics
    # tolerance.
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
    reference = torch.linalg.matrix_rank(
        matrix.cpu().to(torch.float64), atol=0.0, rtol=rtol, hermitian=hermitian
    )
    result = flag_gems.linalg_matrix_rank(matrix, hermitian=hermitian)
    _assert_output_metadata(result, matrix)
    _assert_equal(result, reference.to(flag_gems.device))


@pytest.mark.linalg_matrix_rank
@pytest.mark.skipif(not IS_ASCEND, reason="Ascend-specific path coverage")
@pytest.mark.parametrize(
    "k,hermitian", [(256, False), (513, False), (256, True), (513, True)]
)
def test_linalg_matrix_rank_graph_vs_nograph(k, hermitian, monkeypatch):
    # The NPUGraph-replayed launch sequence must produce bit-identical
    # results to direct launches (FLAGGEMS_MR_NO_GRAPH=1).
    generator = torch.Generator().manual_seed(k)
    if hermitian:
        basis = torch.linalg.qr(
            torch.randn(k, k, generator=generator, dtype=torch.float64)
        )[0]
        values = torch.cat(
            [
                torch.logspace(0, -4, k // 3, dtype=torch.float64),
                torch.zeros(k - k // 3, dtype=torch.float64),
            ]
        )
        matrix = ((basis * values) @ basis.mT).float().to(flag_gems.device)
    else:
        matrix = torch.randn(k, k, generator=generator).float().to(flag_gems.device)

    monkeypatch.setenv("FLAGGEMS_MR_NO_GRAPH", "1")
    direct = flag_gems.linalg_matrix_rank(matrix, hermitian=hermitian)
    monkeypatch.delenv("FLAGGEMS_MR_NO_GRAPH")
    graphed = flag_gems.linalg_matrix_rank(matrix, hermitian=hermitian)  # captures
    replayed = flag_gems.linalg_matrix_rank(matrix, hermitian=hermitian)  # replays
    _assert_equal(direct, graphed)
    _assert_equal(direct, replayed)


@pytest.mark.linalg_matrix_rank
@pytest.mark.skipif(IS_ASCEND, reason="Ascend backend has its own graph test above")
@pytest.mark.parametrize(
    "shape,hermitian",
    [
        pytest.param((65, 65), True, id="herm-k65"),
        pytest.param((257, 257), True, id="herm-k257"),
        pytest.param((513, 513), False, id="bidiag-k513"),
    ],
)
def test_linalg_matrix_rank_generic_graph_vs_nograph(shape, hermitian, monkeypatch):
    # The CUDA-graph-replayed kernel sequence must produce the same rank as
    # direct launches (FLAGGEMS_MR_NO_GRAPH=1), and replays must refresh the
    # staging buffers: fresh input data and changed tolerances both have to
    # be honored by a replayed graph.
    generator = torch.Generator().manual_seed(sum(shape))
    matrix = torch.randn(*shape, generator=generator).float()
    if hermitian:
        matrix = matrix + matrix.mT
    matrix = matrix.to(flag_gems.device)
    reference = torch.linalg.matrix_rank(matrix.cpu(), hermitian=hermitian)

    monkeypatch.setenv("FLAGGEMS_MR_NO_GRAPH", "1")
    direct = flag_gems.linalg_matrix_rank(matrix, hermitian=hermitian)
    _assert_equal(direct, reference.to(flag_gems.device))

    monkeypatch.delenv("FLAGGEMS_MR_NO_GRAPH")
    captured = flag_gems.linalg_matrix_rank(matrix, hermitian=hermitian)  # captures
    replayed = flag_gems.linalg_matrix_rank(matrix, hermitian=hermitian)  # replays
    _assert_equal(captured, reference.to(flag_gems.device))
    _assert_equal(replayed, reference.to(flag_gems.device))

    # Replay with fresh data: the graph must re-read the staging buffers.
    fresh = torch.randn(*shape, generator=generator).float()
    if hermitian:
        fresh = fresh + fresh.mT
    fresh = fresh.to(flag_gems.device)
    fresh_ref = torch.linalg.matrix_rank(fresh.cpu(), hermitian=hermitian)
    _assert_equal(
        flag_gems.linalg_matrix_rank(fresh, hermitian=hermitian),
        fresh_ref.to(flag_gems.device),
    )
    # Replay with a changed tolerance: tolerances are staging inputs too.
    tol_ref = torch.linalg.matrix_rank(fresh.cpu(), hermitian=hermitian, atol=0.5)
    _assert_equal(
        flag_gems.linalg_matrix_rank(fresh, hermitian=hermitian, atol=0.5),
        tol_ref.to(flag_gems.device),
    )


@pytest.mark.linalg_matrix_rank
@pytest.mark.skipif(not IS_ASCEND, reason="Ascend-specific path coverage")
@pytest.mark.parametrize("k", [300, 513])
def test_linalg_matrix_rank_hermitian_ignores_strict_upper_large(k):
    # torch hermitian semantics read only the lower triangle; the large
    # one-sided tridiagonalization (init kernel's max/min addressing) must
    # not let garbage in the strict upper triangle into the spectrum.
    generator = torch.Generator().manual_seed(k)
    lower = torch.tril(torch.randn(k, k, generator=generator))
    matrix = lower.clone()
    matrix.masked_fill_(torch.triu(torch.ones(k, k, dtype=torch.bool), 1), 1e6)
    matrix = matrix.to(flag_gems.device)
    reference = torch.linalg.matrix_rank(
        (torch.tril(matrix.cpu().double()) + torch.tril(matrix.cpu().double(), -1).mT),
        hermitian=True,
    )
    result = flag_gems.linalg_matrix_rank(matrix, hermitian=True)
    _assert_equal(result, reference.to(flag_gems.device))


@pytest.mark.linalg_matrix_rank
@pytest.mark.skipif(IS_ASCEND, reason="Ascend backend has its own implementation")
@pytest.mark.parametrize(
    "shape,hermitian",
    [
        pytest.param((33, 33), True, id="herm-small-tridiag"),
        pytest.param((65, 65), True, id="herm-padded-tridiag"),
        pytest.param((257, 257), True, id="herm-large-tridiag"),
        pytest.param((4, 65, 65), True, id="herm-batched"),
        pytest.param((65, 80), False, id="bidiag-medium"),
        pytest.param((513, 513), False, id="bidiag-k513"),
        pytest.param((600, 513), False, id="bidiag-tall"),
    ],
)
def test_linalg_matrix_rank_ds32_fallback(shape, hermitian, monkeypatch):
    # Force the pure-FP32 double-single Sturm tail (the path selected when
    # runtime device support_fp64 is False) on a device that natively has
    # fp64, and check every rank-relevant case against torch.  The fallback
    # only changes the Sturm count, so diagonal spectra stay exact and the
    # dense cases are built with margins far above the fp32 noise floor.
    module = importlib.import_module("flag_gems.ops.linalg_matrix_rank")
    monkeypatch.setattr(module.runtime_device, "support_fp64", False)

    device = flag_gems.device
    generator = torch.Generator().manual_seed(0)
    k = min(shape[-2:])
    rank = 7

    if hermitian:
        dense = torch.randn(shape, generator=generator)
        dense = dense + dense.mT
        x = torch.randn(shape[:-2] + (k, rank), generator=generator)
        low_rank = x @ x.mT
        values = torch.zeros(k)
        values[:6] = torch.tensor([1.0, -0.5, 1e-3, -1e-3, 1e-8, -1e-8])
        spectrum = torch.diag(values).expand(shape).contiguous()
    else:
        dense = torch.randn(shape, generator=generator)
        x = torch.randn(shape[:-2] + (shape[-2], rank), generator=generator)
        y = torch.randn(shape[:-2] + (shape[-1], rank), generator=generator)
        low_rank = x @ y.mT
        values = torch.zeros(k)
        values[:4] = torch.tensor([1.0, 2.0, 1e-3, 1e-8])
        spectrum = torch.zeros(shape)
        spectrum[..., torch.arange(k), torch.arange(k)] = values
    zero = torch.zeros(shape)

    def check(matrix, **kwargs):
        matrix = matrix.float().to(device)
        reference = torch.linalg.matrix_rank(matrix.cpu(), **kwargs)
        result = module.linalg_matrix_rank(matrix, **kwargs)
        _assert_equal(result, reference.to(device))

    # full-rank dense
    check(dense, hermitian=hermitian)
    # exactly rank-`rank`, kept spectrum well above the default tolerance
    check(low_rank, hermitian=hermitian)
    # zero matrix -> rank 0 (the bracket must hand a defined zero tolerance
    # to the decisive count)
    check(zero, hermitian=hermitian)
    # near-threshold spectrum: 1e-3 above the default tolerance
    # (k*eps*sigma_max ~ 1e-5), 1e-8 below it
    check(spectrum, hermitian=hermitian)
    # explicit atol: only the O(1) part of the spectrum survives
    check(spectrum, hermitian=hermitian, atol=1e-2)


@pytest.mark.linalg_matrix_rank
@pytest.mark.skipif(IS_ASCEND, reason="Ascend backend has its own implementation")
def test_linalg_matrix_rank_fp64_input_requires_native_fp64(monkeypatch):
    # On a device without native FP64 the entry point must reject float64
    # input with NotImplementedError before any shape dispatch, instead of
    # silently computing in demoted precision.
    module = importlib.import_module("flag_gems.ops.linalg_matrix_rank")
    monkeypatch.setattr(module.runtime_device, "support_fp64", False)

    matrix = torch.randn(8, 8, dtype=torch.float64, device=flag_gems.device)
    with pytest.raises(NotImplementedError, match="native FP64"):
        module.linalg_matrix_rank(matrix)
    with pytest.raises(NotImplementedError, match="native FP64"):
        module.linalg_matrix_rank(matrix, hermitian=True)


@pytest.mark.linalg_matrix_rank
@pytest.mark.skipif(IS_ASCEND, reason="Ascend backend has its own implementation")
@pytest.mark.parametrize("log10_scale", [20, -30], ids=lambda s: f"1e{s}")
@pytest.mark.parametrize(
    "shape,hermitian",
    [
        pytest.param((16, 16), False, id="fused"),
        pytest.param((65, 65), True, id="herm-tridiag"),
        pytest.param((513, 513), False, id="bidiag"),
    ],
)
def test_linalg_matrix_rank_extreme_scales(shape, hermitian, log10_scale):
    # 1e20 overflows FP32 sum-of-squares (x^2 ~ 1e40 > fp32 max) and 1e-30
    # underflows it (x^2 ~ 1e-60 -> 0).  The Householder algebra squares the
    # matrix scale (w = A.v is O(sigma^2)), so norm-internal scaling alone
    # cannot fix this: the launcher normalizes each matrix to O(1) up front
    # and shrinks atol by the same factor, which leaves the rank semantics
    # invariant.  Reference: exact CPU fp64 SVD with fp32-default rtol.
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
    reference = torch.linalg.matrix_rank(
        matrix.cpu().double(), hermitian=hermitian, rtol=rtol
    )
    result = flag_gems.linalg_matrix_rank(matrix, hermitian=hermitian)
    _assert_equal(result, reference.to(flag_gems.device))


@pytest.mark.linalg_matrix_rank
@pytest.mark.skipif(IS_ASCEND, reason="Ascend backend has its own implementation")
def test_linalg_matrix_rank_mixed_magnitude_batch():
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
def test_linalg_matrix_rank_tolerance_precision():
    # A float64/Python-float tolerance next to a singular value of a
    # float32 matrix must decide the rank at ITS OWN precision, not after
    # rounding to float32: atol = 0.5 - 1e-16 rounds back to 0.5 in fp32
    # and would wrongly exclude the 0.5 eigenvalue (strict threshold).
    matrix = torch.diag(torch.tensor([1.0, 0.5, 0.0, 0.0])).float()
    matrix = matrix.to(flag_gems.device)
    for atol in (0.5, 0.5 - 1e-16, 0.5 + 1e-16):
        reference = torch.linalg.matrix_rank(
            matrix.cpu().double(), hermitian=True, atol=atol
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
        assert result.item() == expected_rank


@pytest.mark.linalg_matrix_rank
@pytest.mark.skipif(not SUPPORT_FP64, reason="float64 not supported on this device")
@pytest.mark.skipif(IS_ASCEND, reason="Ascend backend has its own implementation")
@pytest.mark.parametrize("k", [33, 65, 257])
def test_linalg_matrix_rank_fp64_critical_spectrum(k):
    # An eigenvalue within ~1e-12 (relative) of the rtol threshold: the
    # sigma_max Gershgorin bracket must be refined to fp64 mantissa depth
    # (64 bisection iterations; 32 would leave ~1e-10 of bracket noise and
    # could flip the rank).  Rotated spectra keep the bracket loose so the
    # refinement really runs; the CPU fp64 oracle arbitrates.
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
        reference = torch.linalg.matrix_rank(
            matrix.cpu(), hermitian=True, atol=0.0, rtol=rtol
        )
        assert reference.item() == expected_rank  # construction sanity
        result = flag_gems.linalg_matrix_rank(
            matrix, hermitian=True, atol=0.0, rtol=rtol
        )
        _assert_equal(result, reference.to(flag_gems.device))


@pytest.mark.linalg_matrix_rank
@pytest.mark.skipif(IS_ASCEND, reason="Ascend backend has its own implementation")
@pytest.mark.skipif(
    not SUPPORT_FP64,
    reason="native-FP64-first capture ordering needs native FP64",
)
def test_linalg_matrix_rank_graph_key_includes_ds32(monkeypatch):
    # The graph cache key must include the native-FP64/DS32 mode: switching
    # support_fp64 off (as the ds32 fallback tests do) must NOT replay a
    # graph captured with the native-FP64 Sturm tail, otherwise the DS32
    # path would never actually execute on a graph-capable device.
    if (
        torch.device(flag_gems.device).type != "cuda"
        or torch.version.cuda is None
        or torch.version.hip is not None
    ):
        pytest.skip("graph capture only on genuine CUDA builds")
    module = importlib.import_module("flag_gems.ops.linalg_matrix_rank")
    module._MR_GRAPHS.clear()
    module._MR_GRAPH_BYTES = 0

    matrix = torch.randn(65, 65).float()
    matrix = (matrix + matrix.mT).to(flag_gems.device)
    reference = torch.linalg.matrix_rank(matrix.cpu(), hermitian=True)

    module.linalg_matrix_rank(matrix, hermitian=True)  # native-fp64 capture
    assert len(module._MR_GRAPHS) == 1
    assert list(module._MR_GRAPHS)[0][0][4] is False

    monkeypatch.setattr(module.runtime_device, "support_fp64", False)
    result = module.linalg_matrix_rank(matrix, hermitian=True)  # ds32 capture
    _assert_equal(result, reference.to(flag_gems.device))
    assert len(module._MR_GRAPHS) == 2
    assert list(module._MR_GRAPHS)[1][0][4] is True


def _blocked_rotated_spectrum(k, values, seed):
    # Hermitian matrix with the given eigenvalues, rotated by a random
    # orthogonal basis so the Householder panel algebra is really exercised.
    # Built on CPU in fp64 and rounded to fp32 ONCE (device-side fp64 is not
    # portable); the fp64 CPU oracle on the rounded matrix arbitrates.
    generator = torch.Generator().manual_seed(seed)
    basis = torch.linalg.qr(
        torch.randn(k, k, generator=generator, dtype=torch.float64)
    )[0]
    spectrum = torch.zeros(k, dtype=torch.float64)
    spectrum[: len(values)] = torch.tensor(values, dtype=torch.float64)
    return ((basis * spectrum) @ basis.mT).float()


@pytest.mark.linalg_matrix_rank
@pytest.mark.skipif(IS_ASCEND, reason="Ascend backend has its own implementation")
@pytest.mark.parametrize(
    "k,path",
    [
        pytest.param(767, "unblocked", id="k767-below-boundary"),
        pytest.param(768, "blocked", id="k768-boundary"),
        pytest.param(769, "blocked", id="k769-unaligned"),
        pytest.param(1000, "blocked", id="k1000-unaligned"),
    ],
)
def test_linalg_matrix_rank_hermitian_blocked_dispatch(k, path, monkeypatch):
    # k == _HERM_TRIDIAG_BLOCKED_MIN_K (768) is the blocked-WY dispatch
    # boundary: 767 must stay on the per-column unblocked run, 768 crosses
    # over, and 769/1000 exercise the non-64-aligned padded tiles and the
    # partial last panel.  Spy on both run functions to prove which path
    # actually executes (a correct result alone would not catch a silent
    # fallback).  The graph cache is cleared so every case really runs its
    # launch sequence instead of replaying a graph captured by an earlier
    # test with the same cache key.
    module = importlib.import_module("flag_gems.ops.linalg_matrix_rank")
    module._MR_GRAPHS.clear()
    module._MR_GRAPH_BYTES = 0
    device = torch.empty((), device=flag_gems.device).device
    dot_ok = module._blocked_tridiag_ok(device)
    monkeypatch.setattr(module, "_blocked_tridiag_ok", lambda device: dot_ok)

    calls = {"blocked": [], "unblocked": []}
    orig_blocked = module._herm_tridiag_blocked_run
    orig_unblocked = module._herm_tridiag_run

    def spy_blocked(ws, k_, batch_count, ds32):
        calls["blocked"].append(k_)
        return orig_blocked(ws, k_, batch_count, ds32)

    def spy_unblocked(ws, k_, batch_count, ds32):
        calls["unblocked"].append(k_)
        return orig_unblocked(ws, k_, batch_count, ds32)

    monkeypatch.setattr(module, "_herm_tridiag_blocked_run", spy_blocked)
    monkeypatch.setattr(module, "_herm_tridiag_run", spy_unblocked)

    matrix = _blocked_rotated_spectrum(k, list(range(1, 101)), seed=k).to(
        flag_gems.device
    )
    # fp32 rounding lifts the zero eigenspace to ~1e-5, so the fp64 oracle
    # must use the FP32-default rtol (k * eps_fp32), not its own eps_fp64.
    rtol = k * torch.finfo(torch.float32).eps
    reference = torch.linalg.matrix_rank(
        matrix.cpu().double(), hermitian=True, rtol=rtol
    )
    result = module.linalg_matrix_rank(matrix, hermitian=True)
    _assert_equal(result, reference.to(flag_gems.device))
    assert result.item() == 100
    # Backends whose blocked pipeline fails the known-answer self-test
    # legitimately fall back to the unblocked run even at k >= 768 -- the
    # boundary assertion is against the self-test verdict, not size alone.
    expect_blocked = path == "blocked" and dot_ok
    assert bool(calls["blocked"]) == expect_blocked
    assert bool(calls["unblocked"]) != expect_blocked


@pytest.mark.linalg_matrix_rank
@pytest.mark.skipif(IS_ASCEND, reason="Ascend backend has its own implementation")
def test_linalg_matrix_rank_hermitian_blocked_self_test_fallback(monkeypatch):
    # When the blocked-path self-test fails (backend miscompile), k >= 768
    # fp32 hermitian inputs must fall back to the unblocked run and stay
    # correct -- blocked results would be silently wrong there.
    module = importlib.import_module("flag_gems.ops.linalg_matrix_rank")
    module._MR_GRAPHS.clear()
    module._MR_GRAPH_BYTES = 0
    calls = {"blocked": [], "unblocked": []}
    orig_blocked = module._herm_tridiag_blocked_run
    orig_unblocked = module._herm_tridiag_run

    def spy_blocked(ws, k_, batch_count, ds32):
        calls["blocked"].append(k_)
        return orig_blocked(ws, k_, batch_count, ds32)

    def spy_unblocked(ws, k_, batch_count, ds32):
        calls["unblocked"].append(k_)
        return orig_unblocked(ws, k_, batch_count, ds32)

    monkeypatch.setattr(module, "_herm_tridiag_blocked_run", spy_blocked)
    monkeypatch.setattr(module, "_herm_tridiag_run", spy_unblocked)
    monkeypatch.setattr(module, "_blocked_tridiag_ok", lambda device: False)

    k = 768
    matrix = _blocked_rotated_spectrum(k, list(range(1, 101)), seed=k).to(
        flag_gems.device
    )
    rtol = k * torch.finfo(torch.float32).eps
    reference = torch.linalg.matrix_rank(
        matrix.cpu().double(), hermitian=True, rtol=rtol
    )
    result = module.linalg_matrix_rank(matrix, hermitian=True)
    _assert_equal(result, reference.to(flag_gems.device))
    assert result.item() == 100
    assert not calls["blocked"]
    assert calls["unblocked"]


@pytest.mark.linalg_matrix_rank
@pytest.mark.skipif(IS_ASCEND, reason="Ascend backend has its own implementation")
def test_linalg_matrix_rank_hermitian_blocked_ignores_strict_upper():
    # The blocked tridiagonalization must read ONLY the lower triangle:
    # garbage in the strict upper triangle (which torch never reads for
    # hermitian=True) must not change the result.  Both matrices are built
    # entirely on CPU -- device-side advanced indexing is not portable --
    # and the CPU fp64 oracle on the garbage matrix arbitrates.
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
    reference = torch.linalg.matrix_rank(
        garbage_cpu.double(), hermitian=True, atol=5e-2
    )
    clean_rank = flag_gems.linalg_matrix_rank(clean, hermitian=True, atol=5e-2)
    garbage_rank = flag_gems.linalg_matrix_rank(garbage, hermitian=True, atol=5e-2)
    _assert_equal(garbage_rank, clean_rank)
    _assert_equal(garbage_rank, reference.to(flag_gems.device))


@pytest.mark.linalg_matrix_rank
@pytest.mark.skipif(IS_ASCEND, reason="Ascend backend has its own implementation")
@pytest.mark.parametrize("rank", [1, 7])
def test_linalg_matrix_rank_hermitian_blocked_deflated(rank):
    # Strong deflation: k - rank columns of the panel factorization see an
    # (almost) zero trailing block, driving tau huge -- the regrouped
    # tau*(tau*dot) coefficient must stay finite and the rank exact.
    k = 1024
    generator = torch.Generator().manual_seed(rank)
    factor = torch.randn(k, rank, dtype=torch.float64, generator=generator)
    matrix = ((factor @ factor.mT) / k).float().to(flag_gems.device)
    reference = torch.linalg.matrix_rank(
        matrix.cpu().double(), hermitian=True, atol=5e-2
    )
    assert reference.item() == rank  # construction sanity
    result = flag_gems.linalg_matrix_rank(matrix, hermitian=True, atol=5e-2)
    _assert_equal(result, reference.to(flag_gems.device))


@pytest.mark.linalg_matrix_rank
@pytest.mark.skipif(IS_ASCEND, reason="Ascend backend has its own implementation")
@pytest.mark.parametrize("log10_scale", [20, -30], ids=lambda s: f"1e{s}")
def test_linalg_matrix_rank_hermitian_blocked_extreme_scales(log10_scale):
    # Blocked path at 1e20 / 1e-30: per-batch normalization plus the
    # in-kernel init scaling must keep the panel algebra (which squares the
    # matrix scale) finite, with the rank semantics invariant.
    k = 768
    scale = 10.0**log10_scale
    generator = torch.Generator().manual_seed(k + log10_scale)
    matrix = torch.randn(k, k, generator=generator, dtype=torch.float64)
    matrix = ((matrix + matrix.mT) * scale).float().to(flag_gems.device)
    rtol = k * torch.finfo(torch.float32).eps
    reference = torch.linalg.matrix_rank(
        matrix.cpu().double(), hermitian=True, rtol=rtol
    )
    result = flag_gems.linalg_matrix_rank(matrix, hermitian=True)
    _assert_equal(result, reference.to(flag_gems.device))


@pytest.mark.linalg_matrix_rank
@pytest.mark.skipif(IS_ASCEND, reason="Ascend backend has its own implementation")
def test_linalg_matrix_rank_hermitian_blocked_critical_spectrum():
    # An eigenvalue at relative 1e-4 from the threshold -- well above the
    # fp32/DS32 noise floor but decisive for the Sturm count -- on a rotated
    # (dense) blocked input; the CPU fp64 oracle arbitrates.
    k = 1024
    for delta, expected_rank in ((1e-4, 2), (-1e-4, 1)):
        matrix = _blocked_rotated_spectrum(
            k, [1.0, 0.5 * (1.0 + delta)], seed=int(delta * 1e8)
        ).to(flag_gems.device)
        reference = torch.linalg.matrix_rank(
            matrix.cpu().double(), hermitian=True, atol=0.5, rtol=0.0
        )
        assert reference.item() == expected_rank  # construction sanity
        result = flag_gems.linalg_matrix_rank(
            matrix, hermitian=True, atol=0.5, rtol=0.0
        )
        _assert_equal(result, reference.to(flag_gems.device))


@pytest.mark.linalg_matrix_rank
@pytest.mark.skipif(IS_ASCEND, reason="Ascend backend has its own implementation")
def test_linalg_matrix_rank_hermitian_blocked_ds32(monkeypatch):
    # Force the pure-FP32 double-single Sturm tail on the BLOCKED path: the
    # panel factorization and the DS32 count must compose (the graph cache
    # key includes ds32, so this cannot replay a native-FP64 graph).
    module = importlib.import_module("flag_gems.ops.linalg_matrix_rank")
    monkeypatch.setattr(module.runtime_device, "support_fp64", False)

    k = 768
    matrix = _blocked_rotated_spectrum(k, list(range(1, 101)), seed=0).to(
        flag_gems.device
    )
    # Same fp32-default rtol as the dispatch test above.
    rtol = k * torch.finfo(torch.float32).eps
    reference = torch.linalg.matrix_rank(
        matrix.cpu().double(), hermitian=True, rtol=rtol
    )
    assert reference.item() == 100  # construction sanity
    result = module.linalg_matrix_rank(matrix, hermitian=True)
    _assert_equal(result, reference.to(flag_gems.device))


@pytest.mark.linalg_matrix_rank
@pytest.mark.skipif(IS_ASCEND, reason="Ascend backend has its own implementation")
def test_linalg_matrix_rank_hermitian_blocked_batched():
    # Batched blocked path: per-batch workspaces, scales and scratch slots
    # must stay independent (batch 0 rank 50, batch 1 full rank).
    k = 768
    low_rank = _blocked_rotated_spectrum(k, list(range(1, 51)), seed=1)
    generator = torch.Generator().manual_seed(2)
    dense = torch.randn(k, k, generator=generator, dtype=torch.float64)
    dense = (dense + dense.mT).float()
    batch = torch.stack([low_rank, dense]).to(flag_gems.device)
    reference = torch.linalg.matrix_rank(
        batch.cpu().double(), hermitian=True, atol=5e-2
    )
    assert reference.tolist() == [50, k]  # construction sanity
    result = flag_gems.linalg_matrix_rank(batch, hermitian=True, atol=5e-2)
    _assert_equal(result, reference.to(flag_gems.device))


@pytest.mark.linalg_matrix_rank
@pytest.mark.skipif(IS_ASCEND, reason="Ascend backend has its own implementation")
def test_linalg_matrix_rank_hermitian_blocked_repeatable():
    # The blocked kernels accumulate through atomics, so reduction ORDER is
    # nondeterministic; the rank on a near-threshold input must still not
    # flap across calls (first call captures the graph, the rest replay).
    # The eigenvalue sits at relative 1e-4 from the threshold -- orders of
    # magnitude above any ulp-level reordering noise.
    k = 1024
    matrix = _blocked_rotated_spectrum(k, [1.0, 0.5 * 1.0001], seed=3).to(
        flag_gems.device
    )
    reference = torch.linalg.matrix_rank(
        matrix.cpu().double(), hermitian=True, atol=0.5, rtol=0.0
    )
    assert reference.item() == 2  # construction sanity
    for _ in range(20):
        result = flag_gems.linalg_matrix_rank(
            matrix, hermitian=True, atol=0.5, rtol=0.0
        )
        _assert_equal(result, reference.to(flag_gems.device))


@pytest.mark.linalg_matrix_rank
@pytest.mark.skipif(IS_ASCEND, reason="Ascend backend has its own implementation")
def test_linalg_matrix_rank_hip_graph_gate(monkeypatch):
    # Graph capture is ON by default on every GPU build -- including
    # HIP-style builds (torch.version.hip set, torch.version.cuda empty) --
    # and FLAGGEMS_MR_NO_GRAPH=1 is the global kill switch.  Simulated on a
    # genuine CUDA build by monkeypatching BOTH version strings (hip alone
    # would not exercise the cuda-is-None branch that real ROCm builds
    # hit), pinning both directions of the gate without HIP hardware.
    if (
        torch.device(flag_gems.device).type != "cuda"
        or torch.version.cuda is None
        or torch.version.hip is not None
    ):
        pytest.skip("genuine CUDA build required to simulate the HIP gate")
    module = importlib.import_module("flag_gems.ops.linalg_matrix_rank")

    matrix = torch.randn(65, 65).float()
    matrix = (matrix + matrix.mT).to(flag_gems.device)
    reference = torch.linalg.matrix_rank(matrix.cpu(), hermitian=True)

    # Simulate an ROCm-stack build: hip set AND cuda empty.
    monkeypatch.setattr(torch.version, "hip", "6.1.0")
    monkeypatch.setattr(torch.version, "cuda", None)
    monkeypatch.delenv("FLAGGEMS_MR_NO_GRAPH", raising=False)

    # Default on HIP builds: capture happens, replay stays correct.
    module._MR_GRAPHS.clear()
    module._MR_GRAPH_BYTES = 0
    result = module.linalg_matrix_rank(matrix, hermitian=True)
    _assert_equal(result, reference.to(flag_gems.device))
    assert len(module._MR_GRAPHS) == 1
    result = module.linalg_matrix_rank(matrix, hermitian=True)  # replay
    _assert_equal(result, reference.to(flag_gems.device))
    assert len(module._MR_GRAPHS) == 1

    # Kill switch: direct launches, nothing captured.
    module._MR_GRAPHS.clear()
    module._MR_GRAPH_BYTES = 0
    monkeypatch.setenv("FLAGGEMS_MR_NO_GRAPH", "1")
    result = module.linalg_matrix_rank(matrix, hermitian=True)
    _assert_equal(result, reference.to(flag_gems.device))
    assert len(module._MR_GRAPHS) == 0


@pytest.mark.linalg_matrix_rank
@pytest.mark.skipif(IS_ASCEND, reason="Ascend backend has its own implementation")
def test_linalg_matrix_rank_empty_validates_tolerances():
    # Native torch runs its same-device / non-complex tolerance checks
    # BEFORE its empty-input return; FlagGems must match, so an empty
    # matrix still rejects invalid tensor tolerances.
    matrix = torch.empty(2, 0, 5, device=flag_gems.device)
    result = flag_gems.linalg_matrix_rank(matrix)
    reference = torch.linalg.matrix_rank(matrix.cpu())
    _assert_equal(result, reference.to(flag_gems.device))

    complex_tol = torch.ones(2, dtype=torch.complex64, device=matrix.device)
    with pytest.raises(RuntimeError, match="complex"):
        flag_gems.linalg_matrix_rank(matrix, atol=complex_tol)
    if matrix.device.type != "cpu":
        cpu_tol = torch.ones(2)
        with pytest.raises(RuntimeError, match="same device"):
            flag_gems.linalg_matrix_rank(matrix, rtol=cpu_tol)


@pytest.mark.linalg_matrix_rank
@pytest.mark.skipif(IS_ASCEND, reason="Ascend backend has its own implementation")
@pytest.mark.parametrize("hermitian", [False, True], ids=["bidiag", "herm"])
@pytest.mark.parametrize(
    "tol_kind", ["scalar", "broadcast"], ids=["scalar-tol", "broadcast-tol"]
)
def test_linalg_matrix_rank_multidim_batch_large_path(hermitian, tol_kind, monkeypatch):
    # A (2, 3, 65, 65) input flattens to batch_count 6 on the large
    # decomposition paths; the staged atol/rtol metadata must be flattened
    # to (6,) as well, or the copy into the (6,) workspace buffer fails
    # (Tensor.copy_ does not reshape equal-numel tensors).  The small
    # single-kernel paths never hit this -- they index raw pointers.  Both
    # direct (no-graph) and graph capture+replay executions are checked.
    module = importlib.import_module("flag_gems.ops.linalg_matrix_rank")
    k = 65
    generator = torch.Generator().manual_seed(13)
    # Batch member 0 is rank 10 by construction (FF.mT/k: smallest nonzero
    # eigenvalue near the Wishart lower edge ~0.4, far above atol); the
    # rest are diagonal-dominant dense (singular values far above atol),
    # so the fp64 CPU oracle and the fp32 result agree without ambiguity.
    factor = torch.randn(k, 10, dtype=torch.float64, generator=generator)
    base = torch.empty(6, k, k, dtype=torch.float64)
    base[0] = (factor @ factor.mT) / k
    dense = torch.randn(5, k, k, dtype=torch.float64, generator=generator)
    if hermitian:
        dense = dense + dense.mT
        shift = 6.0 * k**0.5
    else:
        shift = 3.0 * k**0.5
    dense += torch.eye(k, dtype=torch.float64) * shift
    base[1:] = dense
    matrix = base.float().reshape(2, 3, k, k).to(flag_gems.device)

    atol = 5e-2
    kwargs = {"hermitian": hermitian, "atol": atol}
    if tol_kind == "broadcast":
        # Broadcastable tensor tolerance: (3,) -> batch shape (2, 3).
        kwargs["atol"] = torch.full((3,), atol, device=flag_gems.device)
    reference = torch.linalg.matrix_rank(base, hermitian=hermitian, atol=atol)
    reference = reference.reshape(2, 3).to(flag_gems.device)
    assert reference.flatten()[0].item() == 10  # construction sanity

    expected_shape = (2, 3)

    # Direct launches (kill switch on): nothing captured.
    module._MR_GRAPHS.clear()
    module._MR_GRAPH_BYTES = 0
    monkeypatch.setenv("FLAGGEMS_MR_NO_GRAPH", "1")
    result = flag_gems.linalg_matrix_rank(matrix, **kwargs)
    assert result.shape == expected_shape
    _assert_equal(result, reference)

    # Graph capture + replay (skipped silently on no-graph builds).
    monkeypatch.delenv("FLAGGEMS_MR_NO_GRAPH", raising=False)
    module._MR_GRAPHS.clear()
    module._MR_GRAPH_BYTES = 0
    first = flag_gems.linalg_matrix_rank(matrix, **kwargs)
    replay = flag_gems.linalg_matrix_rank(matrix, **kwargs)
    assert first.shape == expected_shape
    _assert_equal(first, reference)
    _assert_equal(replay, reference)


@pytest.mark.linalg_matrix_rank
@pytest.mark.skipif(IS_ASCEND, reason="Ascend backend has its own implementation")
def test_linalg_matrix_rank_blocked_probe_runs_once(monkeypatch):
    # Cold-call coverage for the blocked-path dispatch: the first eligible
    # call runs the one-time known-answer probe exactly once (the verdict
    # cache is double-checked under a lock), and the dispatched result is
    # correct whether the verdict enables blocked (healthy backend) or
    # falls back to unblocked.
    module = importlib.import_module("flag_gems.ops.linalg_matrix_rank")
    module._BLOCKED_TRIDIAG_OK.clear()
    module._MR_GRAPHS.clear()
    module._MR_GRAPH_BYTES = 0
    calls = []
    orig_probe = module._blocked_tridiag_probe

    def spy_probe(device):
        calls.append(device)
        return orig_probe(device)

    monkeypatch.setattr(module, "_blocked_tridiag_probe", spy_probe)

    k = 768
    matrix = _blocked_rotated_spectrum(k, list(range(1, 101)), seed=5).to(
        flag_gems.device
    )
    # fp32 rounding lifts the zero eigenspace to ~1e-5, so the fp64 oracle
    # uses the FP32-default rtol.
    rtol = k * torch.finfo(torch.float32).eps
    reference = torch.linalg.matrix_rank(
        matrix.cpu().double(), hermitian=True, rtol=rtol
    )
    assert reference.item() == 100  # construction sanity

    first = flag_gems.linalg_matrix_rank(matrix, hermitian=True)
    _assert_equal(first, reference.to(flag_gems.device))
    assert len(calls) == 1  # cold call probed exactly once
    second = flag_gems.linalg_matrix_rank(matrix, hermitian=True)
    _assert_equal(second, reference.to(flag_gems.device))
    assert len(calls) == 1  # verdict cached, no repeat
