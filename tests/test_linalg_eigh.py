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

"""Tests for the user-facing linalg_eigh operator (torch.linalg.eigh path)."""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .linalg_eigh_utils import (
    EIG_2X2_LOWDTYPE,
    EIG_2X2_SHAPES,
    EIG_BATCH_2X2_SHAPES,
    EIG_BATCH_COMPLEX_SHAPES,
    EIG_BATCH_JACOBI_SHAPES,
    EIG_BATCH_TRIVIAL_SHAPES,
    EIG_COMPLEX_GLOBAL_SHAPES,
    EIG_COMPLEX_SHAPES,
    EIG_EVAL_ATOL,
    EIG_JACOBI_GLOBAL_SHAPES,
    EIG_JACOBI_TILE_SHAPES,
    EIG_TRIVIAL_COMPLEX_SHAPES,
    EIG_TRIVIAL_SHAPES,
    EIG_UPLO_U_COMPLEX_SHAPES,
    EIG_UPLO_U_SHAPES,
)
from .linalg_eigh_utils import assert_ascending as _assert_ascending
from .linalg_eigh_utils import assert_close as _assert_close
from .linalg_eigh_utils import check_eigh_decomposition as _check_eigh_decomposition
from .linalg_eigh_utils import gems_eigh_dispatch as _gems_eigh_dispatch
from .linalg_eigh_utils import make_symmetric_matrix
from .linalg_eigh_utils import symmetrise as _symmetrise


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_2X2_SHAPES,
    ids=[f"kernel_2x2-{s[0]}x{s[1]}" for s in EIG_2X2_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_eigh_2x2_kernel(shape, dtype):
    """n == 2 real: the closed-form `_eig_2x2_kernel`."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.linalg.eigh(ref_inp)

    with _gems_eigh_dispatch():
        res_out = torch.linalg.eigh(inp)

    utils.gems_assert_close(res_out[0], ref_out[0], dtype)
    _check_eigh_decomposition(inp, res_out[0], res_out[1])


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_JACOBI_TILE_SHAPES,
    ids=[f"jacobi_n{s[0]}" for s in EIG_JACOBI_TILE_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_eigh_jacobi(shape, dtype):
    """n > 2 real on the user path: register-resident Jacobi."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.linalg.eigh(ref_inp)

    with _gems_eigh_dispatch():
        res_out = torch.linalg.eigh(inp)

    # Eigenvalues element-wise (Jacobi tolerance), plus reconstruction.
    utils.gems_assert_close(res_out[0], ref_out[0], dtype, atol=EIG_EVAL_ATOL[dtype])
    _check_eigh_decomposition(inp, res_out[0], res_out[1])


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_COMPLEX_SHAPES,
    ids=[f"complex_n{s[0]}" for s in EIG_COMPLEX_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_linalg_eigh_complex(shape, dtype):
    """Complex inputs: real embedding then the real Jacobi path."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.linalg.eigh(ref_inp)

    with _gems_eigh_dispatch():
        res_out = torch.linalg.eigh(inp)

    # Eigenvalues of a Hermitian matrix are real.
    _assert_close(res_out[0], ref_out[0], res_out[0].dtype, atol=5e-4)
    _check_eigh_decomposition(inp, res_out[0], res_out[1])


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_BATCH_2X2_SHAPES,
    ids=[f"kernel_2x2-batch{s[0]}" for s in EIG_BATCH_2X2_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_eigh_batch_2x2_kernel(shape, dtype):
    """Batched n == 2 real: each batch element hits the 2x2 kernel."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.linalg.eigh(ref_inp)

    with _gems_eigh_dispatch():
        res_out = torch.linalg.eigh(inp)

    utils.gems_assert_close(res_out[0], ref_out[0], dtype)
    _check_eigh_decomposition(inp, res_out[0], res_out[1])


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_BATCH_JACOBI_SHAPES,
    ids=[f"jacobi-batch{s[-1]}" for s in EIG_BATCH_JACOBI_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_eigh_batch_jacobi(shape, dtype):
    """Batched n > 2 real: register-resident Jacobi per batch element."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.linalg.eigh(ref_inp)

    with _gems_eigh_dispatch():
        res_out = torch.linalg.eigh(inp)

    utils.gems_assert_close(res_out[0], ref_out[0], dtype, atol=EIG_EVAL_ATOL[dtype])
    _check_eigh_decomposition(inp, res_out[0], res_out[1])


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_TRIVIAL_SHAPES,
    ids=[f"trivial_n{s[0]}" for s in EIG_TRIVIAL_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_eigh_trivial(shape, dtype):
    """n < 2 (0x0 / 1x1) real: diagonal as eigenvalues, identity as eigenvectors."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.linalg.eigh(ref_inp)

    with _gems_eigh_dispatch():
        res_out = torch.linalg.eigh(inp)

    utils.gems_assert_close(res_out[0], ref_out[0], dtype)
    _check_eigh_decomposition(inp, res_out[0], res_out[1])


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_BATCH_TRIVIAL_SHAPES,
    ids=[f"trivial-batch_n{s[-1]}" for s in EIG_BATCH_TRIVIAL_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_eigh_batch_trivial(shape, dtype):
    """Batched n < 2 real on the user path: computed on device."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.linalg.eigh(ref_inp)

    with _gems_eigh_dispatch():
        res_out = torch.linalg.eigh(inp)

    utils.gems_assert_close(res_out[0], ref_out[0], dtype)
    _check_eigh_decomposition(inp, res_out[0], res_out[1])


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_2X2_SHAPES,
    ids=[f"lowdtype_2x2-{s[0]}x{s[1]}" for s in EIG_2X2_SHAPES],
)
@pytest.mark.parametrize("dtype", EIG_2X2_LOWDTYPE)
def test_linalg_eigh_2x2_low_precision(shape, dtype):
    """n == 2 fp16/bf16: the 2x2 path widens to fp32 on device and casts back.
    cuSOLVER reference is unavailable for these dtypes, so validate via
    reconstruction only. The output-dtype quantization of w and V bounds the
    achievable reconstruction accuracy: fp16 stays within 1e-2, while bf16
    (eps ~ 7.8e-3, quantization step ~2e-2 at |w| ~ 4) needs 2e-2."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    with _gems_eigh_dispatch():
        res_out = torch.linalg.eigh(inp)

    atol = 2e-2 if dtype == torch.bfloat16 else 1e-2
    _check_eigh_decomposition(inp, res_out[0], res_out[1], atol=atol)


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_TRIVIAL_COMPLEX_SHAPES,
    ids=[f"trivial_complex_n{s[0]}" for s in EIG_TRIVIAL_COMPLEX_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_linalg_eigh_trivial_complex(shape, dtype):
    """Complex n < 2 (0x0 / 1x1): diagonal/identity on device."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.linalg.eigh(ref_inp)

    with _gems_eigh_dispatch():
        res_out = torch.linalg.eigh(inp)

    _assert_close(res_out[0], ref_out[0], res_out[0].dtype)
    if res_out[1].numel() > 0:
        _check_eigh_decomposition(inp, res_out[0], res_out[1])


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_BATCH_COMPLEX_SHAPES,
    ids=[f"complex-batch{s[-1]}" for s in EIG_BATCH_COMPLEX_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_linalg_eigh_batch_complex(shape, dtype):
    """Batched complex n > 2: real embedding then the real Jacobi path."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.linalg.eigh(ref_inp)

    with _gems_eigh_dispatch():
        res_out = torch.linalg.eigh(inp)

    _assert_close(res_out[0], ref_out[0], res_out[0].dtype, atol=5e-4)
    _check_eigh_decomposition(inp, res_out[0], res_out[1])


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_UPLO_U_SHAPES,
    ids=[f"uplo_u_n{s[0]}" for s in EIG_UPLO_U_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_eigh_uplo_upper(shape, dtype):
    """UPLO="U": the upper triangle is used and the lower ignored. Inputs are
    made asymmetric so the triangle selection is genuinely exercised."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device, symmetric_only=False)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.linalg.eigh(ref_inp, UPLO="U")

    with _gems_eigh_dispatch():
        res_out = torch.linalg.eigh(inp, UPLO="U")

    utils.gems_assert_close(res_out[0], ref_out[0], dtype, atol=EIG_EVAL_ATOL[dtype])
    # Reconstruction against the symmetrised matrix (eigh uses one triangle).
    _check_eigh_decomposition(_symmetrise(inp, "U"), res_out[0], res_out[1])


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_UPLO_U_COMPLEX_SHAPES,
    ids=[f"uplo_u_complex_n{s[0]}" for s in EIG_UPLO_U_COMPLEX_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_linalg_eigh_uplo_upper_complex(shape, dtype):
    """UPLO="U" on the complex path."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device, symmetric_only=False)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.linalg.eigh(ref_inp, UPLO="U")

    with _gems_eigh_dispatch():
        res_out = torch.linalg.eigh(inp, UPLO="U")

    _assert_close(res_out[0], ref_out[0], res_out[0].dtype, atol=5e-4)
    _check_eigh_decomposition(_symmetrise(inp, "U"), res_out[0], res_out[1])


# ---------------------------------------------------------------------------
# Global-memory round-robin path (n > 64 real, 2n > 128 complex): the large-n
# Jacobi kernels. Verified by reconstruction + ascending order only; the
# trailing precision is worse than cuSOLVER at these sizes.
# ---------------------------------------------------------------------------


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_JACOBI_GLOBAL_SHAPES,
    ids=[f"jacobi_global_n{s[0]}" for s in EIG_JACOBI_GLOBAL_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_eigh_jacobi_global(shape, dtype):
    """n > 64 real: the global-memory round-robin Jacobi path."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.linalg.eigh(ref_inp)

    with _gems_eigh_dispatch():
        res_out = torch.linalg.eigh(inp)

    _check_eigh_decomposition(inp, res_out[0], res_out[1], atol=1e-1)
    _assert_ascending(res_out[0])
    # Eigenvalue set agrees with the reference to the global-path tolerance.
    res_w = utils.to_cpu(res_out[0], ref_out[0])
    torch.testing.assert_close(
        res_w.sort().values,
        ref_out[0].sort().values,
        atol=2e-2,
        rtol=1e-3,
    )


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_COMPLEX_GLOBAL_SHAPES,
    ids=[f"complex_global_n{s[0]}" for s in EIG_COMPLEX_GLOBAL_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.complex64])
def test_linalg_eigh_complex_global(shape, dtype):
    """Complex n >= 48: real embedding (2n >= 96) hits the global Jacobi path."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.linalg.eigh(ref_inp)

    with _gems_eigh_dispatch():
        res_out = torch.linalg.eigh(inp)

    # Eigenvalues of a Hermitian matrix are real.
    _assert_close(res_out[0], ref_out[0], res_out[0].dtype, atol=2e-2)
    _check_eigh_decomposition(inp, res_out[0], res_out[1], atol=1e-1)
    _assert_ascending(res_out[0])


@pytest.mark.linalg_eigh
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_linalg_eigh_ascending_order(dtype):
    """Eigenvalues are returned in ascending order for every path."""
    shapes = {
        torch.float32: [(2, 2), (8, 8), (96, 96)],
        torch.complex64: [(2, 2), (8, 8), (48, 48)],
    }[dtype]
    for shape in shapes:
        inp = make_symmetric_matrix(shape, dtype, flag_gems.device)
        with _gems_eigh_dispatch():
            res_out = torch.linalg.eigh(inp)
        _assert_ascending(res_out[0])


@pytest.mark.linalg_eigh
def test_linalg_eigh_nonsquare_raises():
    """A non-square input must raise ValueError on the Gems path."""
    A = torch.randn(3, 5, dtype=torch.float32, device=flag_gems.device)
    with pytest.raises(ValueError):
        with _gems_eigh_dispatch():
            torch.linalg.eigh(A)


@pytest.mark.linalg_eigh
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_linalg_eigh_non_contiguous(dtype):
    """A non-contiguous (transposed-view) symmetric input is accepted."""
    n = 8
    A = make_symmetric_matrix((n, n), dtype, flag_gems.device)
    # A transposed view is non-contiguous but still symmetric (A == A.T/mH).
    view = A.transpose(-2, -1).contiguous().transpose(-2, -1)
    assert not view.is_contiguous()

    ref_inp = utils.to_reference(view)
    ref_out = torch.linalg.eigh(ref_inp)

    with _gems_eigh_dispatch():
        res_out = torch.linalg.eigh(view)

    if dtype == torch.complex64:
        _assert_close(res_out[0], ref_out[0], res_out[0].dtype, atol=5e-4)
    else:
        utils.gems_assert_close(
            res_out[0], ref_out[0], dtype, atol=EIG_EVAL_ATOL[dtype]
        )
    _check_eigh_decomposition(view, res_out[0], res_out[1])
