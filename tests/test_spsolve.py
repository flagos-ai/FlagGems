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


def _make_diagonally_dominant_csr(n, dtype, device, density=0.3):
    """Build a well-conditioned, invertible sparse CSR matrix of shape (n, n).

    A random sparse pattern is made diagonally dominant so the linear system has
    a stable, unique solution across the tested dtypes.
    """
    dense = torch.randn((n, n), dtype=dtype, device=device)
    # Sparsify off-diagonal entries.
    mask = torch.rand((n, n), device=device) > density
    dense = dense.masked_fill(mask, 0.0)
    # Enforce diagonal dominance for invertibility / numerical stability.
    row_abs_sum = dense.abs().sum(dim=1)
    diag = row_abs_sum + 1.0
    dense = dense - torch.diag(torch.diagonal(dense)) + torch.diag(diag)
    return dense.to_sparse_csr(), dense


@pytest.mark.spsolve
@pytest.mark.parametrize("n", [4, 8, 16, 32])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_spsolve_vector_rhs(n, dtype):
    """Solve A @ x = b for a single right-hand-side vector (left=True).

    PyTorch's native ``_spsolve`` only supports a 1-D RHS with ``left=True``, so
    this is the sole accuracy contract. The reference is the mathematically
    equivalent dense solve (torch has no runnable native ``_spsolve`` here: CPU
    has no kernel, CUDA needs cuDSS).
    """
    A_csr, A_dense = _make_diagonally_dominant_csr(n, dtype, flag_gems.device)
    b = torch.randn((n,), dtype=dtype, device=flag_gems.device)

    ref_A = utils.to_reference(A_dense, True)
    ref_b = utils.to_reference(b, True)
    ref_out = torch.linalg.solve(ref_A, ref_b)

    res_out = flag_gems.spsolve(A_csr, b)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.spsolve
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_spsolve_identity(dtype):
    """A = I means the solution equals the right-hand side."""
    n = 16
    A_dense = torch.eye(n, dtype=dtype, device=flag_gems.device)
    A_csr = A_dense.to_sparse_csr()
    b = torch.randn((n,), dtype=dtype, device=flag_gems.device)

    res_out = flag_gems.spsolve(A_csr, b)
    ref_out = utils.to_reference(b, True)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.spsolve
def test_spsolve_rejects_matrix_rhs():
    """Native _spsolve only supports a 1-D RHS; a 2-D RHS must error out.

    Enabling FlagGems must preserve the native contract rather than silently
    accepting an unsupported matrix right-hand side.
    """
    A_csr, _ = _make_diagonally_dominant_csr(8, torch.float32, flag_gems.device)
    B = torch.randn((8, 3), dtype=torch.float32, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems.spsolve(A_csr, B)


@pytest.mark.spsolve
def test_spsolve_rejects_left_false():
    """Native _spsolve only supports left=True; left=False must error out."""
    A_csr, _ = _make_diagonally_dominant_csr(8, torch.float32, flag_gems.device)
    b = torch.randn((8,), dtype=torch.float32, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems.spsolve(A_csr, b, left=False)


@pytest.mark.spsolve
def test_spsolve_rejects_device_mismatch():
    """A CPU right-hand side with a device A must fail with a clean RuntimeError.

    Without the device check, a CPU B would reach the Triton kernels and die
    with a raw "Pointer argument ... cannot be accessed" error instead.
    """
    A_csr, _ = _make_diagonally_dominant_csr(8, torch.float32, flag_gems.device)
    b = torch.randn((8,), dtype=torch.float32, device="cpu")

    with pytest.raises(RuntimeError):
        flag_gems.spsolve(A_csr, b)


@pytest.mark.spsolve
def test_spsolve_requires_csr():
    """Passing a non-CSR matrix raises a clear error."""
    n = 8
    A_dense = torch.eye(n, device=flag_gems.device)
    b = torch.randn((n,), device=flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems.spsolve(A_dense, b)


@pytest.mark.spsolve
def test_spsolve_requires_square():
    """Passing a non-square matrix raises a clear error."""
    A_dense = torch.randn((4, 6), device=flag_gems.device)
    A_csr = A_dense.to_sparse_csr()
    b = torch.randn((4,), device=flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems.spsolve(A_csr, b)
