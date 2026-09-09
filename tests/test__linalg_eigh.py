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

"""Tests for the underlying _linalg_eigh aten operator.

Directly calls ``aten::_linalg_eigh.default``, exercising the ``compute_v``
argument (True/False) that distinguishes ``_linalg_eigh`` from the
user-facing ``linalg_eigh`` wrapper.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .linalg_eigh_utils import (
    EIG_2X2_SHAPES,
    EIG_COMPLEX_GLOBAL_SHAPES,
    EIG_EVAL_ATOL,
    EIG_JACOBI_GLOBAL_SHAPES,
    EIG_JACOBI_TILE_SHAPES,
)
from .linalg_eigh_utils import assert_ascending as _assert_ascending
from .linalg_eigh_utils import check_eigh_decomposition as _check_eigh_decomposition
from .linalg_eigh_utils import gems_eigh_dispatch as _gems_eigh_dispatch
from .linalg_eigh_utils import make_symmetric_matrix


@pytest.mark.underscore_linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_2X2_SHAPES + EIG_JACOBI_TILE_SHAPES,
    ids=[f"ueigh-{s[0]}x{s[1]}" for s in EIG_2X2_SHAPES + EIG_JACOBI_TILE_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_underlying_linalg_eigh(shape, dtype):
    """Directly call aten::_linalg_eigh.default with compute_v=True.

    n == 2 hits the 2x2 kernel; n > 2 hits the Jacobi path.
    """
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_w, ref_v = torch.ops.aten._linalg_eigh.default(ref_inp, "L", True)

    with _gems_eigh_dispatch():
        res_w, res_v = torch.ops.aten._linalg_eigh.default(inp, "L", True)

    utils.gems_assert_close(res_w, ref_w, dtype, atol=EIG_EVAL_ATOL[dtype])
    _check_eigh_decomposition(inp, res_w, res_v)


@pytest.mark.underscore_linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_JACOBI_TILE_SHAPES,
    ids=[f"ueigh_no_v_n{s[0]}" for s in EIG_JACOBI_TILE_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_underlying_linalg_eigh_no_vectors(shape, dtype):
    """compute_v=False returns eigenvalues only (empty eigenvectors)."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_w, _ = torch.ops.aten._linalg_eigh.default(ref_inp, "L", False)

    with _gems_eigh_dispatch():
        res_w, res_v = torch.ops.aten._linalg_eigh.default(inp, "L", False)

    utils.gems_assert_close(res_w, ref_w, dtype, atol=EIG_EVAL_ATOL[dtype])
    # Eigenvectors tensor is empty when compute_v=False.
    assert res_v.numel() == 0


@pytest.mark.underscore_linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_2X2_SHAPES,
    ids=[f"ueigh_no_v_2x2-{s[0]}x{s[1]}" for s in EIG_2X2_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_underlying_linalg_eigh_no_vectors_2x2(shape, dtype):
    """compute_v=False with n == 2: still returns eigenvalues only."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_w, _ = torch.ops.aten._linalg_eigh.default(ref_inp, "L", False)

    with _gems_eigh_dispatch():
        res_w, res_v = torch.ops.aten._linalg_eigh.default(inp, "L", False)

    utils.gems_assert_close(res_w, ref_w, dtype)
    assert res_v.numel() == 0


@pytest.mark.underscore_linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_JACOBI_GLOBAL_SHAPES,
    ids=[f"ueigh_no_v_global_n{s[0]}" for s in EIG_JACOBI_GLOBAL_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_underlying_linalg_eigh_no_vectors_global(shape, dtype):
    """compute_v=False on the global-memory Jacobi path (n > 64).

    Eigenvalues only; eigenvector computation is skipped (no V sort), and
    the eigenvector tensor is empty.
    """
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_w, _ = torch.ops.aten._linalg_eigh.default(ref_inp, "L", False)

    with _gems_eigh_dispatch():
        res_w, res_v = torch.ops.aten._linalg_eigh.default(inp, "L", False)

    res_w_cpu = utils.to_cpu(res_w, ref_w)
    torch.testing.assert_close(
        res_w_cpu.sort().values,
        ref_w.sort().values,
        atol=2e-2,
        rtol=1e-3,
    )
    _assert_ascending(res_w)
    assert res_v.numel() == 0


@pytest.mark.underscore_linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_COMPLEX_GLOBAL_SHAPES,
    ids=[f"ueigh_no_v_complex_global_n{s[0]}" for s in EIG_COMPLEX_GLOBAL_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.complex64])
def test_underlying_linalg_eigh_no_vectors_complex_global(shape, dtype):
    """compute_v=False on the complex global path (complex64, 2n > 128).

    The complex pick kernel is skipped; eigenvalues come from the real
    embedding's Jacobi path alone, and eigenvectors are empty.
    """
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_w, _ = torch.ops.aten._linalg_eigh.default(ref_inp, "L", False)

    with _gems_eigh_dispatch():
        res_w, res_v = torch.ops.aten._linalg_eigh.default(inp, "L", False)

    res_w_cpu = utils.to_cpu(res_w, ref_w)
    torch.testing.assert_close(
        res_w_cpu.sort().values,
        ref_w.sort().values,
        atol=2e-2,
        rtol=1e-3,
    )
    _assert_ascending(res_w)
    assert res_v.numel() == 0
