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

# Eigenvalue decomposition needs a square matrix, and cuSOLVER only covers
# float32 among the real dtypes.
EIGVALS_SHAPES = [(2, 2), (3, 3), (5, 5), (10, 10), (20, 20)]


def _sorted_eigvals(values):
    """Order eigenvalues lexicographically by (real, imag).

    Eigenvalues come back in an unspecified order, and LAPACK and cuSOLVER
    genuinely disagree on it for some inputs, so the order has to be
    normalised before an element-wise comparison is meaningful.
    """
    ordered = values[torch.argsort(values.imag, stable=True)]
    return ordered[torch.argsort(ordered.real, stable=True)]


@pytest.mark.linalg_eigvals
@pytest.mark.parametrize("shape", EIGVALS_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_eigvals(shape, dtype):
    """Test _linalg_eigvals accuracy against PyTorch reference."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._linalg_eigvals.default(ref_inp)
    res_out = flag_gems.ops._linalg_eigvals(inp)

    # For float32 input, the eigenvalues come back as complex64.
    utils.gems_assert_close(
        _sorted_eigvals(res_out), _sorted_eigvals(ref_out), res_out.dtype
    )


@pytest.mark.linalg_eigvals
@pytest.mark.parametrize("shape", EIGVALS_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_eigvals_default(shape, dtype):
    """The canonical linalg_eigvals entry must reach the GEMS wrapper."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.linalg_eigvals.default(ref_inp)
    res_out = flag_gems.linalg_eigvals(inp)

    utils.gems_assert_close(
        _sorted_eigvals(res_out), _sorted_eigvals(ref_out), res_out.dtype
    )


@pytest.mark.linalg_eigvals_out
@pytest.mark.parametrize("shape", EIGVALS_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_eigvals_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    # complex64 is the eigenvalue dtype produced for a float32 input
    ref_out = torch.empty(shape[0], dtype=torch.complex64, device=ref_inp.device)
    torch.ops.aten.linalg_eigvals.out(ref_inp, out=ref_out)

    res_out = torch.empty(shape[0], dtype=torch.complex64, device=flag_gems.device)
    returned = flag_gems.ops.linalg_eigvals_out(inp, out=res_out)

    # the out variant must write in place and return the very same tensor
    assert returned.data_ptr() == res_out.data_ptr()
    utils.gems_assert_close(
        _sorted_eigvals(res_out), _sorted_eigvals(ref_out), res_out.dtype
    )


@pytest.mark.linalg_eigvals_out
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_eigvals_out_resize(dtype):
    """An empty ``out`` must be resized to the eigenvalue shape."""
    inp = torch.randn((6, 6), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.empty(0, dtype=torch.complex64, device=ref_inp.device)
    torch.ops.aten.linalg_eigvals.out(ref_inp, out=ref_out)

    res_out = torch.empty(0, dtype=torch.complex64, device=flag_gems.device)
    flag_gems.ops.linalg_eigvals_out(inp, out=res_out)

    assert res_out.shape == (6,)
    utils.gems_assert_close(
        _sorted_eigvals(res_out), _sorted_eigvals(ref_out), res_out.dtype
    )
