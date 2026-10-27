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

# Eigenvalue decomposition needs a square matrix. The native CUDA kernel
# accepts float32/complex64/complex128 (float64 upcasts to complex128).
EIGVALS_SHAPES = [(2, 2), (3, 3), (5, 5), (10, 10), (20, 20)]


def _sorted_eigvals(values):
    """Order eigenvalues lexicographically by (real, imag).

    Eigenvalues come back in an unspecified order, and LAPACK and cuSOLVER
    genuinely disagree on it for some inputs, so the order has to be
    normalised before an element-wise comparison is meaningful.  Works on the
    flattened view so batched inputs sort the eigenvalue axis, not dim 0.
    """
    shape = values.shape
    flat = values.reshape(-1)
    ordered = flat[torch.argsort(flat.imag, stable=True)]
    ordered = ordered[torch.argsort(ordered.real, stable=True)]
    return ordered.reshape(shape)


# ---------------------------------------------------------------------------
# aten::_linalg_eigvals
#
# The private primitive behind torch.linalg.eigvals. The operator id keeps the
# leading underscore, and pytest refuses to build a marker from an attribute
# starting with an underscore, so the marker is spelled
# `underscore_linalg_eigvals` (the #6359 convention; see the same pattern for
# `_linalg_eigh` / `_linalg_svd`). The tests live in this file alongside the
# public entries because they share shapes, helpers and tolerances.
# ---------------------------------------------------------------------------


@pytest.mark.underscore_linalg_eigvals
@pytest.mark.parametrize("shape", EIGVALS_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32])
def test__linalg_eigvals(shape, dtype):
    """Test _linalg_eigvals accuracy against PyTorch reference."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._linalg_eigvals.default(ref_inp)
    res_out = flag_gems.ops._linalg_eigvals(inp)

    # For float32 input, the eigenvalues come back as complex64.
    utils.gems_assert_close(
        _sorted_eigvals(res_out), _sorted_eigvals(ref_out), res_out.dtype
    )


@pytest.mark.underscore_linalg_eigvals
@pytest.mark.parametrize("shape", EIGVALS_SHAPES)
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test__linalg_eigvals_complex(shape, dtype):
    """Complex input is supported natively and must stay delegate-clean."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._linalg_eigvals.default(ref_inp)
    res_out = flag_gems.ops._linalg_eigvals(inp)

    assert res_out.dtype == dtype
    utils.gems_assert_close(
        _sorted_eigvals(res_out), _sorted_eigvals(ref_out), res_out.dtype
    )


@pytest.mark.underscore_linalg_eigvals
@pytest.mark.parametrize("shape", [(3, 4, 4), (2, 3, 5, 5)])
def test__linalg_eigvals_batched(shape):
    """Batched square matrices must keep the native batched result shape."""
    inp = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._linalg_eigvals.default(ref_inp)
    res_out = flag_gems.ops._linalg_eigvals(inp)

    assert res_out.shape == ref_out.shape == (*shape[:-1],)
    utils.gems_assert_close(
        _sorted_eigvals(res_out), _sorted_eigvals(ref_out), res_out.dtype
    )


# ---------------------------------------------------------------------------
# aten::linalg_eigvals / aten::linalg_eigvals.out
# ---------------------------------------------------------------------------


@pytest.mark.linalg_eigvals
@pytest.mark.parametrize("shape", EIGVALS_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_eigvals(shape, dtype):
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


@pytest.mark.linalg_eigvals_out
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_eigvals_out_dtype_mismatch(dtype):
    """A non-castable ``out`` must raise instead of truncating the imaginary part.

    A float32 ``out`` used to silently drop the imaginary component of every
    eigenvalue; the native kernel rejects it, and that contract must hold
    through the GEMS wrapper.
    """
    inp = torch.randn((5, 5), dtype=dtype, device=flag_gems.device)
    out = torch.empty(5, dtype=torch.float32, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems.ops.linalg_eigvals_out(inp, out=out)


@pytest.mark.linalg_eigvals_out
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_eigvals_out_non_contiguous(dtype):
    """A non-contiguous ``out`` view must be written correctly in place."""
    inp = torch.randn((6, 6), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_storage = torch.empty(20, dtype=torch.complex64, device=ref_inp.device)
    ref_out = ref_storage[::3][:6]
    torch.ops.aten.linalg_eigvals.out(ref_inp, out=ref_out)

    res_storage = torch.empty(20, dtype=torch.complex64, device=flag_gems.device)
    res_out = res_storage[::3][:6]
    returned = flag_gems.ops.linalg_eigvals_out(inp, out=res_out)

    assert not res_out.is_contiguous()
    assert returned.data_ptr() == res_out.data_ptr()
    utils.gems_assert_close(
        _sorted_eigvals(res_out), _sorted_eigvals(ref_out), res_out.dtype
    )


@pytest.mark.linalg_eigvals_out
@pytest.mark.parametrize("dtype", [torch.float64])
def test_linalg_eigvals_out_float64(dtype):
    """float64 input returns complex128 and must stay on device."""
    inp = torch.randn((5, 5), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.empty(5, dtype=torch.complex128, device=ref_inp.device)
    torch.ops.aten.linalg_eigvals.out(ref_inp, out=ref_out)

    res_out = torch.empty(5, dtype=torch.complex128, device=flag_gems.device)
    flag_gems.ops.linalg_eigvals_out(inp, out=res_out)

    assert res_out.dtype == torch.complex128
    utils.gems_assert_close(
        _sorted_eigvals(res_out), _sorted_eigvals(ref_out), res_out.dtype
    )
