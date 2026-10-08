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
from . import conftest as cfg

# Reduction shapes plus two extra cases: a large prime-sized 1D tensor and a
# 3D tensor whose inner axis exceeds the shared scan kernel's contiguous stride,
# to exercise the non-contiguous staging path.
if cfg.QUICK_MODE:
    # Minimal shape for quick smoke runs.
    SHAPES = [(2, 32)]
else:
    # Reduction shapes plus non-contiguous edge cases (see top comment).
    SHAPES = utils.REDUCTION_SHAPES + [(2637,), (16, 1025, 255)]


# The last reduction shape (a large 3D tensor) is exercised along its middle
# axis to stay within the Triton grid limits of the shared scan kernels. For
# every other shape we reduce along the last axis.
def _pos_dim(shape):
    if shape == utils.REDUCTION_SHAPES[-1]:
        return 1
    return len(shape) - 1


def _neg_dim(shape):
    # Negative counterpart of `_pos_dim`. `dim % input.ndim` normalises both
    # to the same axis, so the two exercises are semantically equivalent; they
    # only differ in which reference path is valid (see the note below).
    if shape == utils.REDUCTION_SHAPES[-1]:
        return 1 - len(shape)
    return -1


# NOTE on the two reference paths
# --------------------------------
# `torch.ops.aten._cummax_helper` is the out-of-place kernel that
# `torch.cummax` dispatches to, and it is also the op under test here, so it is
# the most faithful reference. However, PyTorch's native `aten::_cummax_helper`
# does *not* accept a negative `dim` -- it segfaults on CUDA for any negative
# dim across every dtype (a PyTorch native bug; `torch.cummax` normalises the
# dim before dispatching, which is why the public API tolerates negatives).
#
# Therefore each test below is split into a positive-dim variant (whose
# reference is `torch.ops.aten._cummax_helper`) and a negative-dim variant
# (whose reference is `torch.cummax`). The FlagGems helper itself accepts both
# signs -- it normalises `dim` internally -- so both variants exercise the same
# GEMS code path; the split is purely about which reference is usable.


def _run_gems(inp, values, indices, dim):
    flag_gems._cummax_helper(inp, values, indices, dim)


def _ref_with_helper(ref_inp, dim):
    """Reference via the aten helper itself. Only valid for non-negative dim."""
    ref_values = torch.empty_like(ref_inp)
    ref_indices = torch.empty(ref_inp.shape, dtype=torch.int64, device=ref_inp.device)
    torch.ops.aten._cummax_helper(ref_inp, ref_values, ref_indices, dim)
    return ref_values, ref_indices


def _ref_with_cummax(ref_inp, dim):
    """Reference via the public `torch.cummax`. Required for negative dim,
    where the raw aten helper is unusable (see note above)."""
    ref_out = torch.cummax(ref_inp, dim)
    return ref_out.values, ref_out.indices


def _make_input(shape, dtype):
    if dtype in utils.INT_DTYPES:
        return torch.randint(-3, 3, shape, device=flag_gems.device).to(dtype)
    return torch.randn(shape, dtype=dtype, device=flag_gems.device)


# ---------------------------------------------------------------------------
# Positive-dim variants -- reference is `torch.ops.aten._cummax_helper`.
# ---------------------------------------------------------------------------


@pytest.mark.cummax_helper
@pytest.mark.skipif(
    utils.SkipVersion("triton", "<3.0"),
    reason="Feature requires Triton >= 3.0.",
)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES)
def test_cummax_helper_posdim(shape, dtype):
    dim = _pos_dim(shape)
    inp = _make_input(shape, dtype)

    values = torch.empty_like(inp)
    indices = torch.empty(inp.shape, dtype=torch.int64, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, True)
    ref_values, ref_indices = _ref_with_helper(ref_inp, dim)

    _run_gems(inp, values, indices, dim)

    utils.gems_assert_close(values, ref_values, dtype, reduce_dim=shape[dim])
    utils.gems_assert_equal(indices, ref_indices)


@pytest.mark.cummax_helper
@pytest.mark.skipif(
    utils.SkipVersion("triton", "<3.0"),
    reason="Feature requires Triton >= 3.0.",
)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cummax_helper_posdim_with_nan(shape, dtype):
    """Test _cummax_helper with NaN values (NaN propagation semantics)."""
    dim = _pos_dim(shape)

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    total_elements = inp.numel()
    nan_count = max(1, int(total_elements * 0.1))
    nan_indices = torch.randperm(total_elements, device=flag_gems.device)[:nan_count]
    flat_inp = inp.flatten()
    flat_inp[nan_indices] = float("nan")
    inp = flat_inp.view(shape)

    values = torch.empty_like(inp)
    indices = torch.empty(inp.shape, dtype=torch.int64, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, True)
    ref_values, ref_indices = _ref_with_helper(ref_inp, dim)

    _run_gems(inp, values, indices, dim)

    utils.gems_assert_close(
        values, ref_values, dtype, reduce_dim=shape[dim], equal_nan=True
    )
    utils.gems_assert_equal(indices, ref_indices)


@pytest.mark.cummax_helper
@pytest.mark.skipif(
    utils.SkipVersion("triton", "<3.0"),
    reason="Feature requires Triton >= 3.0.",
)
@pytest.mark.parametrize("shape", SHAPES)
def test_cummax_helper_posdim_bool(shape):
    """Test _cummax_helper with boolean inputs (kept as bool, matching aten)."""
    dim = _pos_dim(shape)

    inp = torch.randint(0, 2, shape, device=flag_gems.device).to(torch.bool)

    values = torch.empty_like(inp)
    indices = torch.empty(inp.shape, dtype=torch.int64, device=flag_gems.device)

    # `to_reference` upcasts to float64 by default which would change the bool
    # semantics; keep the bool dtype intact on the reference side.
    ref_inp = utils.to_reference(inp)
    ref_values, ref_indices = _ref_with_helper(ref_inp, dim)

    _run_gems(inp, values, indices, dim)

    utils.gems_assert_equal(values, ref_values)
    utils.gems_assert_equal(indices, ref_indices)


# ---------------------------------------------------------------------------
# Negative-dim variants -- reference is `torch.cummax`, because PyTorch's
# native `aten::_cummax_helper` segfaults on a negative dim (see note above).
# ---------------------------------------------------------------------------


@pytest.mark.cummax_helper
@pytest.mark.skipif(
    utils.SkipVersion("triton", "<3.0"),
    reason="Feature requires Triton >= 3.0.",
)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES)
def test_cummax_helper_negdim(shape, dtype):
    dim = _neg_dim(shape)
    inp = _make_input(shape, dtype)

    values = torch.empty_like(inp)
    indices = torch.empty(inp.shape, dtype=torch.int64, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, True)
    ref_values, ref_indices = _ref_with_cummax(ref_inp, dim)

    _run_gems(inp, values, indices, dim)

    utils.gems_assert_close(values, ref_values, dtype, reduce_dim=shape[dim])
    utils.gems_assert_equal(indices, ref_indices)


@pytest.mark.cummax_helper
@pytest.mark.skipif(
    utils.SkipVersion("triton", "<3.0"),
    reason="Feature requires Triton >= 3.0.",
)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cummax_helper_negdim_with_nan(shape, dtype):
    """Test _cummax_helper with NaN values (NaN propagation semantics)."""
    dim = _neg_dim(shape)

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    total_elements = inp.numel()
    nan_count = max(1, int(total_elements * 0.1))
    nan_indices = torch.randperm(total_elements, device=flag_gems.device)[:nan_count]
    flat_inp = inp.flatten()
    flat_inp[nan_indices] = float("nan")
    inp = flat_inp.view(shape)

    values = torch.empty_like(inp)
    indices = torch.empty(inp.shape, dtype=torch.int64, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, True)
    ref_values, ref_indices = _ref_with_cummax(ref_inp, dim)

    _run_gems(inp, values, indices, dim)

    utils.gems_assert_close(
        values, ref_values, dtype, reduce_dim=shape[dim], equal_nan=True
    )
    utils.gems_assert_equal(indices, ref_indices)


@pytest.mark.cummax_helper
@pytest.mark.skipif(
    utils.SkipVersion("triton", "<3.0"),
    reason="Feature requires Triton >= 3.0.",
)
@pytest.mark.parametrize("shape", SHAPES)
def test_cummax_helper_negdim_bool(shape):
    """Test _cummax_helper with boolean inputs (kept as bool, matching aten)."""
    dim = _neg_dim(shape)

    inp = torch.randint(0, 2, shape, device=flag_gems.device).to(torch.bool)

    values = torch.empty_like(inp)
    indices = torch.empty(inp.shape, dtype=torch.int64, device=flag_gems.device)

    # `to_reference` upcasts to float64 by default which would change the bool
    # semantics; keep the bool dtype intact on the reference side.
    ref_inp = utils.to_reference(inp)
    ref_values, ref_indices = _ref_with_cummax(ref_inp, dim)

    _run_gems(inp, values, indices, dim)

    utils.gems_assert_equal(values, ref_values)
    utils.gems_assert_equal(indices, ref_indices)


# ---------------------------------------------------------------------------
# Leading-axis (dim=0) coverage for multi-dim shapes. Uses the aten helper as
# the reference (dim=0 is non-negative), skipping 1D shapes (redundant) and
# large 3D shapes whose non-reduced plane would overflow the scan grid.
# ---------------------------------------------------------------------------


@pytest.mark.cummax_helper
@pytest.mark.skipif(
    utils.SkipVersion("triton", "<3.0"),
    reason="Feature requires Triton >= 3.0.",
)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES)
def test_cummax_helper_dim0(shape, dtype):
    """Exercise the leading axis (dim=0) for each multi-dim shape."""
    if len(shape) < 2:
        pytest.skip("dim=0 redundant for 1D shapes")
    # The shared scan kernels cap one of the grid axes; reduce along the leading
    # axis of the large 3D shapes would overflow that limit. Skip those.
    other = 1
    for d in shape[1:]:
        other *= d
    if other > 65535:
        pytest.skip("leading-axis reduce overflows the shared scan grid limit")

    inp = _make_input(shape, dtype)

    values = torch.empty_like(inp)
    indices = torch.empty(inp.shape, dtype=torch.int64, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, True)
    ref_values, ref_indices = _ref_with_helper(ref_inp, 0)

    _run_gems(inp, values, indices, 0)

    utils.gems_assert_close(values, ref_values, dtype, reduce_dim=shape[0])
    utils.gems_assert_equal(indices, ref_indices)
