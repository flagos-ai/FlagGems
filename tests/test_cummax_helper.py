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


def _dim_for_shape(shape):
    # Mirror the convention used by `tests/test_cummax.py`: the last reduction
    # shape is exercised along its middle axis to stay within the Triton grid
    # limits of the shared scan kernels.
    return 1 if shape == utils.REDUCTION_SHAPES[-1] else -1


@pytest.mark.cummax_helper
@pytest.mark.skipif(
    utils.SkipVersion("triton", "<3.0"),
    reason="Feature requires Triton >= 3.0.",
)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES)
def test_cummax_helper(shape, dtype):
    dim = _dim_for_shape(shape)

    if dtype in utils.INT_DTYPES:
        inp = torch.randint(-3, 3, shape, device=flag_gems.device).to(dtype)
    else:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    # `_cummax_helper` is the out-of-place helper: the caller allocates the
    # `values` / `indices` tensors, which the op then fills in-place along `dim`.
    values = torch.empty_like(inp)
    indices = torch.empty(inp.shape, dtype=torch.int64, device=flag_gems.device)

    # Reference is computed with `torch.cummax` (which internally dispatches to
    # `_cummax_helper`). The GEMS helper output is compared against it.
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.cummax(ref_inp, dim)
    ref_values = ref_out.values
    ref_indices = ref_out.indices

    with flag_gems.use_gems():
        torch.ops.aten._cummax_helper(inp, values, indices, dim)

    utils.gems_assert_close(values, ref_values, dtype, reduce_dim=shape[dim])
    utils.gems_assert_equal(indices, ref_indices)


@pytest.mark.cummax_helper
@pytest.mark.skipif(
    utils.SkipVersion("triton", "<3.0"),
    reason="Feature requires Triton >= 3.0.",
)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__cummax_helper_with_nan(shape, dtype):
    """Test _cummax_helper with NaN values (NaN propagation semantics)."""
    dim = _dim_for_shape(shape)

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
    ref_out = torch.cummax(ref_inp, dim)
    ref_values = ref_out.values
    ref_indices = ref_out.indices

    with flag_gems.use_gems():
        torch.ops.aten._cummax_helper(inp, values, indices, dim)

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
def test__cummax_helper_bool(shape):
    """Test _cummax_helper with boolean inputs (kept as bool, matching aten)."""
    dim = _dim_for_shape(shape)

    inp = torch.randint(0, 2, shape, device=flag_gems.device).to(torch.bool)

    values = torch.empty_like(inp)
    indices = torch.empty(inp.shape, dtype=torch.int64, device=flag_gems.device)

    # `to_reference` upcasts to float64 by default which would change the bool
    # semantics; keep the bool dtype intact on the reference side.
    ref_inp = utils.to_reference(inp)
    ref_out = torch.cummax(ref_inp, dim)
    ref_values = ref_out.values
    ref_indices = ref_out.indices

    with flag_gems.use_gems():
        torch.ops.aten._cummax_helper(inp, values, indices, dim)

    utils.gems_assert_equal(values, ref_values)
    utils.gems_assert_equal(indices, ref_indices)


@pytest.mark.cummax_helper
@pytest.mark.skipif(
    utils.SkipVersion("triton", "<3.0"),
    reason="Feature requires Triton >= 3.0.",
)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES)
def test__cummax_helper_dim0(shape, dtype):
    """Exercise a non-default dim (the leading axis) for each shape."""
    if len(shape) < 2:
        pytest.skip("dim=0 redundant for 1D shapes")
    # The shared scan kernels cap one of the grid axes; reduce along the leading
    # axis of the large 3D shapes would overflow that limit. Skip those.
    other = 1
    for d in shape[1:]:
        other *= d
    if other > 65535:
        pytest.skip("leading-axis reduce overflows the shared scan grid limit")

    if dtype in utils.INT_DTYPES:
        inp = torch.randint(-3, 3, shape, device=flag_gems.device).to(dtype)
    else:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    values = torch.empty_like(inp)
    indices = torch.empty(inp.shape, dtype=torch.int64, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.cummax(ref_inp, 0)
    ref_values = ref_out.values
    ref_indices = ref_out.indices

    with flag_gems.use_gems():
        torch.ops.aten._cummax_helper(inp, values, indices, 0)

    utils.gems_assert_close(values, ref_values, dtype, reduce_dim=shape[0])
    utils.gems_assert_equal(indices, ref_indices)
