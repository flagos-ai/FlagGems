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
    HELPER_SHAPES = [(2, 32)]
else:
    # Reduction shapes plus non-contiguous edge cases (see top comment).
    HELPER_SHAPES = utils.REDUCTION_SHAPES + [(2637,), (16, 1025, 255)]


def _run_helper(inp, dim):
    """Invoke the GEMS-registered ``_cummin_helper`` writing into freshly
    allocated ``values`` / ``indices`` tensors."""
    values = torch.empty_like(inp)
    indices = torch.empty(inp.shape, dtype=torch.int64, device=inp.device)
    torch.ops.aten._cummin_helper(inp, values, indices, dim)
    return values, indices


def _ref_helper(inp, dim):
    """Reference implementation. PyTorch's native ``_cummin_helper`` crashes on
    some shape/dtype combinations on CUDA, so we build the reference from the
    stable ``torch.cummin`` and write the results into allocated buffers."""
    out = torch.cummin(inp, dim=dim)
    values = out.values.contiguous()
    indices = out.indices.to(torch.int64).contiguous()
    return values, indices


@pytest.mark.cummin_helper
@pytest.mark.skipif(
    utils.SkipVersion("triton", "<3.0"),
    reason="Feature requires Triton >= 3.0",
)
@pytest.mark.parametrize("shape", HELPER_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES)
def test_cummin_helper(shape, dtype):
    dim = 1 if shape == utils.REDUCTION_SHAPES[-1] else -1
    if dtype in utils.INT_DTYPES:
        inp = torch.randint(-3, 3, shape, device=flag_gems.device).to(dtype)
    else:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, True)

    ref_values, ref_indices = _ref_helper(ref_inp, dim)
    with flag_gems.use_gems():
        res_values, res_indices = _run_helper(inp, dim)

    utils.gems_assert_close(res_values, ref_values, dtype, reduce_dim=shape[dim])
    utils.gems_assert_equal(res_indices, ref_indices)


@pytest.mark.cummin_helper
@pytest.mark.parametrize("shape", HELPER_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("nan_ratio", [0.1, 0.3, 0.5])
def test_cummin_helper_with_nan(shape, dtype, nan_ratio):
    """Test _cummin_helper with NaN values at different ratios"""
    dim = 1 if shape == utils.REDUCTION_SHAPES[-1] else -1

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    total_elements = inp.numel()
    nan_count = int(total_elements * nan_ratio)
    nan_indices = torch.randperm(total_elements)[:nan_count]
    flat_inp = inp.flatten()
    flat_inp[nan_indices] = float("nan")
    inp = flat_inp.view(shape)

    ref_inp = utils.to_reference(inp, True)

    ref_values, ref_indices = _ref_helper(ref_inp, dim)
    with flag_gems.use_gems():
        res_values, res_indices = _run_helper(inp, dim)

    utils.gems_assert_close(
        res_values,
        ref_values,
        dtype,
        reduce_dim=shape[dim],
        equal_nan=True,
    )
    utils.gems_assert_equal(res_indices, ref_indices, equal_nan=True)


@pytest.mark.cummin_helper
@pytest.mark.parametrize("shape", HELPER_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES)
@pytest.mark.parametrize("dim", [0, 1, -1])
def test_cummin_helper_dim(shape, dtype, dim):
    if abs(dim) >= len(shape):
        pytest.skip("dim out of range for shape")
    # The underlying FlagGems scan-then-fan kernel that this helper reuses
    # launches one program per non-reduced plane; reducing an outer dimension of
    # a large 3D tensor would exceed the supported grid size, so skip that case.
    non_reduced = 1
    for i, s in enumerate(shape):
        if i != dim % len(shape):
            non_reduced *= s
    if dim == 0 and non_reduced > 1 << 14:
        pytest.skip("non-reduced plane too large for scan kernel grid")
    if dtype in utils.INT_DTYPES:
        inp = torch.randint(-3, 3, shape, device=flag_gems.device).to(dtype)
    else:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, True)

    ref_values, ref_indices = _ref_helper(ref_inp, dim)
    with flag_gems.use_gems():
        res_values, res_indices = _run_helper(inp, dim)

    utils.gems_assert_close(res_values, ref_values, dtype, reduce_dim=shape[dim])
    utils.gems_assert_equal(res_indices, ref_indices)
