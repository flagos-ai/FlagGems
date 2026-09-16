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

KEEPDIM = [False, True]


def _unique_input(shape, dim, dtype):
    """Distinct values along ``dim``, so the median index is unambiguous."""
    base = torch.arange(
        torch.Size(shape).numel(), device=flag_gems.device, dtype=torch.int64
    ).reshape(shape)
    order = torch.argsort(torch.rand(shape, device=flag_gems.device), dim=dim)
    return torch.gather(base, dim, order).to(dtype)


def _selected_along_dim(inp, dim, indices, keepdim):
    """Elements that ``indices`` actually points at."""
    idx = indices if keepdim else indices.unsqueeze(dim)
    picked = torch.gather(inp, dim, idx)
    return picked if keepdim else picked.squeeze(dim)


def _call_median_dim_values(inp, dim, keepdim, equal_nan=False, exact_indices=True):
    """Reference against ATen; compare indices only when they are well defined."""
    ref_inp = utils.to_reference(inp)

    ref_result = torch.median(ref_inp, dim=dim, keepdim=keepdim)
    values = torch.empty(
        ref_result.values.shape, dtype=inp.dtype, device=flag_gems.device
    )
    indices = torch.empty(
        ref_result.indices.shape, dtype=torch.int64, device=flag_gems.device
    )
    ref_values = torch.empty(
        ref_result.values.shape, dtype=inp.dtype, device=ref_inp.device
    )
    ref_indices = torch.empty(
        ref_result.indices.shape, dtype=torch.int64, device=ref_inp.device
    )
    torch.ops.aten.median.dim_values(
        ref_inp, dim, keepdim, values=ref_values, indices=ref_indices
    )

    result = flag_gems.median_dim_values(
        inp, dim=dim, keepdim=keepdim, values=values, indices=indices
    )

    # The wrapper must fill and hand back the caller's buffers.
    assert result.values is values
    assert result.indices is indices
    utils.gems_assert_equal(result.values, ref_result.values, equal_nan=equal_nan)
    utils.gems_assert_equal(values, ref_values, equal_nan=equal_nan)

    # Whatever index comes back has to point at the median it reports.  Both
    # sides live on the accelerator, so move them to the reference device
    # first: gems_assert_equal expects `ref` to be there when --ref=cpu is used.
    utils.gems_assert_equal(
        utils.to_reference(_selected_along_dim(inp, dim, result.indices, keepdim)),
        utils.to_reference(result.values),
        equal_nan=equal_nan,
    )
    if exact_indices:
        utils.gems_assert_equal(result.indices, ref_result.indices)
    return result


@pytest.mark.median_dim_values
@pytest.mark.parametrize("keepdim", KEEPDIM)
@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.ALL_INT_DTYPES)
def test_median_dim_values(dim, keepdim, dtype):
    inp = _unique_input((9, 7), dim, dtype)
    _call_median_dim_values(inp, dim, keepdim)


@pytest.mark.median_dim_values
@pytest.mark.parametrize("keepdim", KEEPDIM)
@pytest.mark.parametrize("dtype", utils.ALL_INT_DTYPES)
def test_median_dim_values_keepdim_shapes(keepdim, dtype):
    inp = _unique_input((9, 7), 1, dtype)
    result = _call_median_dim_values(inp, 1, keepdim)

    expected = (9, 1) if keepdim else (9,)
    assert tuple(result.values.shape) == expected
    assert tuple(result.indices.shape) == expected


@pytest.mark.median_dim_values
@pytest.mark.parametrize("keepdim", KEEPDIM)
def test_median_dim_values_ties(keepdim):
    """Integer inputs collide often.  ATen does not pin which of several equal
    elements the index refers to, so only the reported values are compared
    exactly; the index is checked to select the value it claims."""
    inp = torch.randint(0, 6, (5, 9), device=flag_gems.device).to(torch.int32)
    _call_median_dim_values(inp, 1, keepdim, exact_indices=False)


@pytest.mark.median_dim_values
@pytest.mark.parametrize("keepdim", KEEPDIM)
def test_median_dim_values_strided_nonlast_reduction(keepdim):
    """Reducing dim 0 of a (384, 7) tensor exercises the strided input path,
    where the reduction axis is not the innermost one."""
    inp = _unique_input((384, 7), 0, torch.float32)
    _call_median_dim_values(inp, 0, keepdim)


@pytest.mark.median_dim_values
def test_median_dim_values_out_wrong_device():
    if torch.device(flag_gems.device).type != "cuda":
        pytest.skip("mixed-device out= only applies when the accelerator is not cpu")

    inp = torch.randn((7, 5), dtype=torch.float32, device=flag_gems.device)
    values = torch.empty((7,), dtype=inp.dtype, device="cpu")
    indices = torch.empty((7,), dtype=torch.int64, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems.median_dim_values(inp, dim=1, values=values, indices=indices)


@pytest.mark.median_dim_values
def test_median_dim_values_out_rejects_bad_index_dtype():
    inp = torch.randn((7, 5), dtype=torch.float32, device=flag_gems.device)
    values = torch.empty((7,), dtype=inp.dtype, device=flag_gems.device)
    indices = torch.empty((7,), dtype=torch.int32, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems.median_dim_values(inp, dim=1, values=values, indices=indices)
