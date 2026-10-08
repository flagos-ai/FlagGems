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

import warnings

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
    dim = dim % inp.ndim
    idx = indices if keepdim else indices.unsqueeze(dim)
    picked = torch.gather(inp, dim, idx)
    return picked if keepdim else picked.squeeze(dim)


def _call_median_dim(inp, dim, keepdim, equal_nan=False, exact_indices=True):
    """Reference against ATen; compare indices only when they are well defined."""
    ref_inp = utils.to_reference(inp)
    ref = torch.median(ref_inp, dim=dim, keepdim=keepdim)

    result = flag_gems.median_dim(inp, dim=dim, keepdim=keepdim)

    utils.gems_assert_equal(result.values, ref.values, equal_nan=equal_nan)
    assert tuple(result.values.shape) == tuple(ref.values.shape)
    assert tuple(result.indices.shape) == tuple(ref.indices.shape)
    # Whatever index comes back has to point at the median it reports.  Both
    # sides live on the accelerator, so move them to the reference device
    # first: gems_assert_equal expects `ref` to be there when --ref=cpu is used.
    utils.gems_assert_equal(
        utils.to_reference(_selected_along_dim(inp, dim, result.indices, keepdim)),
        utils.to_reference(result.values),
        equal_nan=equal_nan,
    )
    if exact_indices:
        utils.gems_assert_equal(result.indices, ref.indices)
    return result


@pytest.mark.median_dim
@pytest.mark.parametrize("keepdim", KEEPDIM)
@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.ALL_INT_DTYPES)
def test_median_dim(dim, keepdim, dtype):
    inp = _unique_input((9, 7), dim, dtype)
    _call_median_dim(inp, dim, keepdim)


@pytest.mark.median_dim
@pytest.mark.parametrize("keepdim", KEEPDIM)
@pytest.mark.parametrize("dtype", utils.ALL_INT_DTYPES)
def test_median_dim_keepdim_shapes(keepdim, dtype):
    inp = _unique_input((9, 7), 1, dtype)
    result = _call_median_dim(inp, 1, keepdim)

    expected = (9, 1) if keepdim else (9,)
    assert tuple(result.values.shape) == expected
    assert tuple(result.indices.shape) == expected


@pytest.mark.median_dim
@pytest.mark.parametrize("keepdim", KEEPDIM)
def test_median_dim_ties(keepdim):
    """Integer inputs collide often.  ATen does not pin which of several equal
    elements the index refers to, so only the reported values are compared
    exactly; the index is checked to select the value it claims."""
    inp = torch.randint(0, 6, (5, 9), device=flag_gems.device).to(torch.int32)
    _call_median_dim(inp, 1, keepdim, exact_indices=False)


@pytest.mark.median_dim
@pytest.mark.parametrize("keepdim", KEEPDIM)
def test_median_dim_strided_nonlast_reduction(keepdim):
    """Reducing dim 0 of a (384, 7) tensor exercises the strided input path,
    where the reduction axis is not the innermost one."""
    inp = _unique_input((384, 7), 0, torch.float32)
    _call_median_dim(inp, 0, keepdim)


@pytest.mark.median_dim
@pytest.mark.parametrize(
    "values, expected",
    [
        ([1.0, 2.0, 3.0, 4.0], (2.0, 1)),
        ([4.0, 3.0, 2.0, 1.0], (2.0, 2)),
        ([2.0, 2.0, 3.0, 4.0], (2.0, 0)),
        ([1.0, 2.0, 2.0, 2.0], (2.0, 1)),
        ([5.0, 5.0, 5.0, 5.0], (5.0, 0)),
    ],
)
def test_median_dim_even_count_picks_lower_middle(values, expected):
    inp = torch.tensor(values, dtype=torch.float32, device=flag_gems.device)

    result = flag_gems.median_dim(inp, dim=0)

    value, index = expected
    utils.gems_assert_equal(
        result.values, utils.to_reference(torch.full_like(result.values, value))
    )
    utils.gems_assert_equal(
        result.indices, utils.to_reference(torch.full_like(result.indices, index))
    )


@pytest.mark.median_dim
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_median_dim_propagates_nan(dtype):
    inp = torch.tensor([1.0, float("nan"), 3.0], dtype=dtype, device=flag_gems.device)
    _call_median_dim(inp, 0, False, equal_nan=True)


@pytest.mark.median_dim
def test_median_dim_scalar_input():
    inp = torch.tensor(3.0, dtype=torch.float32, device=flag_gems.device)

    result = flag_gems.median_dim(inp, dim=0)

    assert result.values.ndim == 0 and result.indices.ndim == 0
    utils.gems_assert_equal(
        result.values, utils.to_reference(torch.full_like(result.values, 3.0))
    )
    utils.gems_assert_equal(
        result.indices, utils.to_reference(torch.zeros_like(result.indices))
    )


@pytest.mark.median_dim
@pytest.mark.parametrize("keepdim", KEEPDIM)
def test_median_dim_empty_reduction_dim_raises(keepdim):
    """ATen rejects a reduction dim of size 0 rather than returning empty."""
    inp = torch.empty((0, 4), dtype=torch.float32, device=flag_gems.device)

    with pytest.raises(IndexError):
        flag_gems.median_dim(inp, dim=0, keepdim=keepdim)


@pytest.mark.median_dim
@pytest.mark.parametrize("shape, dim", [((3, 0, 5), 0), ((3, 0, 5), 2), ((0, 4), 1)])
def test_median_dim_empty_but_nonzero_reduction(shape, dim):
    """Zero elements overall, but the reduced axis is not the empty one."""
    inp = torch.empty(shape, dtype=torch.float32, device=flag_gems.device)

    result = flag_gems.median_dim(inp, dim=dim)

    ref = torch.median(utils.to_reference(inp), dim=dim)
    utils.gems_assert_equal(result.values, ref.values)
    utils.gems_assert_equal(result.indices, ref.indices)


@pytest.mark.median_dim
@pytest.mark.parametrize("keepdim", KEEPDIM)
def test_median_dim_named_dim_preserves_names(keepdim):
    """``dim`` may be given as a name, and the surviving names are kept."""
    inp = torch.randn((32, 7), dtype=torch.float32, device=flag_gems.device)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Named tensors and all their associated APIs are an experimental feature",
            category=UserWarning,
        )
        inp = inp.refine_names("reduce", "feature")

    ref = torch.median(utils.to_reference(inp.rename(None)), dim=0, keepdim=keepdim)
    result = flag_gems.median_dim(inp, dim="reduce", keepdim=keepdim)

    expected_names = ("reduce", "feature") if keepdim else ("feature",)
    assert result.values.names == expected_names
    assert result.indices.names == expected_names
    utils.gems_assert_equal(result.values.rename(None), ref.values)
    utils.gems_assert_equal(result.indices.rename(None), ref.indices)
