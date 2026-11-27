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


@pytest.fixture(autouse=True)
def _seed_random_inputs():
    torch.manual_seed(0)


def _assert_kth_indices(
    inp, values, indices, reference_values, reference_indices, dim, keepdim
):
    assert indices.device == inp.device
    assert indices.dtype == torch.int64
    assert indices.shape == reference_indices.shape
    if flag_gems.vendor_name not in ("mthreads", "ascend", "iluvatar"):
        utils.gems_assert_equal(indices, reference_indices)
        return
    # Native kthvalue may choose different valid positions for equal values
    # across devices or repeated calls. Keep exact indices away from ties.
    source = inp.cpu()
    actual_indices = indices.cpu()
    expected_indices = reference_indices.cpu()
    expected_values = reference_values.cpu().to(inp.dtype)
    gather_indices = actual_indices if keepdim else actual_indices.unsqueeze(dim)
    observed = source.gather(dim, gather_indices)
    if not keepdim:
        observed = observed.squeeze(dim)
    utils.gems_assert_equal(observed, expected_values)
    mismatch = actual_indices != expected_indices
    reference_at_dim = expected_values if keepdim else expected_values.unsqueeze(dim)
    ties = (source == reference_at_dim).sum(dim, keepdim=keepdim)
    assert bool((ties[mismatch] > 1).all())
    utils.gems_assert_equal(actual_indices[~mismatch], expected_indices[~mismatch])


# kthvalue tests
KTHVALUE_K_VALUES = [1, 2, 4]


@pytest.mark.kthvalue
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("k", KTHVALUE_K_VALUES)
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("keepdim", [True, False])
# kthvalue implementation relies on topk gemm path which requires float32 precision
@pytest.mark.parametrize("dtype", [torch.float32])
def test_kthvalue(shape, k, dim, keepdim, dtype):
    """Test kthvalue accuracy with float32"""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    dim_size = shape[dim]
    if k < 1 or k > dim_size:
        with pytest.raises(
            RuntimeError,
            match="selected number k out of range|Selected index k out or range",
        ):
            torch.kthvalue(ref_inp, k, dim=dim, keepdim=keepdim)
        with pytest.raises(
            RuntimeError,
            match="selected number k out of range|Selected index k out or range",
        ):
            flag_gems.kthvalue(inp, k, dim=dim, keepdim=keepdim)
        return

    ref_values, ref_indices = torch.kthvalue(ref_inp, k, dim=dim, keepdim=keepdim)
    res_values, res_indices = flag_gems.kthvalue(inp, k, dim=dim, keepdim=keepdim)

    utils.gems_assert_close(res_values, ref_values, dtype)
    _assert_kth_indices(
        inp, res_values, res_indices, ref_values, ref_indices, dim, keepdim
    )


@pytest.mark.kthvalue
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("k", [1, 2])
# kthvalue implementation relies on topk gemm path which requires float32 precision
@pytest.mark.parametrize("dtype", [torch.float32])
def test_kthvalue_default_dim(shape, k, dtype):
    """Test kthvalue with default dim (last dimension)"""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    dim_size = shape[-1]
    if k < 1 or k > dim_size:
        with pytest.raises(
            RuntimeError,
            match="selected number k out of range|Selected index k out or range",
        ):
            torch.kthvalue(ref_inp, k)
        with pytest.raises(
            RuntimeError,
            match="selected number k out of range|Selected index k out or range",
        ):
            flag_gems.kthvalue(inp, k)
        return

    ref_values, ref_indices = torch.kthvalue(ref_inp, k)
    res_values, res_indices = flag_gems.kthvalue(inp, k)

    utils.gems_assert_close(res_values, ref_values, dtype)
    _assert_kth_indices(
        inp, res_values, res_indices, ref_values, ref_indices, -1, False
    )


@pytest.mark.kthvalue
@pytest.mark.parametrize(
    "shape, dim",
    [
        ((3, 5), 2),
        ((3, 5), -3),
        ((4,), 1),
    ],
)
def test_kthvalue_invalid_dim(shape, dim):
    inp = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    native_error = (
        RuntimeError
        if flag_gems.vendor_name == "ascend" and not utils.TO_CPU
        else IndexError
    )
    with pytest.raises(native_error, match="Dimension out of range"):
        torch.kthvalue(ref_inp, 1, dim=dim)
    with pytest.raises(IndexError, match="Dimension out of range"):
        flag_gems.kthvalue(inp, 1, dim=dim)


@pytest.mark.kthvalue
def test_kthvalue_empty_tensor():
    inp = torch.randn(3, 0, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ascend_native = flag_gems.vendor_name == "ascend" and not utils.TO_CPU
    native_error = RuntimeError if ascend_native else IndexError
    native_message = (
        "Selected index k out or range" if ascend_native else "Expected reduction dim"
    )
    with pytest.raises(native_error, match=native_message):
        torch.kthvalue(ref_inp, 1, dim=1)
    with pytest.raises(IndexError, match="Expected reduction dim"):
        flag_gems.kthvalue(inp, 1, dim=1)
