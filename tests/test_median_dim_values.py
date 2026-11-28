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
    if flag_gems.vendor_name == "kunlunxin" and inp.dtype in (
        torch.int16,
        torch.uint8,
    ):
        # KLX int16/uint8 gather is unavailable; this is only a post-candidate
        # selected-value observation, not a replacement median operation.
        assert indices.device == inp.device
        assert indices.dtype == torch.int64
        host_inp, host_idx = inp.detach().cpu(), idx.detach().cpu()
        assert host_inp.dtype == inp.dtype and host_inp.shape == inp.shape
        assert host_idx.dtype == idx.dtype and host_idx.shape == idx.shape
        host_picked = torch.gather(host_inp, dim, host_idx)
        picked = host_picked.to(inp.device)
        assert picked.dtype == inp.dtype and picked.shape == idx.shape
        assert picked.device == inp.device
    else:
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
@pytest.mark.parametrize(
    "dtype", utils.FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BYTE_INT_DTYPES
)
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


@pytest.mark.median_dim_values
@pytest.mark.skipif(
    flag_gems.vendor_name != "kunlunxin", reason="Kunlunxin validation regression"
)
@pytest.mark.parametrize("dim", [-4, 3])
def test_median_dim_values_invalid_dim(dim):
    inp = torch.empty((2, 3, 4), device=flag_gems.device)
    values = torch.empty((2, 3), device=flag_gems.device)
    indices = torch.empty((2, 3), dtype=torch.int64, device=flag_gems.device)
    with pytest.raises(IndexError):
        flag_gems.median_dim_values(inp, dim, values=values, indices=indices)


@pytest.mark.median_dim_values
@pytest.mark.skipif(
    flag_gems.vendor_name != "kunlunxin", reason="Kunlunxin 64-bit regression"
)
@pytest.mark.parametrize("dim", [0, 1])
def test_median_dim_values_strided_int64(dim):
    data = [
        [2**40, -(2**40), 3],
        [2**40 + 1, -(2**40) - 1, 2],
        [2**40 + 2, -(2**40) - 2, 1],
    ]
    reference = torch.tensor(data, dtype=torch.int64).t()
    inp = torch.tensor(data, dtype=torch.int64, device=flag_gems.device).t()
    expected = torch.median(reference, dim=dim)
    shape = tuple(expected.values.shape)
    values = torch.empty(shape, dtype=torch.int64, device=flag_gems.device)
    indices = torch.empty(shape, dtype=torch.int64, device=flag_gems.device)
    result = flag_gems.median_dim_values(inp, dim, values=values, indices=indices)
    torch.testing.assert_close(result.values.cpu(), expected.values, rtol=0, atol=0)
    torch.testing.assert_close(result.indices.cpu(), expected.indices)


@pytest.mark.median_dim_values
@pytest.mark.skipif(
    flag_gems.vendor_name != "kunlunxin", reason="Kunlunxin FP64 capability"
)
def test_median_dim_values_rejects_float64_before_launch():
    # This runtime narrows device FP64 allocations to FP32. A CPU tensor keeps
    # the requested dtype and verifies rejection before any device/kernel use.
    inp = torch.tensor([1.0, 2.0], dtype=torch.float64, device="cpu")
    with pytest.raises(NotImplementedError, match="float64 is not supported"):
        flag_gems.median_dim_values(inp)


@pytest.mark.median_dim_values
@pytest.mark.skipif(
    flag_gems.vendor_name != "kunlunxin", reason="Kunlunxin wrapper regression"
)
def test_median_dim_values_default_dim():
    inp = _unique_input((3, 5), 0, torch.float32)
    expected = torch.median(inp.cpu(), dim=0)
    values = torch.empty((5,), device=flag_gems.device)
    indices = torch.empty((5,), dtype=torch.int64, device=flag_gems.device)
    result = flag_gems.median_dim_values(inp, values=values, indices=indices)
    torch.testing.assert_close(result.values.cpu(), expected.values)
    torch.testing.assert_close(result.indices.cpu(), expected.indices)


@pytest.mark.median_dim_values
@pytest.mark.skipif(
    flag_gems.vendor_name != "kunlunxin", reason="Kunlunxin output-stride regression"
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
def test_median_dim_values_strided_output(dtype):
    inp = _unique_input((3, 5, 7), 1, dtype)
    expected = torch.median(inp.cpu(), dim=1)
    values = torch.empty((7, 3), dtype=dtype, device=flag_gems.device).t()
    indices = torch.empty((7, 3), dtype=torch.int64, device=flag_gems.device).t()
    strides = (values.stride(), indices.stride())
    result = flag_gems.median_dim_values(inp, 1, values=values, indices=indices)
    assert result.values is values and result.indices is indices
    assert (values.stride(), indices.stride()) == strides
    torch.testing.assert_close(values.cpu(), expected.values)
    torch.testing.assert_close(indices.cpu(), expected.indices)


@pytest.mark.median_dim_values
@pytest.mark.skipif(
    flag_gems.vendor_name != "kunlunxin", reason="Kunlunxin wrapper limits"
)
def test_median_dim_values_large_reduction_and_unknown_name():
    inp = torch.empty((4097,), device=flag_gems.device)
    with pytest.raises(NotImplementedError, match="limit 4096"):
        flag_gems.median_dim_values(inp)
    with pytest.raises(RuntimeError, match="Name 'missing' not found"):
        flag_gems.median_dim_values(inp, "missing")
