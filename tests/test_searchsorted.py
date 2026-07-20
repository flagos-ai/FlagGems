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

SEARCHSORTED_DTYPES = list(
    dict.fromkeys(
        utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + [torch.int8, torch.uint8]
    )
)
if cfg.QUICK_MODE:
    SIDE_CASES = [(False, None), (False, "right")]
else:
    SIDE_CASES = [(False, None), (True, None), (False, "left"), (False, "right")]

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name == "ascend" and not utils.TO_CPU,
    reason="Ascend native torch.searchsorted reference has dtype and side/right semantic gaps; run with --ref cpu.",
)


def _tensor(data, dtype, device):
    if dtype.is_floating_point:
        return torch.tensor(data, dtype=dtype, device=device)
    return torch.tensor(data, dtype=torch.int64, device="cpu").to(dtype).to(device)


def _make_case(case_name, dtype, device):
    if case_name == "1d":
        sorted_sequence = _tensor([0, 2, 2, 5, 9], dtype, device)
        values = _tensor([[0, 1, 2], [3, 9, 10]], dtype, device)
    elif case_name == "2d":
        sorted_sequence = _tensor([[0, 2, 2, 5, 9], [1, 3, 4, 4, 8]], dtype, device)
        values = _tensor([[0, 2, 6], [1, 4, 9]], dtype, device)
    elif case_name == "3d":
        sorted_sequence = _tensor(
            [
                [[0, 2, 2, 5], [1, 3, 4, 8]],
                [[2, 4, 6, 6], [0, 1, 7, 9]],
            ],
            dtype,
            device,
        )
        values = _tensor(
            [
                [[0, 2, 6], [2, 4, 9]],
                [[1, 5, 7], [0, 8, 10]],
            ],
            dtype,
            device,
        )
    elif case_name == "empty_values":
        sorted_sequence = _tensor([[0, 2, 4], [1, 3, 5]], dtype, device)
        values = torch.empty((2, 0), dtype=dtype, device=device)
    elif case_name == "empty_boundaries":
        sorted_sequence = torch.empty((2, 0), dtype=dtype, device=device)
        values = _tensor([[0, 2], [1, 3]], dtype, device)
    else:
        raise AssertionError(f"unknown case {case_name}")
    return sorted_sequence, values


@pytest.mark.searchsorted
@pytest.mark.parametrize(
    "case_name", ["1d", "2d", "3d", "empty_values", "empty_boundaries"]
)
@pytest.mark.parametrize("right, side", SIDE_CASES)
@pytest.mark.parametrize("out_int32", [False, True])
@pytest.mark.parametrize("dtype", SEARCHSORTED_DTYPES)
def test_searchsorted_tensor(case_name, right, side, out_int32, dtype):
    sorted_sequence, values = _make_case(case_name, dtype, flag_gems.device)
    ref_sorted_sequence = utils.to_reference(sorted_sequence)
    ref_values = utils.to_reference(values)
    kwargs = {"out_int32": out_int32, "right": right, "side": side}
    ref = torch.searchsorted(ref_sorted_sequence, ref_values, **kwargs)

    with flag_gems.use_gems():
        res = torch.searchsorted(sorted_sequence, values, **kwargs)

    assert res.dtype == (torch.int32 if out_int32 else torch.int64)
    utils.gems_assert_equal(res, ref)


@pytest.mark.searchsorted
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("right", [False, True])
def test_searchsorted_tensor_nan_inf(dtype, right):
    sorted_sequence = _tensor([0.0, 2.0, 5.0], dtype, flag_gems.device)
    values = torch.tensor(
        [float("-inf"), 1.0, float("nan"), float("inf")],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref = torch.searchsorted(
        utils.to_reference(sorted_sequence), utils.to_reference(values), right=right
    )

    with flag_gems.use_gems():
        res = torch.searchsorted(sorted_sequence, values, right=right)

    utils.gems_assert_equal(res, ref)


@pytest.mark.searchsorted
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
@pytest.mark.parametrize("right", [False, True])
def test_searchsorted_tensor_sorter(dtype, right):
    sorted_sequence = _tensor([[4, 1, 3, 2], [10, 5, 7, 6]], dtype, flag_gems.device)
    values = _tensor([[0, 2, 5], [5, 6, 8]], dtype, flag_gems.device)
    sorter = torch.argsort(sorted_sequence, dim=-1)
    ref = torch.searchsorted(
        utils.to_reference(sorted_sequence),
        utils.to_reference(values),
        right=right,
        sorter=utils.to_reference(sorter),
    )

    with flag_gems.use_gems():
        res = torch.searchsorted(sorted_sequence, values, right=right, sorter=sorter)

    utils.gems_assert_equal(res, ref)


@pytest.mark.searchsorted
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
@pytest.mark.parametrize("right", [False, True])
def test_searchsorted_tensor_noncontiguous(dtype, right):
    base = _tensor([[0, 1, 2, 3, 4, 5], [1, 2, 4, 6, 8, 10]], dtype, flag_gems.device)
    sorted_sequence = base[:, ::2]
    values = _tensor([[0, 1], [2, 4], [5, 9]], dtype, flag_gems.device).t()
    ref = torch.searchsorted(
        utils.to_reference(sorted_sequence), utils.to_reference(values), right=right
    )

    with flag_gems.use_gems():
        res = torch.searchsorted(sorted_sequence, values, right=right)

    utils.gems_assert_equal(res, ref)


@pytest.mark.searchsorted_scalar
@pytest.mark.parametrize("right, side", SIDE_CASES)
@pytest.mark.parametrize("out_int32", [False, True])
@pytest.mark.parametrize("dtype", SEARCHSORTED_DTYPES)
def test_searchsorted_scalar(right, side, out_int32, dtype):
    sorted_sequence = _tensor([0, 2, 2, 5, 9], dtype, flag_gems.device)
    value = 2.0 if dtype.is_floating_point else 2
    kwargs = {"out_int32": out_int32, "right": right, "side": side}
    ref = torch.searchsorted(utils.to_reference(sorted_sequence), value, **kwargs)

    with flag_gems.use_gems():
        res = torch.searchsorted(sorted_sequence, value, **kwargs)

    assert res.shape == torch.Size([])
    assert res.dtype == (torch.int32 if out_int32 else torch.int64)
    utils.gems_assert_equal(res, ref)


@pytest.mark.searchsorted_out
@pytest.mark.parametrize("out_int32", [False, True])
@pytest.mark.parametrize("right", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_searchsorted_tensor_out(out_int32, right, dtype):
    sorted_sequence = _tensor([[0, 2, 2, 5], [1, 3, 4, 8]], dtype, flag_gems.device)
    values = _tensor([[0, 2, 6], [2, 4, 9]], dtype, flag_gems.device)
    out_dtype = torch.int32 if out_int32 else torch.int64
    ref_out = torch.empty_strided(
        values.shape,
        (1, values.shape[0]),
        dtype=out_dtype,
        device=utils.to_reference(values).device,
    )
    res_out = torch.empty_strided(
        values.shape, (1, values.shape[0]), dtype=out_dtype, device=flag_gems.device
    )

    ref = torch.ops.aten.searchsorted.Tensor_out(
        utils.to_reference(sorted_sequence),
        utils.to_reference(values),
        out_int32=out_int32,
        right=right,
        out=ref_out,
    )
    with flag_gems.use_gems():
        res = torch.ops.aten.searchsorted.Tensor_out(
            sorted_sequence,
            values,
            out_int32=out_int32,
            right=right,
            out=res_out,
        )

    assert res.data_ptr() == res_out.data_ptr()
    assert res.stride() == res_out.stride()
    utils.gems_assert_equal(res, ref)


@pytest.mark.searchsorted_scalar_out
@pytest.mark.parametrize("out_int32", [False, True])
@pytest.mark.parametrize("right", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_searchsorted_scalar_out(out_int32, right, dtype):
    sorted_sequence = _tensor([0, 2, 2, 5], dtype, flag_gems.device)
    value = 2.0 if dtype.is_floating_point else 2
    out_dtype = torch.int32 if out_int32 else torch.int64
    ref_out = torch.empty(
        (), dtype=out_dtype, device=utils.to_reference(sorted_sequence).device
    )
    res_out = torch.empty((), dtype=out_dtype, device=flag_gems.device)

    ref = torch.ops.aten.searchsorted.Scalar_out(
        utils.to_reference(sorted_sequence),
        value,
        out_int32=out_int32,
        right=right,
        out=ref_out,
    )
    with flag_gems.use_gems():
        res = torch.ops.aten.searchsorted.Scalar_out(
            sorted_sequence,
            value,
            out_int32=out_int32,
            right=right,
            out=res_out,
        )

    assert res.data_ptr() == res_out.data_ptr()
    utils.gems_assert_equal(res, ref)
