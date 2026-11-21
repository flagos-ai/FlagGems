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


@pytest.mark.sort
@pytest.mark.sort_stable
@pytest.mark.skipif(
    flag_gems.vendor_name != "hygon", reason="Hygon empty radix sort regression"
)
@pytest.mark.parametrize("stable", [False, True], ids=["default", "stable"])
@pytest.mark.parametrize(
    "shape,dim",
    [
        pytest.param((0,), -1, id="empty-vector"),
        pytest.param((3, 0, 2), -2, id="empty-sort-axis"),
        pytest.param((0, 3), -1, id="empty-batch"),
    ],
)
def test_sort_hygon_empty(shape, dim, stable):
    inp_cpu = torch.empty(shape, dtype=torch.int64, device="cpu")
    inp = inp_cpu.to(flag_gems.device)
    if stable:
        expected = torch.sort(inp_cpu, dim=dim, stable=True)
        actual = flag_gems.sort_stable(inp, dim=dim, stable=True)
    else:
        expected = torch.sort(inp_cpu, dim=dim)
        actual = flag_gems.sort(inp, dim=dim)

    for result, reference in zip(actual, expected):
        assert result.device == inp.device
        torch.testing.assert_close(result.cpu(), reference, rtol=0, atol=0)


@pytest.mark.sort
@pytest.mark.parametrize("batch_size", [4, 8])
@pytest.mark.parametrize(
    "hiddensize", [1, 256, 2048, 9333, 65536, 32768, 128 * 1024, 256 * 1024]
)
@pytest.mark.parametrize("descending", [True, False])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES)
@pytest.mark.parametrize("dim", [0, -1])
def test_sort(batch_size, hiddensize, descending, dtype, dim):
    if dtype in utils.BOOL_TYPES:
        y = torch.randint(
            0, 2, (batch_size, hiddensize), dtype=dtype, device=flag_gems.device
        )
    elif dtype in utils.ALL_INT_DTYPES:
        min_v, max_v = torch.iinfo(dtype).min, torch.iinfo(dtype).max
        y = torch.randint(
            min_v, max_v, (batch_size, hiddensize), dtype=dtype, device="cpu"
        ).to(flag_gems.device)
    else:
        y = torch.randn((batch_size, hiddensize), dtype=dtype, device=flag_gems.device)

    ref_y = utils.to_reference(y)
    # we only implement stable sort, non-stable sort is undefined
    ref_value, ref_index = torch.sort(
        ref_y, dim=dim, stable=True, descending=descending
    )

    res_value, res_index = flag_gems.sort(y, dim=dim, descending=descending)

    utils.gems_assert_close(res_value, ref_value, dtype)
    utils.gems_assert_equal(res_index, ref_index)


@pytest.mark.sort_stable
@pytest.mark.parametrize("batch_size", [4, 8])
@pytest.mark.parametrize(
    "hiddensize", [1, 256, 2048, 9333, 65536, 32768, 128 * 1024, 256 * 1024]
)
@pytest.mark.parametrize("descending", [True, False])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES)
@pytest.mark.parametrize("dim", [0, -1])
def test_sort_stable(batch_size, hiddensize, descending, dtype, dim):
    if dtype in utils.BOOL_TYPES:
        y = torch.randint(
            0, 2, (batch_size, hiddensize), dtype=dtype, device=flag_gems.device
        )
    elif dtype in utils.ALL_INT_DTYPES:
        min_v, max_v = torch.iinfo(dtype).min, torch.iinfo(dtype).max
        y = torch.randint(
            min_v, max_v, (batch_size, hiddensize), dtype=dtype, device="cpu"
        ).to(flag_gems.device)
    else:
        y = torch.randn((batch_size, hiddensize), dtype=dtype, device=flag_gems.device)

    ref_y = utils.to_reference(y)
    ref_value, ref_index = torch.sort(
        ref_y, dim=dim, stable=True, descending=descending
    )

    res_value, res_index = flag_gems.sort(y, dim=dim, descending=descending)

    utils.gems_assert_close(res_value, ref_value, dtype)
    utils.gems_assert_equal(res_index, ref_index)
