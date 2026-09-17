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

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    DIM_LIST = [0]
    KEEPDIM = [True]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    DIM_LIST = [0, 1]
    KEEPDIM = [True, False]


def _assert_mode_matches(inp, dim, keepdim):
    normalized_dim = dim % inp.ndim
    ref_inp = inp.cpu()
    ref_out_value, _ = torch.mode(ref_inp, dim=dim, keepdim=keepdim)
    res_out_value, res_out_index = flag_gems.mode(inp, dim=dim, keepdim=keepdim)

    assert res_out_value.dtype == inp.dtype
    assert res_out_index.dtype == torch.int64
    assert res_out_value.shape == ref_out_value.shape
    assert res_out_index.shape == ref_out_value.shape
    assert torch.all(
        (res_out_index.cpu() >= 0) & (res_out_index.cpu() < inp.shape[dim])
    )
    utils.gems_assert_equal(res_out_value.cpu(), ref_out_value)
    gather_idx = res_out_index.cpu().reshape(
        list(ref_inp.shape[:normalized_dim])
        + [1]
        + list(ref_inp.shape[normalized_dim + 1 :])
    )
    values_at_index = ref_inp.gather(normalized_dim, gather_idx).reshape(
        res_out_index.shape
    )
    utils.gems_assert_equal(values_at_index, ref_out_value)


@pytest.mark.mode
@pytest.mark.parametrize("shape", utils.REDUCTION_SMALL_SHAPES)
@pytest.mark.parametrize("keepdim", KEEPDIM)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize(
    "dtype", FLOAT_DTYPES + utils.ALL_INT_DTYPES + [torch.int8, torch.uint8]
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_mode(shape, dim, keepdim, dtype):
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        low, high = (
            (0, 256)
            if dtype == torch.uint8
            else ((-128, 128) if dtype == torch.int8 else (-100, 100))
        )
        inp = torch.randint(low, high, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )

    _assert_mode_matches(inp, dim, keepdim)


@pytest.mark.mode
@pytest.mark.skipif(
    flag_gems.vendor_name == "cambricon", reason="Issue #5254: Not supported"
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.int8, torch.uint8])
@pytest.mark.parametrize(
    "data, dim, keepdim",
    [
        (
            [
                [1, 2, 2, 3],
                [4, 4, 4, 5],
                [6, 7, 7, 7],
            ],
            1,
            False,
        ),
        (
            [
                [1, 5, 2],
                [1, 5, 3],
                [4, 6, 3],
                [1, 7, 3],
            ],
            0,
            True,
        ),
    ],
)
def test_mode_repeated_values(data, dim, keepdim, dtype):
    inp = torch.tensor(data, dtype=dtype, device=flag_gems.device)

    _assert_mode_matches(inp, dim, keepdim)


@pytest.mark.mode
@pytest.mark.parametrize("dtype", [torch.int8, torch.uint8])
@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("case", ["unique", "tie", "equal"])
@pytest.mark.parametrize("repeat_factor", [1, 3, 103, 1639])
def test_mode_byte_boundaries(dtype, dim, keepdim, case, repeat_factor):
    low, high = torch.iinfo(dtype).min, torch.iinfo(dtype).max
    middle = -1 if dtype == torch.int8 else 128
    if case == "unique":
        data = [[low, high, high, middle, high], [middle, low, low, high, low]]
    elif case == "tie":
        data = [[high, low, high, low, middle], [middle, high, middle, high, low]]
    else:
        data = [[low] * 5, [high] * 5]
    # Preserve the unique/tied frequencies while crossing reduction block sizes.
    inp = torch.tensor(data, dtype=dtype).repeat(1, repeat_factor).to(flag_gems.device)
    if dim == 0:
        inp = inp.t()
    _assert_mode_matches(inp, dim, keepdim)
