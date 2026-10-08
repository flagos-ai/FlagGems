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
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES

MAXPOOL1D_CONFIGS = [
    # (shape, kernel_size, stride, padding, dilation, ceil_mode)
    ((4, 3, 32), 3, 2, 1, 1, False),
    ((8, 16, 28), 5, 2, 1, 1, False),
    ((1, 1, 7), 2, 1, 0, 1, False),
    ((1, 64, 56), 3, 2, 1, 1, False),
    ((2, 8, 16), 2, 2, 0, 1, False),
    ((2, 8, 20), 2, 2, 1, 1, False),
    ((4, 3, 32), 3, 2, 1, 2, False),
    ((4, 3, 33), 3, 2, 0, 1, True),
    ((3, 5, 100), 4, 4, 0, 1, False),
]


@pytest.mark.max_pool1d
@pytest.mark.parametrize(
    "shape, kernel_size, stride, padding, dilation, ceil_mode",
    MAXPOOL1D_CONFIGS,
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_max_pool1d(shape, kernel_size, stride, padding, dilation, ceil_mode, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.max_pool1d(
        ref_inp,
        kernel_size=[kernel_size],
        stride=[stride],
        padding=[padding],
        dilation=[dilation],
        ceil_mode=ceil_mode,
    )

    res_out = flag_gems.max_pool1d(
        inp,
        kernel_size=[kernel_size],
        stride=[stride],
        padding=[padding],
        dilation=[dilation],
        ceil_mode=ceil_mode,
    )

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.max_pool1d
@pytest.mark.parametrize(
    "shape, kernel_size, stride, padding, dilation, ceil_mode",
    MAXPOOL1D_CONFIGS,
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_max_pool1d_default_stride(
    shape, kernel_size, stride, padding, dilation, ceil_mode, dtype
):
    """Test max_pool1d with default stride (stride=[] means stride=kernel_size)."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.max_pool1d(
        ref_inp,
        kernel_size=[kernel_size],
        padding=[padding],
        dilation=[dilation],
        ceil_mode=ceil_mode,
    )

    res_out = flag_gems.max_pool1d(
        inp,
        kernel_size=[kernel_size],
        padding=[padding],
        dilation=[dilation],
        ceil_mode=ceil_mode,
    )

    utils.gems_assert_equal(res_out, ref_out)
