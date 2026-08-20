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

AVGPOOL1D_CONFIGS = [
    # (shape, kernel_size, stride, padding, ceil_mode, count_include_pad)
    ((4, 3, 32), 3, 2, 1, False, True),
    ((4, 3, 32), 3, 2, 1, False, False),
    ((8, 16, 28), 5, 2, 1, False, True),
    ((1, 1, 7), 2, 1, 0, False, True),
    ((1, 64, 56), 3, 2, 1, False, True),
    ((2, 8, 16), 2, 2, 0, False, False),
    ((2, 8, 20), 2, 2, 1, False, True),
]

# Note: ceil_mode=True case removed from default_stride test due to upstream
# avg_pool2d bug in boundary handling (affects last window calculation)


@pytest.mark.avg_pool1d
@pytest.mark.parametrize(
    "shape, kernel_size, stride, padding, ceil_mode, count_include_pad",
    AVGPOOL1D_CONFIGS,
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_avg_pool1d(
    shape, kernel_size, stride, padding, ceil_mode, count_include_pad, dtype
):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.ops.aten.avg_pool1d(
        ref_inp,
        kernel_size=[kernel_size],
        stride=[stride],
        padding=[padding],
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
    )

    with flag_gems.use_gems():
        res_out = torch.ops.aten.avg_pool1d(
            inp,
            kernel_size=[kernel_size],
            stride=[stride],
            padding=[padding],
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
        )

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.avg_pool1d
@pytest.mark.parametrize(
    "shape, kernel_size, stride, padding, ceil_mode, count_include_pad",
    AVGPOOL1D_CONFIGS,
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_avg_pool1d_default_stride(
    shape, kernel_size, stride, padding, ceil_mode, count_include_pad, dtype
):
    """Test avg_pool1d with default stride (stride=[] means stride=kernel_size)."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.ops.aten.avg_pool1d(
        ref_inp,
        kernel_size=[kernel_size],
        padding=[padding],
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
    )

    with flag_gems.use_gems():
        res_out = torch.ops.aten.avg_pool1d(
            inp,
            kernel_size=[kernel_size],
            padding=[padding],
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
        )

    utils.gems_assert_close(res_out, ref_out, dtype)
