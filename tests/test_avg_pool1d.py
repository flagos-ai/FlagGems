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

import logging

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
    # k=3/s=2/p=1 with total >= 4096 exercises the head + body double launch of
    # the pair-view fast path (smaller cases take the single clamped launch).
    # Both count_include_pad variants are kept so the body kernel's divisor
    # branch is covered, and totals span 8K, 64K and 512K outputs.
    ((2, 2, 4100), 3, 2, 1, False, False),  # total 8200, near the 4096 threshold
    ((16, 64, 128), 3, 2, 1, False, True),  # total 65536
    ((16, 64, 128), 3, 2, 1, False, False),  # total 65536, pad-excluded divisor
    ((128, 512, 16), 3, 2, 1, False, True),  # total 524288
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
    shape, kernel_size, stride, padding, ceil_mode, count_include_pad, dtype, caplog
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

    with caplog.at_level(
        logging.DEBUG, logger=utils.gems_log_logger(flag_gems.avg_pool1d)
    ):
        res_out = flag_gems.avg_pool1d(
            inp,
            kernel_size=[kernel_size],
            stride=[stride],
            padding=[padding],
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
        )

    utils.gems_assert_close(res_out, ref_out, dtype)
    assert f"{utils.gems_log_prefix(flag_gems.avg_pool1d)} AVG_POOL1D" in caplog.text


@pytest.mark.avg_pool1d
@pytest.mark.parametrize(
    "shape, kernel_size, stride, padding, ceil_mode, count_include_pad",
    AVGPOOL1D_CONFIGS,
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_avg_pool1d_default_stride(
    shape, kernel_size, stride, padding, ceil_mode, count_include_pad, dtype, caplog
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

    with caplog.at_level(
        logging.DEBUG, logger=utils.gems_log_logger(flag_gems.avg_pool1d)
    ):
        res_out = flag_gems.avg_pool1d(
            inp,
            kernel_size=[kernel_size],
            padding=[padding],
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
        )

    utils.gems_assert_close(res_out, ref_out, dtype)
    assert f"{utils.gems_log_prefix(flag_gems.avg_pool1d)} AVG_POOL1D" in caplog.text
