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

device = flag_gems.device


@pytest.mark.arange_start
@pytest.mark.arange_start_step
@pytest.mark.parametrize("start", utils.ARANGE_START)
@pytest.mark.parametrize("step", [1] if cfg.QUICK_MODE else [1, 2, 5])
@pytest.mark.parametrize("end", [128] if cfg.QUICK_MODE else [128, 256, 1024])
@pytest.mark.parametrize(
    "dtype",
    (
        [torch.float32]
        if cfg.QUICK_MODE
        else utils.FLOAT_DTYPES + utils.ALL_INT_DTYPES + [None]
    ),
)
@pytest.mark.parametrize(
    "device", [flag_gems.device] if cfg.QUICK_MODE else [flag_gems.device, None]
)
# Since triton only target to GPU, pin_memory only used in CPU tensors.
@pytest.mark.parametrize("pin_memory", [False] if cfg.QUICK_MODE else [False, None])
def test_arange(start, step, end, dtype, device, pin_memory):
    # flag_gems.arange_start resolves device=None to the FlagGems device, while
    # ATen's default for device=None is CPU. Compute the reference on the same
    # device the wrapper targets so the comparison is device-consistent (in
    # TO_CPU mode both sides stay on CPU).
    target_device = "cpu" if cfg.TO_CPU else (device or flag_gems.device)

    ref_out = torch.arange(
        start,
        end,
        step,
        dtype=dtype,
        device=target_device,
        pin_memory=pin_memory,
    )

    res_out = flag_gems.arange_start(
        start, end, step, dtype=dtype, device=device, pin_memory=pin_memory
    )

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.arange_start
@pytest.mark.arange_start_step
@pytest.mark.parametrize(
    "start,end,step,dtype",
    [
        (1 / 70, 1.0, 1 / 70, None),
        (-0.5, 0.5, 0.125, None),
        (0.0, 3.0, 1, None),
        (2, 11, 3, None),
        (2, 11, 3.0, None),
        (0, 0, 1, None),
        (11, 2, -3, None),
        (0.0, 1000 / 70, 1 / 70, None),
        (0.1, 5.9, 1.0, torch.int64),
    ],
)
def test_arange_start_dtype_inference(start, end, step, dtype):
    target_device = "cpu" if cfg.TO_CPU else flag_gems.device
    ref_out = torch.arange(start, end, step, dtype=dtype, device=target_device)
    res_out = flag_gems.arange_start(
        start, end, step, dtype=dtype, device=flag_gems.device
    )
    assert res_out.dtype == ref_out.dtype
    assert res_out.shape == ref_out.shape
    if cfg.TO_CPU and ref_out.dtype.is_floating_point:
        utils.gems_assert_close(
            res_out,
            ref_out,
            ref_out.dtype,
            atol=torch.finfo(ref_out.dtype).eps,
        )
    else:
        utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.arange_start
@pytest.mark.arange_start_step
def test_arange_start_float_args_follow_default_dtype():
    start, end, step = 1 / 70, 1.0, 1 / 70
    target_device = "cpu" if cfg.TO_CPU else flag_gems.device
    ref_out = torch.arange(start, end, step, device=target_device)
    res_out = flag_gems.arange_start(start, end, step, device=flag_gems.device)
    assert torch.get_default_dtype() is torch.float32
    assert res_out.dtype == torch.float32
    assert res_out.dtype == ref_out.dtype
    assert res_out.shape == ref_out.shape
    if cfg.TO_CPU:
        utils.gems_assert_close(
            res_out,
            ref_out,
            torch.float32,
            atol=torch.finfo(torch.float32).eps,
        )
    else:
        utils.gems_assert_equal(res_out, ref_out)
