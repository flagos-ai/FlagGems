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

import functools
import random
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

random.seed(time.time() // 100)


@functools.lru_cache(maxsize=None)
def _device_linspace_supports_integral(dtype):
    """Whether torch.linspace on the device handles the given integral dtype.

    Some vendor builds (e.g. xdnn_pytorch_wrapper on Kunlunxin) only implement
    a subset of dtype combinations, so unsupported dtypes fall back to the CPU
    reference.
    """
    try:
        torch.linspace(0, 1, 4, dtype=dtype, device=flag_gems.device)
    except Exception:
        return False
    return True


@pytest.mark.linspace
@pytest.mark.parametrize("start", [0, 2, 4])
@pytest.mark.parametrize("end", [256, 2048, 4096])
@pytest.mark.parametrize("steps", [1, 256, 512])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.ALL_INT_DTYPES + [None])
@pytest.mark.parametrize("device", [flag_gems.device, None])
@pytest.mark.parametrize("pin_memory", [False, None])
def test_linspace(start, end, steps, dtype, device, pin_memory):
    ref_device = "cpu" if cfg.TO_CPU else device
    if (
        flag_gems.vendor_name == "kunlunxin"
        and dtype in (torch.int16, torch.int32, torch.int64)
        and not _device_linspace_supports_integral(dtype)
    ):
        # The vendor's xdnn_pytorch_wrapper only implements a subset of dtype
        # combinations (e.g. not int16), so fall back to the CPU reference.
        ref_device = "cpu"

    ref_out = torch.linspace(
        start,
        end,
        steps,
        dtype=dtype,
        layout=None,
        device=ref_device,
        pin_memory=pin_memory,
    )
    if device is None:
        res_out = torch.linspace(
            start,
            end,
            steps,
            dtype=dtype,
            layout=None,
            device=device,
            pin_memory=pin_memory,
        )
    else:
        res_out = flag_gems.linspace(
            start,
            end,
            steps,
            dtype=dtype,
            layout=None,
            device=device,
            pin_memory=pin_memory,
        )

    if ref_device == "cpu" and res_out.device != torch.device("cpu"):
        res_out = res_out.cpu()

    if dtype in [torch.float16, torch.bfloat16, torch.float32, None]:
        utils.gems_assert_close(res_out, ref_out, dtype=dtype)
    else:
        utils.gems_assert_equal(res_out, ref_out)
