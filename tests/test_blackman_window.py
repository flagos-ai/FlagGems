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


@pytest.mark.blackman_window
@pytest.mark.parametrize("window_length", [1, 16, 256, 4096])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_blackman_window(window_length, dtype):
    # periodic defaults to True; exercises the aten::blackman_window overload.
    device = "cpu" if cfg.TO_CPU else flag_gems.device
    ref_out = torch.blackman_window(
        window_length,
        dtype=dtype,
        device=device,
    )
    with flag_gems.use_gems():
        res_out = torch.blackman_window(
            window_length,
            dtype=dtype,
            device=device,
        )

    utils.gems_assert_close(res_out, ref_out, dtype=dtype)


@pytest.mark.blackman_window_periodic
@pytest.mark.parametrize("window_length", [0, 1, 2, 3, 16, 200, 512, 4096])
@pytest.mark.parametrize("periodic", [True, False])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_blackman_window_periodic(window_length, periodic, dtype):
    # Passing an explicit periodic flag exercises the
    # aten::blackman_window.periodic overload.
    device = "cpu" if cfg.TO_CPU else flag_gems.device
    ref_out = torch.blackman_window(
        window_length,
        periodic=periodic,
        dtype=dtype,
        device=device,
    )
    with flag_gems.use_gems():
        res_out = torch.blackman_window(
            window_length,
            periodic=periodic,
            dtype=dtype,
            device=device,
        )

    utils.gems_assert_close(res_out, ref_out, dtype=dtype)
