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
    FFT_SHAPES = [
        (128, 64),
        (512, 512),
    ]
else:
    FFT_SHAPES = [
        (128, 64),
        (128, 128),
        (128, 256),
        (128, 512),
        (128, 1024),
        (256, 256),
        (512, 512),
        (1024, 1024),
        (4096, 256),
    ]


@pytest.mark.fft
@pytest.mark.parametrize("shape", FFT_SHAPES)
def test_fft(shape):
    m, n = shape
    real = torch.randn((m, n), device=flag_gems.device, dtype=torch.float32)
    imag = torch.randn((m, n), device=flag_gems.device, dtype=torch.float32)
    x = torch.complex(real, imag)

    ref_x = utils.to_reference(x)
    ref_out = torch.fft.fft(ref_x)

    with flag_gems.use_gems():
        res_out = torch.fft.fft(ref_x)

    utils.gems_assert_close(res_out, ref_out, torch.complex64, reduce_dim=n)
