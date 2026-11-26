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


@pytest.mark.isinf
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_isinf(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp = torch.masked_fill(inp, inp > 1.0, -float("inf"))
    ref_inp = utils.to_reference(inp)

    ref_out = torch.isinf(ref_inp)
    res_out = flag_gems.isinf(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.isinf
@pytest.mark.skipif(not utils.fp64_is_supported, reason="FP64 is unsupported")
@pytest.mark.parametrize("layout", ["contiguous", "transpose", "slice", "scalar", "empty"])
def test_isinf_fp64_finite(layout):
    fp32_max = torch.finfo(torch.float32).max
    fp64_max = torch.finfo(torch.float64).max
    inp = torch.tensor(
        [
            fp32_max,
            -fp32_max,
            1e100,
            -1e100,
            fp64_max,
            -fp64_max,
            float("inf"),
            -float("inf"),
            float("nan"),
            0.0,
            1.0,
            -1.0,
        ],
        dtype=torch.float64,
    )
    device_inp = inp.to(flag_gems.device)
    if layout == "transpose":
        inp = inp.reshape(3, 4).t()
        device_inp = device_inp.reshape(3, 4).t()
    elif layout == "slice":
        inp = inp.repeat_interleave(2)[::2]
        device_inp = device_inp.repeat_interleave(2)[::2]
    elif layout == "scalar":
        inp = inp[2]
        device_inp = device_inp[2]
    elif layout == "empty":
        inp = inp[:0]
        device_inp = device_inp[:0]
    expected = torch.isinf(inp)
    actual = flag_gems.isinf(device_inp)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
