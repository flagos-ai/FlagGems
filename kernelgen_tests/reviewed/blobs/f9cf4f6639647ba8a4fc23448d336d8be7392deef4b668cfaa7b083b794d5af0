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

# torch.special.bessel_j0 supports float32 and float64. On the Kunlunxin XPU
# backend a float64 request is silently downgraded to float32, so it can never
# match a true-fp64 reference; explicitly mark that backend’s float64 cases.
_SKIP_FP64_ON_KUNLUNXIN = pytest.mark.skipif(
    flag_gems.runtime.device.vendor_name == "kunlunxin",
    reason="Kunlunxin XPU has no real float64 (silently downgraded to float32)",
)
FLOAT_DTYPES = [
    torch.float32,
    pytest.param(torch.float64, marks=_SKIP_FP64_ON_KUNLUNXIN),
]

# Pointwise shapes covering small, medium, and batched 2D tensors
POINTWISE_SHAPES = [(128,), (512, 256), (2, 128, 128)]


@pytest.mark.special_bessel_j0
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_special_bessel_j0(shape, dtype):
    if dtype == torch.float64 and not utils.fp64_is_supported:
        pytest.skip("backend declares no fp64 support; float64 unavailable")
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref_out = torch.special.bessel_j0(ref_inp)
    res_out = flag_gems.special_bessel_j0(inp)
    assert res_out.device == inp.device
    assert res_out.dtype == inp.dtype
    assert res_out.shape == inp.shape
    utils.gems_assert_close(res_out, ref_out, dtype)
