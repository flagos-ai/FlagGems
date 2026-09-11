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

# ARCTANH operator test

import os
import sys

import pytest
import torch

import flag_gems
from flag_gems.experimental_ops.arctanh import arctanh as gems_arctanh
from flag_gems.experimental_ops.arctanh import arctanh_out as gems_arctanh_out

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
try:
    from tests.accuracy_utils import TO_CPU, gems_assert_close  # noqa: E402
except ImportError:
    # Fallback values when running outside pytest
    TO_CPU = False  # fallback

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


def to_reference(inp, upcast=False):
    if inp is None:
        return None
    if TO_CPU:
        ref_inp = inp.to("cpu")
    else:
        ref_inp = inp.clone()
    if upcast:
        if ref_inp.is_complex():
            ref_inp = ref_inp.to(torch.complex128)
        else:
            ref_inp = ref_inp.to(torch.float64)
    return ref_inp


@pytest.mark.arctanh
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_arctanh_tensor(shape, dtype):
    x = (
        torch.rand(shape, device=flag_gems.device, dtype=torch.float32) * 1.8 - 0.9
    ).to(dtype)
    ref_x = to_reference(x)

    ref_out = torch.ops.aten.arctanh(ref_x)

    with flag_gems.use_gems():
        act_out = gems_arctanh(x)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.arctanh
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_arctanh_out(shape, dtype):
    x = (
        torch.rand(shape, device=flag_gems.device, dtype=torch.float32) * 1.8 - 0.9
    ).to(dtype)
    ref_x = to_reference(x)
    act_x = x.clone()

    ref_out_buf = torch.empty_like(ref_x)
    torch.ops.aten.arctanh.out(ref_x, out=ref_out_buf)

    act_out_buf = torch.empty_like(act_x)
    with flag_gems.use_gems():
        gems_arctanh_out(act_x, act_out_buf)

    gems_assert_close(act_out_buf, ref_out_buf, dtype=dtype)
