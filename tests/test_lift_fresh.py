# Copyright 2026, The FlagOS Contributors.
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
    # Single tiny shape keeps QUICK_MODE fast; mirrors sibling lift_fresh_copy.
    LIFT_FRESH_SHAPES = [(2, 3)]
else:
    # Small-to-medium 2D shapes cover tail masking and multiple blocks; mirrors
    # the sibling lift_fresh_copy test shapes.
    LIFT_FRESH_SHAPES = [(2, 3), (128, 256), (512, 512)]


@pytest.mark.lift_fresh
@pytest.mark.parametrize("shape", LIFT_FRESH_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_lift_fresh(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.ops.aten.lift_fresh(ref_inp)
    res_out = flag_gems.lift_fresh(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.lift_fresh
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend alias semantics")
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.float64,
        torch.bfloat16,
        torch.int64,
        torch.bool,
        torch.complex64,
    ],
)
@pytest.mark.parametrize("layout", ["contiguous", "strided", "empty", "scalar"])
def test_ascend_lift_fresh_alias(dtype, layout):
    cpu = torch.arange(24).reshape(4, 6).to(dtype)
    inp = cpu.to(flag_gems.device)
    if layout == "strided":
        inp = inp[1:, 1::2].transpose(0, 1)
    elif layout == "empty":
        inp = inp[1:1, 1::2]
    elif layout == "scalar":
        inp = inp[1, 2]
    expected = inp.cpu().clone()

    actual = flag_gems.lift_fresh(self=inp)

    assert torch._C._is_alias_of(actual, inp)
    assert actual.shape == inp.shape
    assert actual.stride() == inp.stride()
    assert actual.storage_offset() == inp.storage_offset()
    utils.gems_assert_equal(actual.cpu(), expected)
    if inp.numel():
        inp.fill_(1)
        utils.gems_assert_equal(actual.cpu(), torch.ones_like(expected))


@pytest.mark.lift_fresh
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend launch limit")
def test_ascend_lift_fresh_large_alias():
    # A copying kernel with BLOCK=1024 would exceed the 65535-program limit.
    inp = torch.empty(80_000_000, dtype=torch.float32, device=flag_gems.device)
    actual = flag_gems.lift_fresh(inp)
    assert torch._C._is_alias_of(actual, inp)
    assert actual.shape == inp.shape
