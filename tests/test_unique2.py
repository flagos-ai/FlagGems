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

import random
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

random.seed(time.time() // 100)

device = flag_gems.device


@pytest.mark.unique2
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
@pytest.mark.parametrize("sorted", [True])
@pytest.mark.parametrize("return_inverse", [True, False])
@pytest.mark.parametrize("return_counts", [False, True])
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_unique2(shape, dtype, sorted, return_inverse, return_counts):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    if dtype in utils.FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-10, 10, shape, device=flag_gems.device).to(dtype)

    ref_inp = utils.to_reference(inp, False)

    res_out, res_inverse, res_counts = flag_gems._unique2(
        inp,
        sorted=sorted,
        return_inverse=return_inverse,
        return_counts=return_counts,
    )

    ref = torch.unique(
        ref_inp,
        sorted=sorted,
        return_inverse=return_inverse,
        return_counts=return_counts,
    )
    # torch.unique returns a bare tensor only when both flags are False, else a
    # tuple whose arity depends on the flags; flag_gems._unique2 always returns
    # a (out, inverse, counts) tuple.
    if not return_inverse and not return_counts:
        ref_out, ref_inverse, ref_counts = ref, None, None
    elif return_inverse and return_counts:
        ref_out, ref_inverse, ref_counts = ref
    elif return_inverse:
        ref_out, ref_inverse = ref
        ref_counts = None
    else:
        ref_out, ref_counts = ref
        ref_inverse = None

    assert res_out.numel() == ref_out.numel()

    if return_inverse:
        utils.gems_assert_equal(res_inverse, ref_inverse)
    if return_counts:
        utils.gems_assert_equal(res_counts, ref_counts)

    utils.gems_assert_equal(res_out, ref_out)
