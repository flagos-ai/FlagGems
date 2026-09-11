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
from flag_gems.runtime import torch_device_fn


@pytest.mark.assert_async
@pytest.mark.parametrize(
    "shape, value, expected_err, match_str",
    [
        ((), 1.0, None, None),
        ((2,), 1.0, RuntimeError, "is ambiguous"),
        ((1,), 1.0, None, None),
    ],
)
def test_assert_async(shape, value, expected_err, match_str):
    msg = "Assertion failed!"
    inp_pt = torch.full(shape, value, device=flag_gems.device)
    inp_triton = inp_pt.clone()
    if expected_err:
        with flag_gems.use_gems():
            with pytest.raises(expected_err, match=match_str):
                flag_gems._assert_async(inp_triton, msg)
                if value == 0:
                    torch_device_fn.synchronize()
    else:
        with flag_gems.use_gems():
            flag_gems._assert_async(inp_triton, msg)
            torch_device_fn.synchronize()

    if flag_gems.device == "cuda":
        if expected_err:
            with pytest.raises(expected_err, match=match_str):
                torch._assert_async(inp_pt, msg)
                if value == 0:
                    torch_device_fn.synchronize()
        else:
            torch._assert_async(inp_pt, msg)
            torch_device_fn.synchronize()
