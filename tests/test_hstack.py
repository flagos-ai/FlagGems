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
    HSTACK_SHAPES = [
        [(8,), (16,)],
    ]
    HSTACK_EXCEPTION_SHAPES = [
        [(16, 256), (16,)],
    ]
else:
    HSTACK_SHAPES = [
        [(8,), (16,)],
        [(16, 256), (16, 128)],
        [(20, 320, 15), (20, 160, 15), (20, 80, 15)],
    ]
    HSTACK_EXCEPTION_SHAPES = [
        [(16, 256), (16,)],
        [(16, 256), (8, 128)],
    ]


@pytest.mark.hstack
@pytest.mark.parametrize("shape", HSTACK_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES)
def test_accuracy_hstack(shape, dtype):
    if dtype in utils.FLOAT_DTYPES:
        inp = [torch.randn(s, dtype=dtype, device=flag_gems.device) for s in shape]
    else:
        inp = [
            torch.randint(low=0, high=0x7FFF, size=s, dtype=dtype, device="cpu").to(
                flag_gems.device
            )
            for s in shape
        ]

    ref_inp = [utils.to_reference(_) for _ in inp]
    ref_out = torch.hstack(ref_inp)

    with flag_gems.use_gems():
        res_out = torch.hstack(inp)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.hstack
@pytest.mark.parametrize("shape", HSTACK_EXCEPTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES)
def test_exception_hstack(shape, dtype):
    if dtype in utils.FLOAT_DTYPES:
        inp = [torch.randn(s, dtype=dtype, device=flag_gems.device) for s in shape]
    else:
        inp = [
            torch.randint(low=0, high=0x7FFF, size=s, dtype=dtype, device="cpu").to(
                flag_gems.device
            )
            for s in shape
        ]

    with pytest.raises(RuntimeError):
        with flag_gems.use_gems():
            _ = torch.hstack(inp)
