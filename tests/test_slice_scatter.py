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
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    REGULAR_DIM_SHAPE_STRIDES = [(1, *utils.CONTIGUOUS_SHAPE_STRIDES_2D[1])]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    REGULAR_DIM_SHAPE_STRIDES = list(
        (random.randint(0, len(shape) - 1), shape, stride)
        for shape, stride in utils.CONTIGUOUS_SHAPE_STRIDES_2D
    )

random.seed(time.time() // 100)


@pytest.mark.slice_scatter
@pytest.mark.parametrize(("dim", "shape", "stride"), REGULAR_DIM_SHAPE_STRIDES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("start", [16, 64])
@pytest.mark.parametrize("end", [1024, 256])
@pytest.mark.parametrize("step", [1, 2])
def test_slice_scatter(shape, stride, dim, dtype, start, end, step):
    inp = torch.empty_strided(shape, stride, dtype=dtype, device=flag_gems.device)
    inp.copy_(1)

    valid_shape = list(inp.shape)
    size = valid_shape[dim]

    start = start % size
    end = end % (size + 1)

    if end < start:
        end, start = start, end
    elif end == start:
        end = size

    valid_shape[dim] = (end - start + step - 1) // step

    src = torch.rand(valid_shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_src = utils.to_reference(src)
    ref_out = torch.slice_scatter(
        ref_inp, dim=dim, src=ref_src, start=start, end=end, step=step
    )

    res_out = flag_gems.slice_scatter(
        inp, dim=dim, src=src, start=start, end=end, step=step
    )

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.slice_scatter
def testslice_scatter_with_self_overlapping_input():
    inp = torch.randn((3, 1), device=flag_gems.device).broadcast_to((3, 8))
    src = torch.rand((3, 4), device=flag_gems.device)

    start = 0
    end = 8
    step = 2
    dim = 1
    ref_inp = utils.to_reference(inp)
    ref_src = utils.to_reference(src)
    ref_out = torch.slice_scatter(
        ref_inp, dim=dim, src=ref_src, start=start, end=end, step=step
    )
    res_out = flag_gems.slice_scatter(
        inp, dim=dim, src=src, start=start, end=end, step=step
    )

    utils.gems_assert_equal(res_out, ref_out)
