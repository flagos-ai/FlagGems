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
    # QUICK_MODE: only test float32 for faster CI
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES

# Representative tensor shapes covering 1D to 4D cases with varying sizes
SLICE_SHAPES = [
    (128,),
    (256, 128),
    (1024, 1024),
    (512, 1024, 512),
    (16, 8192, 4096),
    (8, 4096, 11008),
    (4, 32, 4096, 128),
    (32, 256, 256, 128),
]


@pytest.mark.slice
@pytest.mark.parametrize("shape", SLICE_SHAPES)
@pytest.mark.parametrize("dim", [0, 1, -1])
@pytest.mark.parametrize("start", [0, 16])
@pytest.mark.parametrize("end", [64, 128])
@pytest.mark.parametrize("step", [1, 2])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_slice_forward(shape, dim, start, end, step, dtype):
    device = flag_gems.device

    ndim = len(shape)
    dim = dim % ndim
    size = shape[dim]

    start = start % size
    end = end % (size + 1)

    if end < start:
        end, start = start, end
    elif end == start:
        end = size

    inp = torch.randn(shape, dtype=dtype, device=device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.ops.aten.slice.Tensor(ref_inp, dim, start, end, step)

    res_out = flag_gems.slice(inp, dim, start, end, step)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.slice
@pytest.mark.parametrize("shape", SLICE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_slice_none_params(shape, dtype):
    device = flag_gems.device
    dim = 0

    inp = torch.randn(shape, dtype=dtype, device=device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.ops.aten.slice.Tensor(ref_inp, dim, None, None, 1)

    res_out = flag_gems.slice(inp, dim, None, None, 1)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.slice
@pytest.mark.parametrize("shape", SLICE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_slice_negative_indices(shape, dtype):
    device = flag_gems.device
    dim = 0
    size = shape[dim]

    inp = torch.randn(shape, dtype=dtype, device=device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.ops.aten.slice.Tensor(ref_inp, dim, -size, -1, 1)

    res_out = flag_gems.slice(inp, dim, -size, -1, 1)

    utils.gems_assert_equal(res_out, ref_out)
