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
    DIMS_LIST = [1]
    CORRECTION = [1]
    KEEP_DIM = [True]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    DIMS_LIST = [0, 1, [0, 1], [1, 0]]
    CORRECTION = [0, 1]
    KEEP_DIM = [True, False]

# Make sure every thread has same seed.
random.seed(time.time() // 100)


@pytest.mark.std
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIMS_LIST + [None])
@pytest.mark.parametrize("correction", CORRECTION)
@pytest.mark.parametrize("keepdim", KEEP_DIM)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_std(shape, dim, correction, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    dims_to_check = []
    if isinstance(dim, int):
        dims_to_check = [dim]
    elif isinstance(dim, (list, tuple)):
        dims_to_check = dim

    if any(d >= len(shape) or d < -len(shape) for d in dims_to_check):
        # skip the test if dimension is out of range for the given shape.
        return

    if correction == 1:
        if dim is not None:
            positive_dims = [d % len(shape) for d in dims_to_check]
            reduction_size = 1
            for d in positive_dims:
                reduction_size *= shape[d]
            if reduction_size < 2:
                # Invalid case: correction=1 requires reduction size of at least 2
                return
        elif inp.numel() < 2:
            # Invalid case: correction=1 requires numel >= 2 for global reduction.
            return

    ref_inp = utils.to_reference(inp)

    with flag_gems.use_gems():
        res_out = torch.std(inp, dim=dim, correction=correction, keepdim=keepdim)

    ref_out = torch.std(ref_inp, dim=dim, correction=correction, keepdim=keepdim)

    utils.gems_assert_close(res_out, ref_out, dtype)
