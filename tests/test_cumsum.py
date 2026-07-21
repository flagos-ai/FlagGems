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
    CUMSUM_SHAPES = [(2, 32)]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    CUMSUM_SHAPES = utils.REDUCTION_SHAPES + [(2637,), (16, 1025, 255)]

random.seed(time.time() // 100)


@pytest.mark.cumsum
@pytest.mark.parametrize("shape", CUMSUM_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + utils.INT_DTYPES)
def test_cumsum(shape, dtype):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    dim = 1 if shape == utils.REDUCTION_SHAPES[-1] else -1
    if dtype in utils.INT_DTYPES:
        inp = torch.randint(-3, 3, shape, device=flag_gems.device).to(dtype)
        ref_inp = utils.to_reference(inp)
    else:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        ref_inp = utils.to_reference(inp, True)

    ref_out = torch.cumsum(ref_inp, dim=dim)
    # Issue 2806: This customization doesn't look correct.
    if flag_gems.vendor_name == "kunlunxin":
        from flag_gems.runtime.backend._kunlunxin import ops as kl_ops

        res_out = kl_ops.cumsum(inp, dim=dim)
    else:
        with flag_gems.use_gems():
            res_out = torch.cumsum(inp, dim=dim)

    # we should use ref's output type, since cumsum of int dtype results in int64
    if flag_gems.vendor_name in ["cambricon", "enflame", "tsingmicro"]:
        check_dtype = dtype
    elif dtype in utils.INT_DTYPES:
        check_dtype = ref_out.dtype
    else:
        check_dtype = dtype

    utils.gems_assert_close(res_out, ref_out, check_dtype, reduce_dim=shape[dim])


@pytest.mark.cumsum_out
@pytest.mark.parametrize("shape", CUMSUM_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_cumsum_out(shape, dtype):
    dim = 1 if shape == utils.REDUCTION_SHAPES[-1] else -1
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    out = torch.empty_like(inp)
    ref_out_buf = torch.empty_like(ref_inp)

    torch.cumsum(ref_inp, dim=dim, out=ref_out_buf)
    with flag_gems.use_gems():
        torch.cumsum(inp, dim=dim, out=out)

    utils.gems_assert_close(out, ref_out_buf, dtype, reduce_dim=shape[dim])
