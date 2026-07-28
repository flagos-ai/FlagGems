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

import numpy as np
import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    ADDCMUL_BROADCAST_SHAPES = [
        ((128, 256), (128, 256), (128, 256), (1,)),
    ]
else:
    ADDCMUL_BROADCAST_SHAPES = [
        ((1, 256, 1, 1), (1, 256, 56, 56), (1, 256, 56, 56), (1, 256, 1, 1)),
        ((1, 3), (2, 1), (2, 3), (1, 1)),
        ((4, 1, 16), (1, 8, 1), (4, 8, 16), (4, 1, 1)),
        ((128, 256), (128, 256), (128, 256), (1,)),
    ]


@pytest.mark.addcmul
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro",
    reason="Issues #3861: some ops hang in op tests",
)
def test_addcmul(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    t1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    t2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(res_inp, True)
    ref_t1 = utils.to_reference(t1, True)
    ref_t2 = utils.to_reference(t2, True)

    v = float(np.float32(random.random()))

    ref_out = torch.addcmul(ref_inp, ref_t1, ref_t2, value=v)
    with flag_gems.use_gems():
        res_out = torch.addcmul(res_inp, t1, t2, value=v)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.addcmul_out
@pytest.mark.parametrize(
    "inp_shape, t1_shape, t2_shape, out_shape",
    ADDCMUL_BROADCAST_SHAPES,
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro",
    reason="Issues #3861: some ops hang in op tests",
)
def test_addcmul_out_broadcast(inp_shape, t1_shape, t2_shape, out_shape, dtype):
    res_inp = torch.randn(inp_shape, dtype=dtype, device=flag_gems.device)
    t1 = torch.randn(t1_shape, dtype=dtype, device=flag_gems.device)
    t2 = torch.randn(t2_shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(res_inp, True)
    ref_t1 = utils.to_reference(t1, True)
    ref_t2 = utils.to_reference(t2, True)

    v = float(np.float32(random.random()))

    ref_out_tensor = torch.randn(out_shape, dtype=dtype, device=ref_inp.device)
    ref_result = torch.addcmul(ref_inp, ref_t1, ref_t2, value=v, out=ref_out_tensor)

    res_out_tensor = torch.randn(out_shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        res_result = torch.addcmul(res_inp, t1, t2, value=v, out=res_out_tensor)

    broadcast_shape = torch.broadcast_shapes(inp_shape, t1_shape, t2_shape)
    assert list(res_out_tensor.shape) == list(
        broadcast_shape
    ), f"out tensor was not resized: expected {broadcast_shape}, got {res_out_tensor.shape}"
    utils.gems_assert_close(res_result, ref_result, dtype)
