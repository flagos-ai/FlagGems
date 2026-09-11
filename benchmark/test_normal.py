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

from . import base, consts


def normal_input_fn(shape, cur_dtype, device):
    loc = torch.full(shape, fill_value=3.0, dtype=cur_dtype, device=device)
    scale = torch.full(shape, fill_value=10.0, dtype=cur_dtype, device=device)
    yield loc, scale


@pytest.mark.normal_tensor_tensor
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_normal_tensor_tensor():
    bench = base.GenericBenchmark(
        input_fn=normal_input_fn,
        op_name="normal_tensor_tensor",
        torch_op=torch.normal,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def normal_tensor_float_input_fn(shape, cur_dtype, device):
    loc = torch.full(shape, fill_value=3.0, dtype=cur_dtype, device=device)
    scale = 10.0
    yield loc, scale


@pytest.mark.normal_tensor_float
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_normal_tensor_float():
    bench = base.GenericBenchmark(
        input_fn=normal_tensor_float_input_fn,
        op_name="normal_tensor_float",
        torch_op=torch.normal,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def normal_inplace_input_fn(shape, dtype, device):
    self = torch.randn(shape, dtype=dtype, device=device)
    loc = 3.0
    scale = 10.0
    yield self, loc, scale


@pytest.mark.normal_float_float_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_normal_inplace():
    bench = base.GenericBenchmark(
        input_fn=normal_inplace_input_fn,
        op_name="normal_float_float_",
        torch_op=torch.Tensor.normal_,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def normal_float_tensor_input_fn(shape, cur_dtype, device):
    mean = 3.0
    scale = torch.full(shape, fill_value=10.0, dtype=cur_dtype, device=device)
    yield mean, scale


@pytest.mark.normal_float_tensor
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_normal_float_tensor():
    bench = base.GenericBenchmark(
        input_fn=normal_float_tensor_input_fn,
        op_name="normal_float_tensor",
        torch_op=torch.normal,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
