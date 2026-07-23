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

from . import base, consts, utils


@pytest.mark.floor_divide
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_floor_divide():
    bench = base.BinaryPointwiseBenchmark(
        op_name="floor_divide",
        torch_op=torch.floor_divide,
        dtypes=consts.INT_DTYPES,
    )
    bench.run()


@pytest.mark.floor_divide_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_floor_divide_inplace():
    bench = base.BinaryPointwiseBenchmark(
        op_name="floor_divide_",
        torch_op=lambda a, b: a.floor_divide_(b),
        dtypes=consts.INT_DTYPES,
        is_inplace=True,
    )
    bench.run()


def _floor_divide_scalar_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, 3


@pytest.mark.floor_divide_scalar
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_floor_divide_scalar():
    bench = base.GenericBenchmark(
        op_name="floor_divide_scalar",
        torch_op=torch.floor_divide,
        input_fn=_floor_divide_scalar_input_fn,
        dtypes=consts.INT_DTYPES,
    )
    bench.run()


def _floor_divide_scalar_inplace_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, 3


@pytest.mark.floor_divide_scalar_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_floor_divide_scalar_():
    bench = base.GenericBenchmark(
        op_name="floor_divide_scalar_",
        torch_op=lambda a, b: a.floor_divide_(b),
        input_fn=_floor_divide_scalar_inplace_input_fn,
        dtypes=consts.INT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.floor_divide_tensor
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_floor_divide_tensor():
    bench = base.BinaryPointwiseBenchmark(
        op_name="floor_divide_tensor",
        torch_op=torch.floor_divide,
        dtypes=[torch.float32] + consts.INT_DTYPES,
    )
    bench.run()


@pytest.mark.floor_divide_tensor_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_floor_divide_tensor_inplace():
    bench = base.BinaryPointwiseBenchmark(
        op_name="floor_divide_tensor_",
        torch_op=lambda a, b: a.floor_divide_(b),
        dtypes=[torch.float32] + consts.INT_DTYPES,
        is_inplace=True,
    )
    bench.run()
