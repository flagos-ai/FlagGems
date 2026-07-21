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


@pytest.mark.pow_tensor_tensor
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_pow_tensor_tensor():
    bench = base.ScalarBinaryPointwiseBenchmark(
        op_name="pow_tensor_tensor",
        torch_op=torch.pow,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.pow_tensor_tensor_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_pow_inplace():
    bench = base.BinaryPointwiseBenchmark(
        op_name="pow_tensor_tensor_",
        torch_op=lambda a, b: a.pow_(b),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.pow_scalar
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_pow_scalar():
    bench = base.ScalarBinaryPointwiseBenchmark(
        op_name="pow_scalar",
        torch_op=torch.pow,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def pow_tensor_scalar_input_fn(shape, dtype, device):
    inp = base.generate_tensor_input(shape, dtype, device)
    scalar = 0.001
    yield inp, scalar


@pytest.mark.pow_tensor_scalar
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_pow_tensor_scalar():
    bench = base.GenericBenchmark(
        input_fn=pow_tensor_scalar_input_fn,
        op_name="pow_tensor_scalar",
        torch_op=torch.pow,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.pow_tensor_scalar_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_pow_tensor_scalar_inplace():
    bench = base.GenericBenchmark(
        input_fn=pow_tensor_scalar_input_fn,
        op_name="pow_tensor_scalar_",
        torch_op=lambda a, b: a.pow_(b),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
