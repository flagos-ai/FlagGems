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


def _scalar_input_fn(shape, dtype, device):
    inp1 = utils.generate_tensor_input(shape, dtype, device)
    yield inp1, 0.5


@pytest.mark.greater
def test_greater():
    bench = base.BinaryPointwiseBenchmark(
        op_name="greater",
        torch_op=torch.greater,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def greater_out_input_fn(shape, dtype, device):
    inp1 = base.generate_tensor_input(shape, dtype, device)
    inp2 = base.generate_tensor_input(shape, dtype, device)
    out = torch.empty(shape, dtype=torch.bool, device=device)
    yield inp1, inp2, {"out": out}


@pytest.mark.greater_out
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_greater_out():
    bench = base.GenericBenchmark(
        op_name="greater_out",
        torch_op=torch.greater,
        input_fn=greater_out_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.greater_scalar
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_greater_scalar():
    bench = base.GenericBenchmark(
        input_fn=_scalar_input_fn,
        op_name="greater_scalar",
        torch_op=torch.greater,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def greater_scalar_out_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    out = torch.empty(shape, dtype=torch.bool, device=device)
    yield inp, 0, {"out": out}


@pytest.mark.greater_scalar_out
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_greater_scalar_out():
    bench = base.GenericBenchmark(
        op_name="greater_scalar_out",
        input_fn=greater_scalar_out_input_fn,
        torch_op=torch.greater,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
