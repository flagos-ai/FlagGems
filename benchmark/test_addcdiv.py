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


def _input_fn(shape, dtype, device):
    inp1 = utils.generate_tensor_input(shape, dtype, device)
    inp2 = utils.generate_tensor_input(shape, dtype, device)
    inp3 = utils.generate_tensor_input(shape, dtype, device)

    yield inp1, inp2, inp3, {"value": 0.5}


@pytest.mark.addcdiv
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_addcdiv():
    bench = base.GenericBenchmark(
        op_name="addcdiv",
        input_fn=_input_fn,
        torch_op=torch.addcdiv,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def _input_fn_out(shape, dtype, device):
    inp1 = utils.generate_tensor_input(shape, dtype, device)
    inp2 = utils.generate_tensor_input(shape, dtype, device)
    inp3 = utils.generate_tensor_input(shape, dtype, device)
    out = torch.empty_like(inp1)

    yield inp1, inp2, inp3, {"value": 0.5, "out": out}


@pytest.mark.addcdiv_out
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_addcdiv_out():
    bench = base.GenericBenchmark(
        op_name="addcdiv_out",
        input_fn=_input_fn_out,
        torch_op=torch.addcdiv,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
