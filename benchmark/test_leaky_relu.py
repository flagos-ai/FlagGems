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

from typing import Generator

import pytest
import torch

import flag_gems

from . import base, consts, utils


@pytest.mark.leaky_relu
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_leaky_relu():
    bench = base.UnaryPointwiseBenchmark(
        op_name="leaky_relu",
        torch_op=torch.nn.functional.leaky_relu,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.leaky_relu_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_leaky_relu_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="leaky_relu_",
        torch_op=torch.nn.functional.leaky_relu_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.leaky_relu_out
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_leaky_relu_out():
    bench = base.UnaryPointwiseBenchmark(
        op_name="leaky_relu_out",
        torch_op=torch.nn.functional.leaky_relu,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


class LeakyReluBackwardBenchmark(base.UnaryPointwiseBenchmark):
    def get_input_iter(self, dtype: torch.dtype) -> Generator:
        for shape in self.shapes:
            inp = utils.generate_tensor_input(shape, dtype, self.device)
            grad_out = torch.randn_like(inp)
            yield grad_out, inp, 0.01, False


@pytest.mark.leaky_relu_backward
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_leaky_relu_backward():
    bench = LeakyReluBackwardBenchmark(
        op_name="leaky_relu_backward",
        torch_op=torch.ops.aten.leaky_relu_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
