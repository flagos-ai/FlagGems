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


@pytest.mark.softmax
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_softmax():
    bench = base.UnaryReductionBenchmark(
        op_name="softmax",
        torch_op=torch.nn.functional.softmax,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.softmax_backward
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_softmax_backward():
    bench = base.UnaryReductionBenchmark(
        op_name="softmax",
        torch_op=torch.nn.functional.softmax,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=True,
    )

    bench.run()


def softmax_backward_out_input_fn(shape, dtype, device):
    grad_output = torch.randn(shape, dtype=dtype, device=device)
    output = torch.randn(shape, dtype=dtype, device=device)
    grad_input = torch.empty(shape, dtype=dtype, device=device)
    dim = 1 if len(shape) > 1 else 0
    yield grad_output, output, dim, dtype, {"grad_input": grad_input}


@pytest.mark.softmax_backward_out
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_softmax_backward_out():
    bench = base.GenericBenchmark(
        op_name="softmax_backward_out",
        torch_op=torch.ops.aten._softmax_backward_data.out,
        input_fn=softmax_backward_out_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def softmax_out_input_fn(shape, dtype, device):
    inp = base.generate_tensor_input(shape, dtype, device)
    out = torch.empty(shape, dtype=dtype, device=device)
    dim = 1 if inp.ndim > 1 else 0
    yield inp, dim, False, {"out": out}


@pytest.mark.softmax_out
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_softmax_out():
    bench = base.GenericBenchmark(
        op_name="softmax_out",
        torch_op=torch.ops.aten._softmax.out,
        input_fn=softmax_out_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
