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

from . import base
from .generated_operator_utils import OperatorBenchmark


_SCALARS = (0.5, -1.25, 2.0)


def _tensor_case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"self": list(dims), "exponent": list(dims)},
        params={"overload": "Tensor"},
        builder_args=(dims,),
    )


def _tensor_build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    torch.manual_seed(0)
    inp = torch.empty(shape, dtype=dtype, device=device).uniform_(0.25, 2.0)
    exponent = torch.empty(shape, dtype=dtype, device=device).uniform_(-2.0, 3.0)
    return inp, exponent


def _scalar_case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    for exponent in _SCALARS:
        yield base.BenchmarkCasePlan(
            shape={"self": list(dims)},
            params={"overload": "Scalar", "exponent": exponent},
            builder_args=(dims, exponent),
        )


def _scalar_build_inputs_fn(plan, dtype, device):
    shape, exponent = plan.builder_args
    torch.manual_seed(0)
    inp = torch.empty(shape, dtype=dtype, device=device).uniform_(0.25, 2.0)
    return inp, exponent


@pytest.mark.float_power_tensor_tensor_
@pytest.mark.float_power_
def test_float_power_tensor_():
    bench = OperatorBenchmark(
        op_name="float_power_tensor_tensor_",
        torch_op=lambda inp, exponent: inp.float_power_(exponent),
        case_fn=_tensor_case_fn,
        build_inputs_fn=_tensor_build_inputs_fn,
        dtypes=[torch.float64],
        is_inplace=True,
        fresh_inputs=True,
    )
    bench.run()


@pytest.mark.float_power_tensor_scalar_
@pytest.mark.float_power_
def test_float_power_scalar_():
    bench = OperatorBenchmark(
        op_name="float_power_tensor_scalar_",
        torch_op=lambda inp, exponent: inp.float_power_(exponent),
        case_fn=_scalar_case_fn,
        build_inputs_fn=_scalar_build_inputs_fn,
        dtypes=[torch.float64],
        is_inplace=True,
        fresh_inputs=True,
    )
    bench.run()
