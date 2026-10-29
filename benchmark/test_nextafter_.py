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

from . import base
from .generated_operator_utils import OperatorBenchmark


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"self": list(dims), "other": list(dims)},
        params={},
        builder_args=(dims,),
    )


def _build_inputs_fn(plan, dtype, device):
    if flag_gems.vendor_name == "enflame" and dtype == torch.float64:
        pytest.skip("Enflame Float64 storage loses adjacent-double nextafter values")
    (shape,) = plan.builder_args
    torch.manual_seed(0)
    return (
        torch.randn(shape, dtype=dtype, device=device),
        torch.randn(shape, dtype=dtype, device=device),
    )


def _fresh_nextafter_latency(self, op, inp, other):
    if inp.untyped_storage()._cdata == other.untyped_storage()._cdata:
        raise ValueError("Fresh nextafter benchmark inputs must not alias")
    # Only self may change. Keep the independent read-only other tensor and
    # clone self through the existing timer hook, outside every timed call.
    return self._get_fresh_input_latency(
        lambda fresh_self: op(fresh_self, other), (inp,), {}
    )


class NextafterInplaceBenchmark(base.BinaryPointwiseBenchmark, OperatorBenchmark):
    get_latency = _fresh_nextafter_latency

    def supports_cases(self):
        return True

    get_case_iter = OperatorBenchmark.get_case_iter
    build_inputs = OperatorBenchmark.build_inputs


@pytest.mark.nextafter_
def test_nextafter_():
    bench = NextafterInplaceBenchmark(
        op_name="nextafter_",
        torch_op=lambda a, b: a.nextafter_(b),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.float16, torch.bfloat16, torch.float32, torch.float64],
        is_inplace=True,
        fresh_inputs=True,
    )
    bench.run()
