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

# aten::is_complex(Tensor self) -> bool returns a Python bool and reads no
# element, so no public Benchmark family covers it: the two-phase
# GenericBenchmark below (case_fn + build_inputs_fn) keeps the shared shape and
# dtype handling. Both dtype families are benchmarked so the timing covers the
# True and the False branch of the query.
_IS_COMPLEX_DTYPES = consts.FLOAT_DTYPES + consts.COMPLEX_DTYPES


def _case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(shape={"input": shape}, builder_args=(shape,))


def _build_inputs_fn(plan, dtype, device):
    # One positional tensor, matching the operator signature. The query never
    # reads an element, so an uninitialized allocation is enough and keeps the
    # shared default shapes from being dominated by input generation.
    return (torch.empty(plan.builder_args[0], dtype=dtype, device=device),)


@pytest.mark.is_complex
def test_is_complex():
    bench = base.GenericBenchmark(
        op_name="is_complex",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.is_complex,
        gems_op=getattr(flag_gems, "is_complex", None),
        dtypes=_IS_COMPLEX_DTYPES,
    )
    bench.run()
