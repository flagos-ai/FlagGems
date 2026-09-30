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

IS_FLOATING_POINT_DTYPES = (
    consts.FLOAT_DTYPES
    + consts.INT_DTYPES
    + consts.EXTRA_INT_DTYPES
    + consts.BOOL_DTYPES
    + consts.COMPLEX_DTYPES
    # get_fp8_dtype() is None off CUDA, so the fp8 family drops out there.
    + [dtype for dtype in consts.FP8_DTYPES if dtype is not None]
)


def _case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(shape={"input": shape}, builder_args=(shape,))


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    # The operator reads the dtype only, so the operand needs no data at all:
    # no generation cost per sample and no dtype the input helper cannot build.
    return torch.empty(shape, dtype=dtype, device=device), {}


@pytest.mark.is_floating_point
def test_is_floating_point():
    bench = base.GenericBenchmark(
        op_name="is_floating_point",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.is_floating_point,
        gems_op=getattr(flag_gems, "is_floating_point", None),
        dtypes=IS_FLOATING_POINT_DTYPES,
    )
    bench.run()
