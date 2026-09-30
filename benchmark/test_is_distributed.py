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

# is_distributed reads tensor metadata and returns a Python bool, so every dtype
# the backend can allocate is a valid workload. The shared benchmark dtype sets
# supply the float, integer, bool and device-appropriate FP8 types, and the
# shared shape sets (including the COMPREHENSIVE extras) are kept as they are.
BENCH_DTYPES = (
    consts.FLOAT_DTYPES
    + consts.INT_DTYPES
    + consts.EXTRA_INT_DTYPES
    + consts.BOOL_DTYPES
    + [dtype for dtype in consts.FP8_DTYPES if dtype is not None]
)


def _case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    # No value is ever read, so an uninitialized allocation is the honest
    # fixture; torch.empty also covers the FP8 and bool dtypes that the shared
    # value generator does not.
    return torch.empty(shape, dtype=dtype, device=device), {}


@pytest.mark.is_distributed
def test_is_distributed():
    bench = base.GenericBenchmark(
        op_name="is_distributed",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.is_distributed,
        gems_op=getattr(flag_gems, "is_distributed", None),
        dtypes=BENCH_DTYPES,
    )
    bench.run()
