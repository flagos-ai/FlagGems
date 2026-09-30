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

"""Benchmark for ``aten::is_signed``.

The operator returns a Python bool and reads no stored element, so the shared
default shapes and dtype families apply unchanged and the inputs are
uninitialized buffers.
"""

import pytest
import torch

import flag_gems

from . import base, consts

# is_signed takes no parameter besides the input tensor.
IS_SIGNED_DTYPES = (
    consts.FLOAT_DTYPES
    + consts.INT_DTYPES
    + consts.EXTRA_INT_DTYPES
    + consts.BOOL_DTYPES
)


def _case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    # No element is read, so the buffer contents cannot enter the measurement.
    return torch.empty(plan.builder_args[0], dtype=dtype, device=device), {}


@pytest.mark.is_signed
def test_is_signed():
    bench = base.GenericBenchmark(
        op_name="is_signed",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.is_signed,
        gems_op=getattr(flag_gems, "is_signed", None),
        dtypes=IS_SIGNED_DTYPES,
    )
    bench.run()
