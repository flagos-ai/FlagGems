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

"""Benchmark cases for ``aten::_test_optional_floatlist``.

The native kernel is CPU-only, so the reference and the candidate both receive
CPU operands. Two call forms share the configured shape grid (core shapes, a
user shape file and the comprehensive level):

  * identity -- ``op(values, None)`` for every supported storage dtype;
  * list -- ``op(values, addends)`` with rank-1 ``float32`` values, where the
    addends operand follows the element count of the shared shape (the schema
    requires ``len(addends) >= values.numel()``); the list is built in the
    input builder, outside the timed region.
"""

import math

import pytest
import torch

import flag_gems

from . import base
from .generated_operator_utils import OperatorBenchmark

TEST_OP = torch.ops.aten._test_optional_floatlist

# CPU-only native kernel: operands must not be moved to the accelerator.
_INPUT_DEVICE = "cpu"

# The identity call form accepts any storage dtype; the list call form adds a
# float32-only branch on top of it.
IDENTITY_DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
    torch.complex64,
    torch.complex128,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
]

_ADDEND = 0.5


def _case_fn(shape, dtype):
    shape = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"values": list(shape)},
        params={"call": "identity", "dtype": str(dtype)},
        builder_args=(shape, False),
    )
    if dtype == torch.float32:
        # The list call form needs rank-1 values: the shared shape keeps its
        # element count and is only flattened to the operand's 1-D sizes.
        sizes = (math.prod(shape),)
        yield base.BenchmarkCasePlan(
            shape={"values": list(sizes)},
            params={
                "call": "addends",
                "addends_len": sizes[0],
                "dtype": str(dtype),
            },
            builder_args=(sizes, True),
        )


def _build_inputs_fn(plan, dtype, device):
    # ``device`` is unused: this kernel only has a CPU implementation.
    shape, with_addends = plan.builder_args
    values = (
        torch.zeros(shape, dtype=dtype, device=_INPUT_DEVICE)
        if with_addends
        else torch.empty(shape, dtype=dtype, device=_INPUT_DEVICE)
    )
    if with_addends:
        return values, {"addends": [_ADDEND] * plan.params["addends_len"]}
    return values, None


class OptionalFloatlistBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        self.shapes = list(
            dict.fromkeys(
                tuple(shape)
                for shape in list(self.shapes)
                + [(1024,), (65536,), (1048576,), (4194304,)]
            )
        )


@pytest.mark.test_optional_floatlist
def test__test_optional_floatlist():
    bench = OptionalFloatlistBenchmark(
        op_name="_test_optional_floatlist",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=TEST_OP,
        gems_op=getattr(flag_gems, "_test_optional_floatlist", None),
        dtypes=IDENTITY_DTYPES,
    )
    bench.run()
