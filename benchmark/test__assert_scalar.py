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

MSG = "assert_scalar benchmark"

# The operator has no tensor operand and no shape, so one row is one (scalar, message)
# descriptor and this matrix is the operator's whole parameter space. Every scalar is
# truthy: a falsy one raises inside the timed call, which belongs to the correctness
# tests rather than to a latency measurement.
DEFAULT_SCALARS = [
    (1, MSG),
    (-7, MSG),
    (True, MSG),
    (0.5, MSG),
    (-2.25, MSG),
    (1e30, MSG),
    (5e-324, MSG),
    (float("nan"), MSG),
    (float("inf"), MSG),
    (1 + 2j, MSG),
    (0 + 1j, MSG),
    (-3 - 4j, MSG),
    (1, ""),
    (1, MSG.upper()),
]


def _check_scalar_case(row):
    # Consumed as an error, never as a filter: an invalid row must fail before anything
    # is timed, and a requested row is never replaced by the defaults.
    if (
        isinstance(row, (list, tuple))
        and len(row) == 2
        and isinstance(row[0], (bool, int, float, complex))
        and isinstance(row[1], str)
    ):
        return (row[0], row[1])
    raise ValueError(
        f"assert_scalar case {row!r} is invalid: expected a (number, str) pair"
    )


def _case_fn(row, dtype):
    del dtype
    self_value, assert_msg = _check_scalar_case(row)
    # str() keeps the JSON case metadata representable for nan/inf/subnormal/complex rows.
    yield base.BenchmarkCasePlan(
        shape={"self": str(self_value), "assert_msg": assert_msg},
        params={"self": str(self_value), "assert_msg": assert_msg},
        builder_args=(self_value, assert_msg),
    )


def _build_inputs_fn(plan, dtype, device):
    del dtype, device
    self_value, assert_msg = plan.builder_args
    return (self_value, assert_msg)


class AssertScalarBenchmark(OperatorBenchmark):
    """Benchmark over the scalar/message descriptors of a tensor-free operator."""

    def unpack_to_args_kwargs(self, input_tuple):
        # Keep Python complex scalars positional; the generic unpacker ignores them.
        return input_tuple, {}

    def set_shapes(self, shape_file_path=None):
        # The (scalar, message) descriptors replace the tensor-shape rows, carried by the
        # public default_shapes hook; a shape file configuring this operator still wins.
        super().set_shapes(shape_file_path, default_shapes=DEFAULT_SCALARS)

    def set_more_shapes(self):
        # The descriptors above already cover the operator's whole operand space.
        return []


@pytest.mark.assert_scalar
def test__assert_scalar():
    bench = AssertScalarBenchmark(
        op_name="_assert_scalar",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._assert_scalar,
        gems_op=getattr(flag_gems, "_assert_scalar", None),
        # One placeholder dtype: the operand is a Python scalar, so dtype does not change
        # the workload and repeating the same scalar per float dtype would be noise.
        dtypes=[torch.float32],
    )
    bench.run()
