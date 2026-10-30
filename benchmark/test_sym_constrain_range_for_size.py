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

# aten::sym_constrain_range_for_size has no tensor input; the workload axis is
# the scalar size being checked. Each entry below is therefore a size
# descriptor (a 1-tuple, the form the shared loader yields) and every bound row
# keeps that size in range, i.e. the accepting, timed path. No tensor is
# allocated for listing or execution.
_DEFAULT_SIZES = [
    (1,),
    (16,),
    (256,),
    (4096,),
    (65536,),
    (1048576,),
]

# (min, max); a None bound is omitted from the call, i.e. the schema default.
_BOUND_ROWS = [
    (None, None),
    (0, None),
    (0, 2**63 - 1),
]


def _size_of(shape):
    if isinstance(shape, (tuple, list)):
        if len(shape) != 1:
            raise ValueError("a size descriptor must contain exactly one scalar")
        shape = shape[0]
    if isinstance(shape, bool) or not isinstance(shape, int):
        raise TypeError("a size descriptor must be an integer")
    return shape


def _case_fn(shape, dtype):
    del dtype
    size = _size_of(shape)
    for min_value, max_value in _BOUND_ROWS:
        yield base.BenchmarkCasePlan(
            shape={"size": size},
            params={"min": min_value, "max": max_value},
            builder_args=(size, min_value, max_value),
        )


def _build_inputs_fn(plan, dtype, device):
    del dtype, device
    size, min_value, max_value = plan.builder_args
    kwargs = {
        name: value
        for name, value in (("min", min_value), ("max", max_value))
        if value is not None
    }
    return size, kwargs


class SymConstrainRangeForSizeBenchmark(base.GenericBenchmark):
    """Custom benchmark for sym_constrain_range_for_size - it validates a scalar
    bound with a Triton kernel and returns nothing (void), so the inputs are
    scalar sizes rather than tensors."""

    def set_shapes(self, shape_file_path=None):
        # The scalar size descriptors are the whole workload axis, so they are
        # set directly instead of being loaded from core_shapes.yaml, which
        # only describes tensor shapes.
        self.shapes = _DEFAULT_SIZES
        self.shape_desc = "size"

    def set_more_shapes(self):
        return []


@pytest.mark.sym_constrain_range_for_size
def test_sym_constrain_range_for_size():
    bench = SymConstrainRangeForSizeBenchmark(
        op_name="sym_constrain_range_for_size",
        torch_op=torch.ops.aten.sym_constrain_range_for_size,
        # No gems_op override: the operator is registered on the dispatcher key
        # it actually resolves to, so the candidate side goes through use_gems()
        # dispatch like every other op and the timing covers that path.
        #
        # The operator takes a Scalar size and no tensor, so the dtype grid does
        # not apply; int64 is the schema type of the bounds it is compared to.
        dtypes=[torch.int64],
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
    )
    bench.run()
