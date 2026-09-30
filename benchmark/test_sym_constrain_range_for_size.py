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


class SymConstrainRangeForSizeBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        # Shared loader with the operator's scalar descriptors as the default;
        # a caller entry is unioned in rather than replacing them. These sizes
        # are the whole workload axis, so no extra shapes are merged in.
        super().set_shapes(shape_file_path, default_shapes=_DEFAULT_SIZES)
        for size in _DEFAULT_SIZES:
            if size not in self.shapes:
                self.shapes.append(size)

    def set_more_shapes(self):
        return []


@pytest.mark.sym_constrain_range_for_size
def test_sym_constrain_range_for_size():
    bench = SymConstrainRangeForSizeBenchmark(
        op_name="sym_constrain_range_for_size",
        torch_op=torch.ops.aten.sym_constrain_range_for_size,
        gems_op=getattr(flag_gems, "sym_constrain_range_for_size", None),
        dtypes=[torch.int64],
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
    )
    bench.run()
