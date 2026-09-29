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

import numbers

import pytest
import torch

import flag_gems

from . import base, consts, utils
from .generated_operator_utils import OperatorBenchmark

# aten::_cast_Half(Tensor self, bool non_blocking=False) -> Tensor
#
# One tensor in, one float16 tensor out: a unary pointwise conversion, so the
# benchmark keeps the unary pointwise shape family - the shape-file geometry and
# the comprehensive shape scales - on top of the two-phase case plans of the
# generated-operator helper. float16 is a legal input dtype whose native call
# returns the input tensor unchanged, so it stays in the pool as the operator's
# identity branch.
_SUPPORT_BF16 = flag_gems.runtime.device.support_bf16
_BENCH_DTYPES = [
    dtype
    for dtype in consts.FLOAT_DTYPES
    if dtype is not torch.bfloat16 or _SUPPORT_BF16
]

# One plan per call form. The params describe exactly the call each plan makes,
# so replaying a plan from its listing metadata reproduces it: no params entry
# for the omitted schema default, and the JSON booleans for the explicit forms.
_NON_BLOCKING_FORMS = (None, False, True)


def _validated_shape(shape):
    """Reject malformed extents while building listing metadata."""
    if not isinstance(shape, (tuple, list)):
        raise TypeError(f"_cast_Half benchmark shape must be a sequence, got {shape!r}")
    extents = tuple(shape)
    for extent in extents:
        if (
            isinstance(extent, bool)
            or not isinstance(extent, numbers.Integral)
            or extent < 0
        ):
            raise ValueError(
                f"_cast_Half benchmark shape has an invalid extent: {extents!r}"
            )
    return tuple(int(extent) for extent in extents)


def _case_fn(shape, dtype):
    # Metadata only: listing allocates no input and calls no operator. A 0-dim
    # shape and zero extents stay valid.
    del dtype
    extents = _validated_shape(shape)
    plans = []
    for non_blocking in _NON_BLOCKING_FORMS:
        params = {} if non_blocking is None else {"non_blocking": non_blocking}
        plans.append(
            base.BenchmarkCasePlan(
                shape={"input": extents},
                params=params,
                builder_args=(extents, non_blocking),
            )
        )
    return plans


def _build_inputs_fn(plan, dtype, device):
    shape, non_blocking = plan.builder_args
    inp = utils.generate_tensor_input(shape, dtype, device)
    if non_blocking is None:
        # Omitted argument: exercises the schema default.
        return inp, {}
    return inp, {"non_blocking": non_blocking}


class CastHalfBenchmark(OperatorBenchmark, base.UnaryPointwiseBenchmark):
    """Two-phase _cast_Half cases with the unary pointwise shape scales."""

    def set_more_shapes(self):
        # GenericBenchmark precedes UnaryPointwiseBenchmark in the MRO, so the
        # unary comprehensive shapes have to be delegated explicitly. Shapes
        # come from the framework shape files; an invalid shape file raises
        # instead of falling back to the defaults.
        return base.UnaryPointwiseBenchmark.set_more_shapes(self)


@pytest.mark.cast_Half
def test__cast_Half():
    bench = CastHalfBenchmark(
        op_name="_cast_Half",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._cast_Half,
        gems_op=getattr(flag_gems, "_cast_Half", None),
        dtypes=_BENCH_DTYPES,
    )
    bench.run()
