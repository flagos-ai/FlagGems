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
from .generated_operator_utils import OperatorBenchmark

# aten::_test_ambiguous_defaults never reads its operand and returns a constant
# int64 CPU scalar, so timing is dispatch/schema overhead and every operand shape
# is valid. The shared core/comprehensive shape sets stay in place; the scales
# below are added on top of them.
AMBIGUOUS_DEFAULTS_SHAPES = [
    (256,),
    (2, 19, 7),
    (1024, 1024),
    (20, 320, 15),
    (16, 128, 64),
    (16, 7, 57, 32),
]

_DTYPE_CAPABILITY = {
    torch.bfloat16: "support_bf16",
    torch.int64: "support_int64",
    torch.float64: "support_fp64",
    torch.complex128: "support_fp64",
    torch.float8_e4m3fn: "support_fp8",
    torch.float8_e5m2: "support_fp8",
}


def _dtype_supported(dtype):
    flag_name = _DTYPE_CAPABILITY.get(dtype)
    if flag_name is None:
        return True
    return bool(getattr(flag_gems.runtime.device, flag_name))


# The operator ignores its operand, so every supported dtype family is covered:
# float, complex, int and bool.
BENCH_DTYPES = [
    dtype
    for dtype in (
        consts.FLOAT_DTYPES
        + consts.COMPLEX_DTYPES
        + consts.INT_DTYPES
        + consts.EXTRA_INT_DTYPES
        + consts.BOOL_DTYPES
        + [torch.float64, torch.complex128, torch.float8_e4m3fn, torch.float8_e5m2]
    )
    if _dtype_supported(dtype)
]


# (native overload, positional parameter values, case label). The candidate gets
# the identical flat positional arguments.
_CALL_FORMS = [
    pytest.param(torch.ops.aten._test_ambiguous_defaults.a, (1, 1), "a", id="a"),
    pytest.param(torch.ops.aten._test_ambiguous_defaults.b, (2, "2"), "b", id="b"),
]


class AmbiguousDefaultsBenchmark(OperatorBenchmark):
    """Keep the shared shapes (and any shape file) and add this op's own scales."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        extras = [tuple(shape) for shape in AMBIGUOUS_DEFAULTS_SHAPES]
        current = [
            tuple(shape) if isinstance(shape, (list, tuple)) else shape
            for shape in self.shapes
        ]
        self.shapes = list(dict.fromkeys(current + extras))


@pytest.mark.test_ambiguous_defaults
@pytest.mark.parametrize("native_overload,params,form", _CALL_FORMS)
def test__test_ambiguous_defaults(native_overload, params, form):
    def case_fn(shape, dtype):
        del dtype
        shape = tuple(shape)
        yield base.BenchmarkCasePlan(
            shape={"input": list(shape)},
            params={"call_form": form, "args": list(params)},
            builder_args=(shape,),
        )

    def build_inputs_fn(plan, dtype, device):
        (shape,) = plan.builder_args
        # The operand is never read, so an uninitialized allocation is the exact
        # payload here; allocation happens outside the timed region.
        inp = torch.empty(shape, dtype=dtype, device=device)
        return (inp, *params)

    bench = AmbiguousDefaultsBenchmark(
        op_name="_test_ambiguous_defaults",
        case_fn=case_fn,
        build_inputs_fn=build_inputs_fn,
        torch_op=native_overload,
        gems_op=getattr(flag_gems, "_test_ambiguous_defaults", None),
        dtypes=BENCH_DTYPES,
    )
    bench.run()
