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

# Only a CPU kernel is registered for this placeholder operator, so the timed
# reference and candidate both run on CPU tensors.
_CPU = torch.device("cpu")

# Extra workloads appended to the shared shape set (core shapes or shape file):
# the op returns its input unchanged, so latency is dominated by call overhead
# and these shapes keep that signal measurable.
_EXTRA_SHAPES = [
    (2, 19, 7),
    (256,),
    (1024, 1024),
    (20, 320, 15),
    (16, 128, 64, 60),
    (16, 7, 57, 32, 29),
]

# arg1/arg2/arg3 do not change the result; time the schema defaults and the
# explicit-False call form.
_PARAM_CASES = [
    {},
    {"arg1": False, "arg2": False, "arg3": False},
]

# Every dtype family the CPU kernel accepts.
_BENCH_DTYPES = (
    consts.FLOAT_DTYPES
    + consts.INT_DTYPES
    + consts.BOOL_DTYPES
    + consts.COMPLEX_DTYPES
    + consts.EXTRA_INT_DTYPES
    + [torch.float64, torch.complex128, torch.float8_e4m3fn, torch.float8_e5m2]
)


def _case_fn(shape, dtype):
    del dtype
    for params in _PARAM_CASES:
        yield base.BenchmarkCasePlan(
            shape={"input": list(shape)},
            params=params,
            builder_args=(tuple(shape),),
        )


def _build_inputs_fn(plan, dtype, device):
    del device
    shape = plan.builder_args[0]
    # The op returns its input unchanged and the timing never reads the values, so
    # an uninitialized CPU allocation is enough and keeps the measurement free of
    # fill work. utils.generate_tensor_input returns None for several supported
    # dtypes, so it is not used here. Positional args plus a trailing kwargs dict
    # is the form unpack_to_args_kwargs expects.
    inp = torch.empty(shape, dtype=dtype, device=_CPU)
    return inp, dict(plan.params)


class FoobarBenchmark(OperatorBenchmark):
    """Two-phase GenericBenchmark for the CPU-only aten::_foobar placeholder."""

    def set_shapes(self, shape_file_path=None):
        # Keep the shared shape set (caller shape file or core shapes) and append
        # the extra workloads above, deduplicating in place without dropping any
        # of the shared core/comprehensive workloads.
        super().set_shapes(shape_file_path)
        merged = [tuple(shape) for shape in self.shapes]
        for shape in _EXTRA_SHAPES:
            if shape not in merged:
                merged.append(shape)
        self.shapes = merged


@pytest.mark.foobar
def test__foobar():
    bench = FoobarBenchmark(
        op_name="_foobar",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._foobar,
        gems_op=getattr(flag_gems, "_foobar", None),
        dtypes=_BENCH_DTYPES,
    )
    bench.run()
