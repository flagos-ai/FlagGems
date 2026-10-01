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


# The operator copies its input, so every supported dtype
# family is timed: float, complex, int and bool.
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


# Extra dispatch-friendly shapes appended to the shared core/comprehensive sets;
# every shape is valid for all three call forms.
DISPATCH_SHAPES = [
    (1024, 1024),
    (4096, 4096),
    (20, 320, 15),
    (16, 128, 64),
]

DISPATCH_FORMS = ("fullcoverage", "ntonly", "out")


def _case_fn(shape, dtype):
    del dtype
    for form in DISPATCH_FORMS:
        shape_meta = {"input": shape}
        if form == "out":
            # The out buffer mirrors the input shape.
            shape_meta["out"] = shape
        yield base.BenchmarkCasePlan(
            shape=shape_meta,
            params={"form": form},
            builder_args=(shape, form),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, form = plan.builder_args
    # Benchmark values are never compared and allocation is outside timing, so an
    # uninitialized payload with the right shape/dtype is enough.
    inp = torch.empty(shape, dtype=dtype, device=device)
    # unpack_to_args_kwargs: flat positional arguments plus a trailing kwargs dict.
    if form == "ntonly":
        return inp, {"b": True}
    if form == "out":
        return inp, {"out": torch.empty(shape, dtype=dtype, device=device)}
    return (inp,)


class AutogradMultipleDispatchBenchmark(OperatorBenchmark):
    """Keep the shared shape sets and append the dispatch shapes."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        merged = [tuple(shape) for shape in self.shapes]
        for shape in DISPATCH_SHAPES:
            if tuple(shape) not in merged:
                merged.append(tuple(shape))
        self.shapes = merged


@pytest.mark.test_autograd_multiple_dispatch
def test__test_autograd_multiple_dispatch():
    bench = AutogradMultipleDispatchBenchmark(
        op_name="_test_autograd_multiple_dispatch",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._test_autograd_multiple_dispatch,
        gems_op=getattr(flag_gems, "_test_autograd_multiple_dispatch", None),
        dtypes=BENCH_DTYPES,
    )
    bench.run()
