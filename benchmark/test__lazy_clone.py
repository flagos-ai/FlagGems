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

# _lazy_clone relocates storage only. The shared shape file / default shapes are
# kept unchanged (no override, no cap); the extra bandwidth-relevant shapes
# below are contributed at the comprehensive level, and every case is expanded
# into a contiguous and a transposed plan so the copied non-contiguous geometry
# is measured as well.
LAZY_CLONE_EXTRA_SHAPES = [
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (20, 320, 15),
    (16, 128, 64, 60),
]

# Static capability flags read at import time: listing allocates no tensor and
# calls no operator.
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


# The operator is a dtype-agnostic storage relocation, so every supported dtype
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


def _transposed_shape(shape):
    return list(shape[:-2]) + [shape[-1], shape[-2]]


def _case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={"layout": "contiguous"},
        builder_args=(shape, "contiguous"),
    )
    if len(shape) >= 2:
        yield base.BenchmarkCasePlan(
            shape={"input": _transposed_shape(shape), "storage": list(shape)},
            params={"layout": "transposed"},
            builder_args=(shape, "transposed"),
        )


def _build_inputs_fn(plan, dtype, device):
    storage_shape, layout = plan.builder_args
    inp = torch.empty(storage_shape, dtype=dtype, device=device)
    if layout == "transposed":
        inp = inp.transpose(-1, -2)
    # Flat positional arguments with a trailing kwargs dict.
    return inp, {}


class LazyCloneBenchmark(OperatorBenchmark):
    def set_more_shapes(self):
        return super().set_more_shapes() + LAZY_CLONE_EXTRA_SHAPES


@pytest.mark.lazy_clone
def test__lazy_clone():
    bench = LazyCloneBenchmark(
        op_name="_lazy_clone",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._lazy_clone,
        gems_op=getattr(flag_gems, "_lazy_clone", None),
        dtypes=BENCH_DTYPES,
    )
    bench.run()
