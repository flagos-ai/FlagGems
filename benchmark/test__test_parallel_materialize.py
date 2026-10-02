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

"""Benchmark for aten::_test_parallel_materialize.

Two call kinds are measured for every dtype the native schema accepts: a plain
operand (the call is a pure alias and does no work) and a lazily negated operand
(the call copies the payload into a fresh contiguous tensor). num_parallel and
skip_first are scheduling hints, so their boundaries run on the small shapes.
"""

import math

import pytest
import torch

import flag_gems

from . import base, consts

_DEVICE = flag_gems.runtime.device

_DTYPE_FLAGS = {
    torch.bfloat16: _DEVICE.support_bf16,
    torch.float64: _DEVICE.support_fp64,
    torch.complex128: _DEVICE.support_fp64,
    torch.int64: _DEVICE.support_int64,
    torch.float8_e4m3fn: _DEVICE.support_fp8,
    torch.float8_e5m2: _DEVICE.support_fp8,
}

# The operator is a pure alias for a materialized operand and a parallel copy for
# a lazy one, so every dtype the native schema accepts is benchmarked: the
# framework's float/int/bool/complex groups plus the remaining supported dtypes,
# gated by the static runtime capability flags.
_BENCH_DTYPES = tuple(
    dtype
    for dtype in dict.fromkeys(
        tuple(consts.FLOAT_DTYPES)
        + tuple(consts.INT_DTYPES)
        + tuple(consts.BOOL_DTYPES)
        + tuple(consts.COMPLEX_DTYPES)
        + (
            torch.int8,
            torch.uint8,
            torch.int64,
            torch.float64,
            torch.complex128,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        )
    )
    if _DTYPE_FLAGS.get(dtype, True)
)

# Materializing a lazy negative view needs a negation kernel, absent on CUDA FP8
# ("neg_cuda" not implemented for 'Float8_e4m3fn' / 'Float8_e5m2') or bool, so
# those dtypes only get the identity kind.
_LAZY_DTYPES = frozenset(
    dtype
    for dtype in _BENCH_DTYPES
    if dtype not in (torch.float8_e4m3fn, torch.float8_e5m2, torch.bool)
)

# Workload sizes kept from the operator's previous benchmark; the framework's
# core and comprehensive shapes stay untouched and are merged with these.
_EXTRA_SHAPES = ((1024, 1024), (20, 320, 15), (16, 128, 64, 60), (16, 7, 57, 32, 29))

_MAIN_CASE = {"kind": "plain", "num_parallel": 3, "skip_first": False}
# num_parallel / skip_first are part of the native contract, so their boundaries
# run as extra semantic cases on the shapes that are cheap to iterate.
_HINT_CASES = (
    {"kind": "plain", "num_parallel": 0, "skip_first": False},
    {"kind": "plain", "num_parallel": 1, "skip_first": False},
    {"kind": "plain", "num_parallel": -1, "skip_first": False},
    {"kind": "plain", "num_parallel": 10**6, "skip_first": False},
    {"kind": "plain", "num_parallel": 2, "skip_first": True},
)
_HINT_ELEMENTS = 64 * 64


def _case_fn(shape, dtype):
    """Plan the two call kinds, plus the hint sweep on the small shapes."""
    for kind in ("plain", "neg"):
        if kind == "neg" and dtype not in _LAZY_DTYPES:
            continue
        params = dict(_MAIN_CASE, kind=kind)
        yield base.BenchmarkCasePlan(
            shape={"input": shape},
            params=params,
            builder_args=(shape, kind, params["num_parallel"], params["skip_first"]),
        )
    if math.prod(shape) <= _HINT_ELEMENTS:
        for params in _HINT_CASES:
            yield base.BenchmarkCasePlan(
                shape={"input": shape},
                params=dict(params),
                builder_args=(
                    shape,
                    params["kind"],
                    params["num_parallel"],
                    params["skip_first"],
                ),
            )


def _build_inputs_fn(plan, dtype, device):
    shape, kind, num_parallel, skip_first = plan.builder_args
    inp = torch.empty(shape, dtype=dtype, device=device)
    if kind == "neg":
        # The materializing branch copies payload data, so it is the branch that
        # actually reads the buffer; the identity branch returns the operand.
        inp = torch._neg_view(inp)
    return inp, num_parallel, {"skip_first": skip_first}


class ParallelMaterializeBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        shapes = [tuple(shape) for shape in list(self.shapes) + list(_EXTRA_SHAPES)]
        self.shapes = list(dict.fromkeys(shapes))


@pytest.mark.test_parallel_materialize
def test__test_parallel_materialize():
    bench = ParallelMaterializeBenchmark(
        op_name="_test_parallel_materialize",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._test_parallel_materialize,
        gems_op=getattr(flag_gems, "_test_parallel_materialize", None),
        dtypes=_BENCH_DTYPES,
    )
    bench.run()
