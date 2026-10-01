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

"""Benchmark for ``aten::_test_optional_filled_intlist``.

The native op is CPU-only, so every payload tensor is allocated on the CPU.
``addends=None`` (object identity) accepts any dtype and rank, while the
arithmetic form needs rank-1 int32 values plus one addend per element (the
``int[2]?`` fill only covers tensors of up to two elements). Each shared shape
is therefore timed once as an identity case and, for int32, again as a
flattened rank-1 arithmetic case with the same element count. ``torch_op`` is
the ATen reference and ``gems_op`` the FlagGems candidate; both receive the
identical positional ``(values, addends)`` argument list.
"""

import math

import pytest
import torch

import flag_gems

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

# Native dispatch is CPU only (CUDA and Meta raise NotImplementedError), so no
# device capability gate applies to these allocations.
_CPU = torch.device("cpu")

# The identity form is dtype-agnostic. The float8 pair is listed explicitly
# because consts.FP8_DTYPES is None on a host without FP8 support.
_BENCH_DTYPES = list(
    dict.fromkeys(
        consts.FLOAT_DTYPES
        + consts.INT_DTYPES
        + consts.EXTRA_INT_DTYPES
        + consts.BOOL_DTYPES
        + consts.COMPLEX_DTYPES
        + [torch.float64, torch.complex128, torch.float8_e4m3fn, torch.float8_e5m2]
    )
)

_BENCH_EXTRA_SHAPES = [(1,), (2,), (64,), (1024,), (16384,)]

_FILLED_INT_ADDEND = 7
_ADDEND_CYCLE = (3, -4, 0, 7, -2)


def _checked_shape(shape):
    """Validate shape metadata while listing, without allocating anything."""
    try:
        dims = tuple(shape)
    except TypeError:
        raise ValueError("a shape must be a sequence of extents") from None
    for dim in dims:
        if isinstance(dim, bool) or not isinstance(dim, int) or dim < 0:
            raise ValueError(f"invalid shape dimension {dim!r}")
    return dims


def _dedup_shapes(shapes):
    dims_list = []
    for shape in shapes:
        dims = _checked_shape(shape)
        if dims not in dims_list:
            dims_list.append(dims)
    return dims_list


def _addend_list(numel):
    # One addend per element is required for the list form. Repeating a short
    # cycle keeps the int objects shared instead of building new Python ints
    # per element; extra entries past `numel` are ignored by the native loop.
    repeats = max(numel, 2) // len(_ADDEND_CYCLE) + 1
    return list(_ADDEND_CYCLE) * repeats


def _case_fn(shape, dtype):
    dims = _checked_shape(shape)
    numel = math.prod(dims)
    # Identity: `addends=None` is rank- and dtype-agnostic.
    yield base.BenchmarkCasePlan(
        shape={"input": list(dims)},
        params={"addends": None},
        builder_args=(dims, "identity", numel),
    )
    if dtype != torch.int32:
        # The arithmetic path is int32 only; every other dtype keeps the
        # identity form above.
        return
    flat = (numel,)
    if numel <= 2:
        yield base.BenchmarkCasePlan(
            shape={"input": list(flat)},
            params={"addends": _FILLED_INT_ADDEND},
            builder_args=(flat, "filled_int", numel),
        )
    yield base.BenchmarkCasePlan(
        shape={"input": list(flat)},
        params={"addends": {"mode": "list", "length": numel}},
        builder_args=(flat, "list", numel),
    )


def _build_inputs_fn(plan, dtype, device):
    # Identity does not read the payload; arithmetic uses initialized zeros.
    # `device` is ignored: the native op is CPU-only.
    dims, form, numel = plan.builder_args
    values = (
        torch.empty(dims, dtype=dtype, device=_CPU)
        if form == "identity"
        else torch.zeros(dims, dtype=dtype, device=_CPU)
    )
    if form == "identity":
        return values, None
    if form == "filled_int":
        return values, _FILLED_INT_ADDEND
    return values, _addend_list(numel)


class OptionalFilledIntlistBenchmark(OperatorBenchmark):
    """Shared core/comprehensive grid plus the rank-1 arithmetic shapes.

    ``super().set_shapes`` keeps the framework grid and any caller shape-file
    entries; the extra rank-1 shapes are unioned in and the result is deduped,
    so no shared workload is replaced or dropped.
    """

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        self.shapes = _dedup_shapes(list(self.shapes) + _BENCH_EXTRA_SHAPES)


@pytest.mark.test_optional_filled_intlist
def test__test_optional_filled_intlist():
    bench = OptionalFilledIntlistBenchmark(
        op_name="_test_optional_filled_intlist",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._test_optional_filled_intlist,
        gems_op=getattr(flag_gems, "_test_optional_filled_intlist", None),
        dtypes=_BENCH_DTYPES,
    )
    bench.run()
