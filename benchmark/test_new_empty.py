# Copyright 2025 The FlagGems Authors.
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

"""Benchmark for ``aten::new_empty``.

new_empty is an allocation-only operator: the timed work is the allocator call
itself, so the input is a fresh uninitialized source tensor and the requested
size is the measured allocation.  The shared ``core_shapes.yaml`` grid is kept
(and a caller supplied ``--shape_file`` still selects its own grid) with the
extra source/target combinations below appended on top of it.

``torch_op`` is the ATen reference (the perf baseline) and ``gems_op`` is the
FlagGems candidate resolved through ``--override``; both share the exact same
call semantics ``op(source, size, **kwargs)``.  Case metadata is JSON-compatible
(public ``shape``/``params``) while the real objects stay in the private
``builder_args``, so ``--list-cases`` allocates nothing and runs no operator.
"""

import collections

import pytest
import torch

import flag_gems

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

# The explicit-dtype row overrides the output dtype, so an fp16/bf16 source
# allocates a float32 buffer instead of the inherited one.
_OVERRIDE_DTYPE = torch.float32

_NewEmptyCase = collections.namedtuple("_NewEmptyCase", "source size dtype layout")

# Appended rows: (source geometry, requested size, output dtype override, output
# layout override).  The source geometry is what the case allocates and the
# requested size is the new_empty output shape; the two override columns are the
# explicit dtype/layout variants no shared shape can express.
NEW_EMPTY_CASES = [
    _NewEmptyCase((256,), (256,), None, None),
    _NewEmptyCase((1024, 1024), (1024, 1024), None, None),
    _NewEmptyCase((4096, 4096), (4096, 4096), None, None),
    _NewEmptyCase((20, 320, 15), (20, 320, 15), None, None),
    _NewEmptyCase((64, 512, 512), (64, 512, 512), None, None),
    _NewEmptyCase((16, 128, 64, 60), (16, 128, 64, 60), None, None),
    _NewEmptyCase((64, 512, 512), (1024, 1024), None, None),
    _NewEmptyCase((1024, 1024), (1024, 1024), _OVERRIDE_DTYPE, None),
    _NewEmptyCase((1024, 1024), (1024, 1024), None, torch.strided),
]

_DTYPE_FLAGS = {
    torch.bfloat16: flag_gems.runtime.device.support_bf16,
    torch.float64: flag_gems.runtime.device.support_fp64,
    torch.int64: flag_gems.runtime.device.support_int64,
    torch.float8_e4m3fn: flag_gems.runtime.device.support_fp8,
    torch.float8_e5m2: flag_gems.runtime.device.support_fp8,
}
_BENCH_DTYPES = [
    dtype
    for dtype in consts.FLOAT_DTYPES
    + consts.INT_DTYPES
    + consts.EXTRA_INT_DTYPES
    + [
        torch.bool,
        torch.complex64,
        torch.float64,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ]
    if _DTYPE_FLAGS.get(dtype, True)
]


def _as_case(shape):
    # A shared-grid shape asks for its own geometry; the appended rows carry an
    # independent source geometry plus the optional overrides.
    if isinstance(shape, _NewEmptyCase):
        return shape
    size = tuple(int(extent) for extent in shape)
    return _NewEmptyCase(size, size, None, None)


def _case_fn(shape, dtype):
    del dtype
    case = _as_case(shape)
    params = {"size": list(case.size)}
    if case.dtype is not None:
        params["dtype"] = str(case.dtype)
    if case.layout is not None:
        params["layout"] = str(case.layout)
    yield base.BenchmarkCasePlan(
        shape={"input": list(case.source)},
        params=params,
        builder_args=(case,),
    )


def _build_inputs_fn(plan, dtype, device):
    case = plan.builder_args[0]
    # new_empty never reads the source payload, so the input is an uninitialized
    # allocation of the source geometry.
    inp = torch.empty(case.source, dtype=dtype, device=device)
    kwargs = {}
    if case.dtype is not None:
        kwargs["dtype"] = case.dtype
    if case.layout is not None:
        kwargs["layout"] = case.layout
    # Flat positional arguments with a trailing kwargs dict, matching
    # unpack_to_args_kwargs.
    return inp, case.size, kwargs


class NewEmptyBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        self.shapes = list(
            dict.fromkeys([_as_case(shape) for shape in self.shapes] + NEW_EMPTY_CASES)
        )


@pytest.mark.new_empty
def test_new_empty():
    bench = NewEmptyBenchmark(
        op_name="new_empty",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.new_empty,
        gems_op=getattr(flag_gems, "new_empty", None),
        dtypes=_BENCH_DTYPES,
    )
    bench.run()
