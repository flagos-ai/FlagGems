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

# aten::_test_autograd_multiple_dispatch_view is a metadata-only flat view
# (Tensor(a) -> Tensor(a)): it launches no device kernel, so the measured latency
# is dispatch plus view construction. The shared shape file, the shared default
# shape list and the comprehensive extras all stay in place, and these
# dispatch-sized shapes are added to them.
VIEW_SHAPES = [
    (64, 64),
    (256, 256),
    (1024, 1024),
    (2048, 2048),
    (20, 320, 15),
    (16, 128, 64, 60),
    (16, 7, 57, 32, 29),
]

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


# Static capability gate: the payload is never read, so only dtype support and
# allocation matter here.
def _supported(dtype):
    device = flag_gems.runtime.device
    if dtype in _FP8_DTYPES:
        return device.support_fp8
    if dtype in (torch.float64, torch.complex128):
        return device.support_fp64
    if dtype == torch.bfloat16:
        return device.support_bf16
    if dtype == torch.int64:
        return device.support_int64
    return True


# Every dtype family the flat view accepts, so each element size is measured.
BENCH_DTYPES = [
    dtype
    for dtype in (
        consts.FLOAT_DTYPES
        + consts.INT_DTYPES
        + consts.EXTRA_INT_DTYPES
        + consts.BOOL_DTYPES
        + consts.COMPLEX_DTYPES
        + [torch.float64, torch.complex128, torch.float8_e4m3fn, torch.float8_e5m2]
    )
    if dtype is not None and _supported(dtype)
]


def _merged_shapes(shared, extra):
    merged = []
    for shape in list(shared) + list(extra):
        shape = tuple(shape)
        if shape not in merged:
            merged.append(shape)
    return merged


class ViewDispatchBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        # Keep whatever the shared shape file or the shared default shape list
        # selected (including the comprehensive extras merged by the base class)
        # and add the operator's own shapes to it.
        super().set_shapes(shape_file_path)
        self.shapes = _merged_shapes(self.shapes, VIEW_SHAPES)


def _case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    # The flat view reads only size and stride, so an uninitialized buffer is a
    # faithful payload and nothing here is ever read.
    shape = plan.builder_args[0]
    inp = torch.empty(shape, dtype=dtype, device=device)
    return inp, {}


@pytest.mark.test_autograd_multiple_dispatch_view
def test_autograd_multiple_dispatch_view():
    bench = ViewDispatchBenchmark(
        op_name="_test_autograd_multiple_dispatch_view",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._test_autograd_multiple_dispatch_view,
        gems_op=getattr(flag_gems, "_test_autograd_multiple_dispatch_view", None),
        dtypes=BENCH_DTYPES,
    )
    bench.run()
