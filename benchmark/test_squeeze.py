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

_DTYPE_FLAGS = {
    torch.bfloat16: flag_gems.runtime.device.support_bf16,
    torch.float64: flag_gems.runtime.device.support_fp64,
    torch.complex128: flag_gems.runtime.device.support_fp64,
    torch.int64: flag_gems.runtime.device.support_int64,
    torch.float8_e4m3fn: flag_gems.runtime.device.support_fp8,
    torch.float8_e5m2: flag_gems.runtime.device.support_fp8,
}

# squeeze only relabels dims: the measured work is dispatch plus view
# construction and the input values are never read, so the builder uses
# torch.empty. A size-1 or empty extent is the only thing the operator reacts
# to, and the shared core/comprehensive grids contain none, so these boundary
# shapes are appended to the shared grids instead of replacing them; shapes
# configured through a shape file are kept as well.
_SQUEEZE_BOUNDARY_SHAPES = [
    (1,),
    (1, 256),
    (256, 1),
    (1, 1, 1),
    (2, 1, 3),
    (1, 1024, 1024),
    (20, 1, 320, 15),
    (16, 1, 128, 64, 60),
    (16, 7, 57, 1, 29),
    (1, 7, 1, 32, 1),
    (0, 1, 3),
]

# Metadata-only op: every dtype the native schema accepts is benchmarked, and
# the same list is used at core and comprehensive level (no --quick shrink).
_BENCH_DTYPES = list(
    dict.fromkeys(
        consts.FLOAT_DTYPES
        + consts.INT_DTYPES
        + consts.EXTRA_INT_DTYPES
        + consts.BOOL_DTYPES
        + consts.COMPLEX_DTYPES
        + [torch.float8_e4m3fn, torch.float8_e5m2, torch.float64]
    )
)


def _case_fn(shape, dtype):
    del dtype
    # squeeze(x) (every size-1 dim) for all shapes; the .dim and .dims forms are
    # added for shapes that actually own a size-1 dim, since a shape without one
    # only ever exercises the no-op squash path.
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={"dim": None},
        builder_args=(shape, None),
    )
    singletons = [dim for dim, size in enumerate(shape) if size == 1]
    if singletons:
        yield base.BenchmarkCasePlan(
            shape={"input": list(shape)},
            params={"dim": singletons[0]},
            builder_args=(shape, singletons[0]),
        )
    if len(singletons) > 1:
        yield base.BenchmarkCasePlan(
            shape={"input": list(shape)},
            params={"dim": singletons},
            builder_args=(shape, singletons),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, dim = plan.builder_args
    inp = torch.empty(shape, dtype=dtype, device=device)
    if dim is None:
        return (inp,)
    return inp, dim


class SqueezeBenchmark(base.GenericBenchmark):
    """Shared core/comprehensive shape grids plus the squeeze boundaries."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        self.shapes = list(
            dict.fromkeys(
                tuple(shape) for shape in self.shapes + _SQUEEZE_BOUNDARY_SHAPES
            )
        )


@pytest.mark.squeeze
def test_squeeze():
    bench = SqueezeBenchmark(
        op_name="squeeze",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.squeeze,
        gems_op=getattr(flag_gems, "squeeze", None),
        dtypes=[dtype for dtype in _BENCH_DTYPES if _DTYPE_FLAGS.get(dtype, True)],
    )
    bench.run()
