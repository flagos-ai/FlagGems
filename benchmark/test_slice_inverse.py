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

"""Benchmark for ``aten::slice_inverse``.

The operator is a view, so a workload is the operand pair (self, src) plus the
slice arguments; every shape is measured with a full-extent and a half-extent
src, and boundary shapes cover scalar, size-1 and zero-extent inputs.
``benchmark/core_shapes.yaml`` has no ``slice_inverse`` entry, so the shared
default grid is loaded through the base class and the boundary shapes are
appended, keeping caller-supplied shape files intact.
"""

import pytest
import torch

import flag_gems

from . import base, consts

BOUNDARY_SHAPES = [
    (),
    (4, 0),
    (512,),
    (4, 6),
    (256, 256),
    (1024, 1024),
    (4096, 4096),
    (128, 512, 256),
    (20, 320, 15),
    (16, 128, 64, 60),
    (16, 7, 57, 32, 29),
]

_DIM = 0


def _dtypes():
    dtypes = list(
        consts.FLOAT_DTYPES
        + consts.INT_DTYPES
        + consts.EXTRA_INT_DTYPES
        + consts.BOOL_DTYPES
        + consts.COMPLEX_DTYPES
    )
    fp8 = consts.get_fp8_dtype()
    if fp8 is not None and fp8 not in dtypes:
        dtypes.append(fp8)
    return dtypes


SLICE_INVERSE_DTYPES = _dtypes()


def _case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"self": list(shape), "src": list(shape)},
        params={"dim": _DIM, "start": None, "end": None, "step": 1},
        builder_args=(shape, None),
    )
    if not shape:
        return
    half = shape[0] // 2
    if half == shape[0]:
        # No shorter slice exists for this extent, so the full-extent plan is
        # the only valid workload.
        return
    yield base.BenchmarkCasePlan(
        shape={"self": list(shape), "src": [half, *shape[1:]]},
        params={"dim": _DIM, "start": 0, "end": half, "step": 1},
        builder_args=(shape, half),
    )


def _build_inputs_fn(plan, dtype, device):
    shape, end = plan.builder_args
    inp = torch.empty(shape, dtype=dtype, device=device)
    if end is None:
        # Flat positional operands plus the trailing kwargs dict consumed by
        # Benchmark.unpack_to_args_kwargs.
        return inp, inp, {"dim": _DIM, "start": None, "end": None, "step": 1}
    return inp, inp[:end], {"dim": _DIM, "start": 0, "end": end, "step": 1}


class SliceInverseBenchmark(base.GenericBenchmark):
    """Keep the shared shape grid and add the slice_inverse boundary shapes."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        self.shapes = list(
            dict.fromkeys(tuple(shape) for shape in list(self.shapes) + BOUNDARY_SHAPES)
        )


@pytest.mark.slice_inverse
def test_slice_inverse():
    bench = SliceInverseBenchmark(
        op_name="slice_inverse",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.slice_inverse,
        gems_op=getattr(flag_gems, "slice_inverse", None),
        dtypes=SLICE_INVERSE_DTYPES,
    )
    bench.run()
