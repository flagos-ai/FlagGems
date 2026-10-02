# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark for aten::as_strided_ (in-place metadata reinterpretation).

Shapes come from the shared resolver: the operator entry in the benchmark shape
file (core_shapes.yaml) when present, otherwise consts.DEFAULT_SHAPES plus the
level extras. Every shape expands into the metadata layouts that as_strided_
actually changes (contiguous, reversed, flattened and dilated); no element is
moved or read, so the fixtures are uninitialised buffers.

The operator is in place, hence fresh_inputs=True: without it a timed call would
reinterpret the already-reinterpreted operand and leak metadata across
iterations. torch_op and gems_op share one call form,
op(input, size, stride, storage_offset), with every argument positional.
"""

import math

import pytest
import torch

import flag_gems

from . import base, consts

BENCH_DTYPES = list(
    dict.fromkeys(
        consts.FLOAT_DTYPES
        + consts.INT_DTYPES
        + consts.EXTRA_INT_DTYPES
        + consts.BOOL_DTYPES
        + consts.COMPLEX_DTYPES
        + [dtype for dtype in consts.FP8_DTYPES if dtype is not None]
    )
)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for extent in reversed(shape):
        stride.append(running)
        running *= extent
    return list(reversed(stride))


def _case_fn(shape, dtype):
    del dtype
    size = list(shape)
    stride = _contiguous_stride(shape)

    yield base.BenchmarkCasePlan(
        shape={"input": size},
        params={"layout": "contiguous"},
        builder_args=(shape, size, stride, 0),
    )
    # reversing both the extents and the strides keeps every element in bounds
    yield base.BenchmarkCasePlan(
        shape={"input": size},
        params={"layout": "reversed"},
        builder_args=(shape, size[::-1], stride[::-1], 0),
    )
    yield base.BenchmarkCasePlan(
        shape={"input": size},
        params={"layout": "flatten"},
        builder_args=(shape, [math.prod(size)], [1], 0),
    )
    if size and size[-1] >= 2:
        yield base.BenchmarkCasePlan(
            shape={"input": size},
            params={"layout": "sub_block"},
            builder_args=(shape, size[:-1] + [size[-1] // 2], stride, 0),
        )
        yield base.BenchmarkCasePlan(
            shape={"input": size},
            params={"layout": "strided"},
            builder_args=(
                shape,
                size[:-1] + [size[-1] - size[-1] // 2],
                stride[:-1] + [2],
                0,
            ),
        )
        # half of the last extent with stride 2 and a shifted origin
        yield base.BenchmarkCasePlan(
            shape={"input": size},
            params={"layout": "dilated"},
            builder_args=(shape, size[:-1] + [size[-1] // 2], stride[:-1] + [2], 1),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, size, stride, storage_offset = plan.builder_args
    inp = torch.empty(shape, dtype=dtype, device=device)
    return inp, size, stride, storage_offset


@pytest.mark.as_strided_
def test_as_strided_():
    bench = base.GenericBenchmark(
        op_name="as_strided_",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.as_strided_,
        gems_op=getattr(flag_gems, "as_strided_", None),
        dtypes=BENCH_DTYPES,
        is_inplace=True,
        fresh_inputs=True,
    )
    bench.run()
