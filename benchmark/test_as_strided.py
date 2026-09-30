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

"""Benchmark for ``aten::as_strided``.

as_strided is a zero-copy metadata operator: the measured work is dispatch plus
the metadata write, never element traffic. The shared shape grid is therefore
used unchanged (``core_shapes.yaml`` has no ``as_strided`` entry, so the
``Benchmark`` default shapes apply and COMPREHENSIVE mode adds the generic extra
shapes); a caller-supplied shape file still overrides it.

Each shape expands into metadata-only layouts: the identity metadata, a stride
relayout, a stride-0 overlap (the layout a broadcast operand would produce), an
absolute storage offset, a zero-extent view and, where the storage has elements,
a 0-dim collapse. Inputs are allocated with ``torch.empty`` because a
metadata-only view never reads the values, so every supported dtype can be
benchmarked without a value generator. Plan metadata holds integers only, so
``--list-cases`` allocates no tensor and never calls the operator, and the
reference and the candidate receive the same flat positional
``(input, size, stride, storage_offset)`` call.
"""

import pytest
import torch

import flag_gems

from . import base, consts


def _contiguous_strides(size):
    strides = [1] * len(size)
    for dim in range(len(size) - 2, -1, -1):
        strides[dim] = strides[dim + 1] * size[dim + 1]
    return strides


def _layouts(shape):
    """Metadata-only view layouts that fit entirely inside ``shape``'s storage.

    ``overlap`` zeroes the leading stride, ``offset`` starts one element into
    the storage from the end of the last dimension, ``empty`` requests a
    zero-extent view, ``scalar`` collapses the first element to a 0-dim tensor.
    A layout whose requested span cannot fit the storage is skipped.
    """
    rank = len(shape)
    contiguous = _contiguous_strides(shape)
    layouts = [("contiguous", list(shape), contiguous, 0)]
    if rank >= 2:
        size = list(reversed(shape))
        layouts.append(("relayout", size, _contiguous_strides(size), 0))
    if rank >= 1:
        layouts.append(("overlap", list(shape), [0] + contiguous[1:], 0))
        if shape[-1] >= 2:
            layouts.append(
                ("offset", list(shape[:-1]) + [shape[-1] - 1], contiguous, 1)
            )
        layouts.append(("empty", list(shape[:-1]) + [0], contiguous, 0))
        if all(extent > 0 for extent in shape):
            layouts.append(("scalar", [], [], 0))
    return layouts


def _case_fn(shape, dtype):
    del dtype
    storage = list(shape)
    for layout, size, stride, offset in _layouts(storage):
        yield base.BenchmarkCasePlan(
            shape={"storage": storage, "view": size},
            params={"layout": layout, "storage_offset": offset},
            builder_args=(storage, size, stride, offset),
        )


def _build_inputs_fn(plan, dtype, device):
    storage, size, stride, offset = plan.builder_args
    inp = torch.empty(tuple(storage), dtype=dtype, device=device)
    return inp, size, stride, offset


# A metadata rewrite addresses any dtype whose storage the backend holds, so the
# shared dtype constants are combined instead of benchmarking floats alone.
# int8/uint8/int64 are required coverage; fp8 is added only where the backend
# exposes it (``consts.FP8_DTYPES`` is ``[None]`` off CUDA).
_BENCH_DTYPES = (
    consts.FLOAT_DTYPES
    + consts.INT_DTYPES
    + consts.EXTRA_INT_DTYPES
    + consts.BOOL_DTYPES
    + consts.COMPLEX_DTYPES
    + [dtype for dtype in consts.FP8_DTYPES if dtype is not None]
)


@pytest.mark.as_strided
def test_as_strided():
    bench = base.GenericBenchmark(
        op_name="as_strided",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.as_strided,
        gems_op=getattr(flag_gems, "as_strided", None),
        dtypes=_BENCH_DTYPES,
    )
    bench.run()
