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

import math

import pytest
import torch

import flag_gems

from . import base
from .generated_operator_utils import OperatorBenchmark

# aten::is_non_overlapping_and_dense(Tensor self) -> bool reads only sizes,
# strides and storage_offset, so every case varies the storage scale or the layout
# of one tensor.  The builders allocate uninitialised buffers: no element is ever
# read, and torch.empty covers the dtypes benchmark.utils has no branch for.
# Plans are json-compatible metadata with torch state kept in builder_args, so
# --list-cases allocates nothing and --case-id replays the execution builders.
_METADATA_SHAPES = [
    (),
    (2, 19, 7),
    (256,),
    (1024, 1024),
    (20, 320, 15),
    (16, 128, 64, 60),
    (16, 7, 57, 32, 29),
    (64, 128, 256),
    (4096, 4096),
]

# Static runtime capability flags, read once at import time (no probe call).
_DTYPE_CAPABILITY = {
    torch.bfloat16: "support_bf16",
    torch.float64: "support_fp64",
    torch.complex128: "support_fp64",
    torch.int64: "support_int64",
    torch.float8_e4m3fn: "support_fp8",
    torch.float8_e5m2: "support_fp8",
}


def _dtype_supported(dtype):
    flag_name = _DTYPE_CAPABILITY.get(dtype)
    return flag_name is None or bool(getattr(flag_gems.runtime.device, flag_name))


_CASE_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.int32,
    torch.int64,
    torch.float64,
    torch.bool,
    torch.complex64,
    torch.complex128,
]

BENCH_DTYPES = [dtype for dtype in _CASE_DTYPES if _dtype_supported(dtype)]


def _layout_shape(shape, layout):
    """Metadata shape of the view a layout produces (builder_args keeps the base)."""
    if layout == "transposed":
        return shape[:-2] + (shape[-1], shape[-2])
    if layout == "strided":
        return shape[:-1] + ((shape[-1] + 1) // 2,)
    if layout == "singleton":
        return (1,) + shape[1:]
    if layout == "empty":
        return (0,) + shape[1:]
    return shape


def _layouts(shape):
    yield "contiguous"
    if len(shape) >= 2:
        yield "transposed"
    if shape and shape[-1] > 1:
        yield "strided"
    if shape and all(dim > 0 for dim in shape):
        # stride-0 expansion overlaps; sliced singleton / empty dims and a
        # non-zero storage offset are the predicate's boundary layouts.
        yield "expand_overlap"
        yield "singleton"
        yield "empty"
    yield "offset"


def _case_fn(shape, dtype):
    del dtype
    shape = tuple(shape)
    for layout in _layouts(shape):
        yield base.BenchmarkCasePlan(
            shape={"input": list(_layout_shape(shape, layout))},
            params={"layout": layout},
            builder_args=(shape, layout),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, layout = plan.builder_args
    if layout == "offset":
        flat = torch.empty(
            math.prod(shape) + 1 if shape else 2, dtype=dtype, device=device
        )
        return flat[1:].reshape(shape), {}
    inp = torch.empty(shape, dtype=dtype, device=device)
    if layout == "transposed":
        inp = inp.transpose(-1, -2)
    elif layout == "strided":
        inp = inp[..., ::2]
    elif layout == "singleton":
        inp = inp[:1]
    elif layout == "empty":
        inp = inp[:0]
    elif layout == "expand_overlap":
        inp = inp[:1].expand(shape)
    return inp, {}


class IsNonOverlappingAndDenseBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        # Resolve shapes exactly like the framework (caller shape file, else the
        # shared defaults and their comprehensive extras) and append the metadata
        # shapes, as tuples so deduplication compares equal types.
        super().set_shapes(shape_file_path)
        self.shapes = list(
            dict.fromkeys(
                tuple(shape) for shape in list(self.shapes) + _METADATA_SHAPES
            )
        )


@pytest.mark.is_non_overlapping_and_dense
def test_is_non_overlapping_and_dense():
    bench = IsNonOverlappingAndDenseBenchmark(
        op_name="is_non_overlapping_and_dense",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.is_non_overlapping_and_dense,
        gems_op=getattr(flag_gems, "is_non_overlapping_and_dense", None),
        dtypes=BENCH_DTYPES,
    )
    bench.run()
