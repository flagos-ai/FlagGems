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

from . import base, consts, utils
from .generated_operator_utils import OperatorBenchmark

# Case descriptors: (input_shape, size, stride). The op copies the requested
# strided view into a fresh dense tensor, so every case is a memory transfer.
# Each source shape appears with the request reproducing its own dense layout and
# with a permuted (transposed) request, which is what reshape hands to this
# operator whenever a contiguous copy is required.
RESHAPE_ALIAS_COPY_CASES = [
    ((1024, 1024), [1024, 1024], [1024, 1]),
    ((1024, 1024), [1024, 1024], [1, 1024]),
    ((4096, 4096), [4096, 4096], [4096, 1]),
    ((4096, 4096), [4096, 4096], [1, 4096]),
    ((64, 512, 512), [64, 512, 512], [262144, 512, 1]),
    ((20, 320, 15), [15, 320, 20], [1, 15, 4800]),
    ((16, 128, 64, 60), [16, 128, 64, 60], [491520, 3840, 60, 1]),
    ((16, 128, 64, 60), [60, 64, 128, 16], [1, 60, 3840, 491520]),
    ((16, 7, 57, 32, 29), [29, 32, 57, 7, 16], [1, 29, 928, 52896, 370272]),
    ((256,), [16, 16], [1, 16]),
]

# Static capability flags only: no tensor is built or probed at import time.
_BENCH_DTYPES = [
    dtype
    for dtype in consts.FLOAT_DTYPES
    if dtype is not torch.bfloat16 or flag_gems.runtime.device.support_bf16
]


def _int_extent(value, what):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{what} must be a non-bool integer, got {value!r}")
    if value < 0:
        raise ValueError(f"{what} must not be negative, got {value}")
    return value


def _decode(case):
    # Validate one descriptor before any allocation: malformed metadata and any
    # request whose elements fall outside the input storage fail here, clearly,
    # instead of reading undefined memory. Valid requests are never clamped,
    # filtered or dropped.
    if not isinstance(case, (tuple, list)) or len(case) != 3:
        raise TypeError(f"case must be (input_shape, size, stride), got {case!r}")
    shape, size, stride = case
    parts = (("input_shape", shape), ("size", size), ("stride", stride))
    for name, sequence in parts:
        if not isinstance(sequence, (tuple, list)):
            raise TypeError(f"{name} must be a sequence of ints, got {sequence!r}")
    shape = tuple(_int_extent(dim, "input_shape dim") for dim in shape)
    size = tuple(_int_extent(dim, "size dim") for dim in size)
    stride = tuple(_int_extent(step, "stride dim") for step in stride)
    if len(size) != len(stride):
        raise ValueError(f"size rank {len(size)} != stride rank {len(stride)}")
    reach = 0
    for dim, step in zip(size, stride):
        if dim > 0:
            reach += (dim - 1) * step
    numel = math.prod(shape)
    if all(dim > 0 for dim in size) and (numel == 0 or reach >= numel):
        raise ValueError(
            f"size={size} stride={stride} reaches element {reach}, outside an "
            f"input storage of {numel} elements"
        )
    return shape, size, stride


def _case_fn(case, dtype):
    # Listing and execution share these plans. The plan's metadata and its
    # builder arguments come from the same validated decode, so a listed shape
    # always matches the tensor the builder allocates.
    del dtype
    shape, size, stride = _decode(case)
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={"size": list(size), "stride": list(stride)},
        builder_args=(shape, size, stride),
    )


def _build_inputs_fn(plan, dtype, device):
    shape, size, stride = plan.builder_args
    inp = utils.generate_tensor_input(shape, dtype, device)
    return inp, size, stride


class ReshapeAliasCopyBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=RESHAPE_ALIAS_COPY_CASES)


@pytest.mark.reshape_alias_copy
def test__reshape_alias_copy():
    bench = ReshapeAliasCopyBenchmark(
        op_name="_reshape_alias_copy",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._reshape_alias_copy,
        gems_op=getattr(flag_gems, "_reshape_alias_copy", None),
        dtypes=_BENCH_DTYPES,
    )
    bench.run()
