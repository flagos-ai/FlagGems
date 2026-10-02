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

# Benchmark for aten::_nested_view_from_buffer. The operator only builds a
# nested view over a flat buffer, so the workload is the component list: a case
# shape is the tuple of component shapes and the buffer length is derived.

import pytest
import torch

import flag_gems

from . import base, consts, utils
from .generated_operator_utils import OperatorBenchmark


def _contiguous_strides(shape):
    strides = [1] * len(shape)
    for dim in range(len(shape) - 2, -1, -1):
        strides[dim] = strides[dim + 1] * max(shape[dim + 1], 1)
    return strides


def _packed_metadata(component_shapes):
    # Packed components (stride 1, no gaps) tile the buffer exactly.
    sizes, strides, offsets = [], [], []
    running = 0
    for shape in component_shapes:
        sizes.append(list(shape))
        strides.append(_contiguous_strides(shape))
        offsets.append(running)
        numel = 1
        for dim in shape:
            numel *= dim
        running += numel
    return sizes, strides, offsets


def _buffer_length(sizes, strides, offsets):
    length = 0
    for size, stride, offset in zip(sizes, strides, offsets):
        if 0 in size:
            continue
        span = sum((dim - 1) * step for dim, step in zip(size, stride))
        length = max(length, offset + span + 1)
    return length


# Each entry is the tuple of component shapes; a file passed with --shape-file
# uses the same literal form, one tuple per line.
NESTED_VIEW_FROM_BUFFER_CASES = [
    ((1,),),
    ((1024,), (1024,), (1024,)),
    ((1000,), (2000,), (3000,)),
    ((16384,),) * 8,
    ((512, 512), (512, 512)),
    ((64, 256), (256, 64)),
    ((20, 320, 15), (20, 320, 15)),
    ((1024, 1024), (1024, 1024)),
    ((4 * 1024 * 1024,),),
]


def _component_shapes(descriptor):
    # Rows above are tuples of component shapes; the shared large shapes are
    # plain dim tuples and describe a single component with that shape.
    if all(isinstance(dim, int) for dim in descriptor):
        return (tuple(descriptor),)
    return tuple(tuple(component) for component in descriptor)


def _case_fn(shape, dtype):
    del dtype
    component_shapes = _component_shapes(shape)
    sizes, strides, offsets = _packed_metadata(component_shapes)
    yield base.BenchmarkCasePlan(
        shape={"buffer": _buffer_length(sizes, strides, offsets)},
        params={
            "component_shapes": [list(component) for component in component_shapes]
        },
        builder_args=(component_shapes,),
    )


def _build_inputs_fn(plan, dtype, device):
    (component_shapes,) = plan.builder_args
    sizes, strides, offsets = _packed_metadata(component_shapes)
    buffer = utils.generate_tensor_input(
        (_buffer_length(sizes, strides, offsets),), dtype, device
    )
    # The metadata is read on the host, so it is created on CPU.
    return (
        buffer,
        torch.tensor(sizes, dtype=torch.int64),
        torch.tensor(strides, dtype=torch.int64),
        torch.tensor(offsets, dtype=torch.int64),
    )


class NestedViewFromBufferBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        for descriptor in NESTED_VIEW_FROM_BUFFER_CASES:
            if descriptor not in self.shapes:
                self.shapes.append(descriptor)


@pytest.mark.nested_view_from_buffer
def test__nested_view_from_buffer():
    bench = NestedViewFromBufferBenchmark(
        op_name="_nested_view_from_buffer",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._nested_view_from_buffer,
        gems_op=getattr(flag_gems, "_nested_view_from_buffer", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
