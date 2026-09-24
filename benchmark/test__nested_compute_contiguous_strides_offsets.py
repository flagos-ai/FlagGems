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
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from .generated_operator_utils import OperatorBenchmark
from . import base, consts

_OP_NAME = "_nested_compute_contiguous_strides_offsets"

# ``_nested_compute_contiguous_strides_offsets`` starts with an underscore, and
# ``pytest.mark`` refuses to generate a marker via attribute access for such
# names. Register it directly on the MarkGenerator so
# ``@pytest.mark._nested_compute_contiguous_strides_offsets`` and
# ``-m _nested_compute_contiguous_strides_offsets`` both work.
setattr(
    pytest.mark,
    _OP_NAME,
    MarkDecorator(Mark(_OP_NAME, (), {}, _ispytest=True), _ispytest=True),
)

# aten::_nested_compute_contiguous_strides_offsets(Tensor nested_size)
# -> (Tensor, Tensor) computes the contiguous strides and storage offsets of
# each sub-tensor of a nested tensor from its (num_tensors, num_dims) int64
# sizes tensor. The sizes metadata tensor is always a CPU tensor (torch creates
# it that way even for CUDA nested tensors) and the op only touches that
# metadata, so the benchmark measures dispatch plus the small int64 scan rather
# than data movement. int64 is the only dtype the op accepts (non-int64 inputs
# raise RuntimeError), so the benchmark runs int64 cases only. The default
# core_shapes set is dominated by huge dense tensors that are meaningless for a
# (num_tensors, num_dims) layout, so the benchmark restricts itself to
# batch x dim layouts.
_NESTED_SIZE_DTYPES = [torch.int64]

_NESTED_SIZE_SHAPES = [
    (16, 2),
    (64, 3),
    (256, 3),
    (1024, 2),
    (2048, 4),
    (4096, 5),
    (8192, 3),
]


def _case_fn(shape, dtype):
    del dtype
    num_tensors, num_dims = shape
    yield base.BenchmarkCasePlan(
        shape={"nested_size": (num_tensors, num_dims)},
        params={"bound": 8},
        builder_args=((num_tensors, num_dims),),
    )


def _build_inputs_fn(plan, dtype, device):
    # The reference reads the int64 payload through a host pointer, so the
    # sizes metadata must stay on the CPU regardless of the benchmark device.
    del dtype, device
    num_tensors, num_dims = plan.builder_args[0]
    gen = torch.Generator("cpu").manual_seed(0)
    nested_size = torch.randint(
        1,
        plan.params["bound"],
        (num_tensors, num_dims),
        dtype=torch.int64,
        generator=gen,
    )
    return (nested_size,)


class NestedComputeContiguousStridesOffsetsBenchmark(OperatorBenchmark):
    """Two-phase GenericBenchmark restricted to nested-size layout shapes."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_NESTED_SIZE_SHAPES)

    def set_more_shapes(self):
        return []


@pytest.mark._nested_compute_contiguous_strides_offsets
def test__nested_compute_contiguous_strides_offsets():
    bench = NestedComputeContiguousStridesOffsetsBenchmark(
        op_name=_OP_NAME,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._nested_compute_contiguous_strides_offsets,
        gems_op=getattr(flag_gems, _OP_NAME, None),
        dtypes=_NESTED_SIZE_DTYPES,
        metrics=consts.DEFAULT_METRICS,
    )
    bench.run()
