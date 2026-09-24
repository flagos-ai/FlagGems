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

# ``_nested_tensor_strides`` starts with an underscore, and ``pytest.mark``
# refuses to generate a marker via attribute access for such names. Register it
# directly on the MarkGenerator so ``@pytest.mark._nested_tensor_strides`` and
# ``-m _nested_tensor_strides`` both work.
setattr(
    pytest.mark,
    "_nested_tensor_strides",
    MarkDecorator(
        Mark("_nested_tensor_strides", (), {}, _ispytest=True), _ispytest=True
    ),
)

# aten::_nested_tensor_strides(Tensor self) -> Tensor reads the (num_tensors,
# num_dims) int64 strides metadata of a strided-layout nested tensor. Its cost is
# proportional to num_tensors * num_dims and independent of the stored values, so
# the benchmark uses (num_tensors, num_dims) layouts instead of the dense
# core_shapes set, which is dominated by huge 1-D tensors that are meaningless
# for a nested-tensor batch.
_NESTED_STRIDES_SHAPES = [
    (16, 2),
    (64, 3),
    (256, 3),
    (1024, 2),
    (2048, 4),
    (4096, 5),
    (8192, 3),
]


def _case_fn(shape, dtype):
    # One (num_tensors, num_dims) layout per Workload; the builder args carry the
    # layout so build_inputs_fn stays dtype-agnostic.
    del dtype
    num_tensors, num_dims = shape
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={},
        builder_args=((num_tensors, num_dims),),
    )


def _build_inputs_fn(plan, dtype, device):
    # build_inputs_fn is the second phase of the two-phase GenericBenchmark: it is
    # called per (case, dtype) with the parsed plan and the test device. Components
    # are ordinary contiguous tensors whose dim 0 is ragged (length in [1, 8]) and
    # whose remaining dims are fixed at 4; their values are irrelevant to the
    # operator, so plain randn input is used here (the correctness file is the one
    # that sweeps the spec value ranges).
    num_tensors, num_dims = plan.builder_args[0]
    gen = torch.Generator("cpu").manual_seed(0)
    lengths = torch.randint(1, 9, (num_tensors,), generator=gen).tolist()
    components = [
        torch.randn((length,) + (4,) * (num_dims - 1), dtype=dtype, device=device)
        for length in lengths
    ]
    return torch.nested.nested_tensor(components, device=device), {}


class NestedTensorStridesBenchmark(OperatorBenchmark):
    """Two-phase GenericBenchmark restricted to (num_tensors, num_dims) shapes.

    ``set_shapes``/``set_more_shapes`` are overridden because the default
    implementations read the dense core_shapes.yaml set, which has no meaning for
    a nested-tensor batch. ``record_shapes`` is overridden because a
    strided-layout NestedTensor does not support ``Tensor.size()`` (it raises
    "NestedTensorImpl doesn't support sizes"); the strides metadata tensor
    returned by the operator itself is used as the shape descriptor instead.
    """

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_NESTED_STRIDES_SHAPES)

    def set_more_shapes(self):
        return []

    def record_shapes(self, *args, **kwargs):
        def deep_parse(item):
            if isinstance(item, torch.Tensor):
                if item.is_nested:
                    return ("nested",) + tuple(
                        torch.ops.aten._nested_tensor_strides(item).shape
                    )
                return item.size()
            elif isinstance(item, (int, float, str, torch.dtype)):
                return item
            elif isinstance(item, (list, tuple)):
                return [deep_parse(sub_item) for sub_item in item]
            elif isinstance(item, dict):
                return {key: deep_parse(value) for key, value in item.items()}
            return None

        parsed_args = [deep_parse(arg) for arg in args]
        parsed_kwargs = {key: deep_parse(value) for key, value in kwargs.items()}
        if parsed_args and parsed_kwargs:
            return parsed_args, parsed_kwargs
        return parsed_args if parsed_args else parsed_kwargs


@pytest.mark._nested_tensor_strides
def test__nested_tensor_strides():
    # torch_op is the perf comparison reference; gems_op is the candidate under
    # test. They share the same single-argument call semantics
    # (Tensor) -> Tensor. The candidate name starts with an underscore, so the
    # default is fetched with getattr to stay importable before the kernel is
    # registered; KernelGen's injected candidate is visible here at call time.
    bench = NestedTensorStridesBenchmark(
        op_name="_nested_tensor_strides",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._nested_tensor_strides,
        gems_op=getattr(flag_gems, "_nested_tensor_strides", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
