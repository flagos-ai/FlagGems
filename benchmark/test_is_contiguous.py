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

# is_contiguous is metadata-only and has no core_shapes.yaml entry, so the
# default shape set is expressed here: scalar, empty, 1-D to 5-D boundaries.
# Every plan is a (layout, memory_format) pair; a user --shape_file still wins.

import pytest
import torch

import flag_gems

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

CORE_SHAPES = list(
    dict.fromkeys(
        consts.DEFAULT_SHAPES
        + [
            (),
            (1,),
            (0, 3),
            (256,),
            (2, 19, 7),
            (2, 3, 4, 5),
            (2, 3, 4, 5, 6),
            (1024, 1024),
            (20, 320, 15),
            (16, 128, 64, 60),
            (16, 7, 57, 32, 29),
        ]
    )
)

_MEMORY_FORMATS = {
    "contiguous": torch.contiguous_format,
    "preserve": torch.preserve_format,
    "channels_last": torch.channels_last,
    "channels_last_3d": torch.channels_last_3d,
}


def _layout_kinds(shape):
    # Rank-only metadata, so case listing allocates no tensor.
    rank = len(shape)
    kinds = ["plain"]
    if rank >= 2:
        kinds.append("transposed")
    if rank >= 1:
        kinds.append("column_step")
    if rank == 4:
        kinds.append("channels_last")
    elif rank == 5:
        kinds.append("channels_last_3d")
    return kinds


def _view_shape(layout, shape):
    # Exact shape of the tensor handed to the operator (builder_args keeps the
    # storage shape private).
    if layout == "transposed":
        return shape[:-2] + (shape[-1], shape[-2])
    if layout == "column_step":
        return shape[:-1] + ((shape[-1] + 1) // 2,)
    return shape


def _case_fn(shape, dtype):
    del dtype
    shape = tuple(shape)
    for layout in _layout_kinds(shape):
        for name in ("default", *_MEMORY_FORMATS):
            yield base.BenchmarkCasePlan(
                shape={"input": list(_view_shape(layout, shape))},
                params={"layout": layout, "memory_format": name},
                builder_args=(layout, shape),
            )


def _build_layout(layout, shape, dtype, device):
    # generate_tensor_input cannot express views or memory formats, so the
    # tested layout is applied on top of it.
    inp = torch.empty(shape, dtype=dtype, device=device)
    if layout == "plain":
        return inp
    if layout == "transposed":
        return inp.transpose(-2, -1)
    if layout == "column_step":
        return inp[..., ::2]
    if layout == "channels_last":
        return inp.to(memory_format=torch.channels_last)
    if layout == "channels_last_3d":
        return inp.to(memory_format=torch.channels_last_3d)
    raise ValueError(f"unknown layout: {layout}")


def _build_inputs_fn(plan, dtype, device):
    layout, shape = plan.builder_args
    inp = _build_layout(layout, shape, dtype, device)
    name = plan.params["memory_format"]
    if name == "default":
        return inp, {}
    return inp, {"memory_format": _MEMORY_FORMATS[name]}


class IsContiguousBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=CORE_SHAPES)


@pytest.mark.is_contiguous
def test_is_contiguous():
    bench = IsContiguousBenchmark(
        op_name="is_contiguous",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.is_contiguous,
        gems_op=getattr(flag_gems, "is_contiguous", None),
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES + consts.BOOL_DTYPES,
    )
    bench.run()
