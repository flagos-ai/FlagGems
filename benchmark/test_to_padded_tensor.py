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

from . import base, consts, utils
from .generated_operator_utils import OperatorBenchmark

# aten::to_padded_tensor(Tensor self, float padding, SymInt[]? output_size=None)
# densifies a nested tensor; the measured work is the ragged copy plus the
# padding fill. A benchmark descriptor is (layout, lengths, trailing): the input
# is a values buffer of shape (sum(lengths), *trailing) split into ragged rows
# and wrapped in a nested tensor. The plan carries only JSON metadata, so
# --list-cases never allocates a tensor.
_BENCH_PADDING = 3.0
_LAYOUTS = ("jagged", "strided")
_BENCH_CASES = [
    ("jagged", [8, 4], [1024, 1024]),
    ("jagged", [32, 16, 8], [256, 256]),
    ("jagged", [4, 8, 2, 6], [16, 128, 64]),
    ("strided", [8, 4], [20, 320, 15]),
    ("jagged", [0, 0], []),
    ("jagged", [0, 0], [2]),
    ("jagged", [2, 3], [0]),
]

# Keep the harness dtype set, minus bfloat16 where the device does not report
# bf16 support. This is a static device capability, not a runtime probe.
_BENCH_DTYPES = [
    dtype
    for dtype in consts.FLOAT_DTYPES
    if dtype != torch.bfloat16 or flag_gems.runtime.device.support_bf16
]


def _as_dim(value, what):
    # Accept the built-in descriptor table and an equivalent --shape_file entry.
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{what} must be an integer, got {value!r}")
    text = str(value)
    if not text.lstrip("-").isdigit():
        raise ValueError(f"{what} must be an integer, got {value!r}")
    return int(text)


def _normalize_case(case):
    if not isinstance(case, (list, tuple)) or len(case) != 3:
        raise ValueError(f"shape entry must be (layout, lengths, trailing): {case!r}")
    layout, lengths, trailing = case
    if layout not in _LAYOUTS:
        raise ValueError(f"layout must be one of {_LAYOUTS}, got {layout!r}")
    row_lengths = tuple(_as_dim(value, "ragged length") for value in lengths)
    trailing_dims = tuple(_as_dim(value, "trailing dim") for value in trailing)
    if not row_lengths or any(length < 0 for length in row_lengths):
        raise ValueError(f"ragged lengths must be non-negative: {row_lengths!r}")
    if any(dim < 0 for dim in trailing_dims):
        raise ValueError(f"trailing dims must be non-negative: {trailing_dims!r}")
    if layout == "strided" and (sum(row_lengths) == 0 or 0 in trailing_dims):
        raise ValueError("strided nested input needs a non-empty constituent")
    return layout, row_lengths, trailing_dims


def _case_fn(case, dtype):
    del dtype
    layout, lengths, trailing = _normalize_case(case)
    yield base.BenchmarkCasePlan(
        shape={
            "layout": layout,
            "values": [sum(lengths), *trailing],
            "lengths": list(lengths),
            "padded": [len(lengths), max(lengths), *trailing],
        },
        params={"padding": _BENCH_PADDING},
        builder_args=(layout, lengths, trailing),
    )


def _build_inputs_fn(plan, dtype, device):
    layout, lengths, trailing = plan.builder_args
    values = utils.generate_tensor_input(
        (sum(lengths),) + tuple(trailing), dtype, device
    )
    parts = list(torch.split(values, list(lengths)))
    if layout == "jagged":
        nested = torch.nested.nested_tensor(parts, layout=torch.jagged)
    else:
        nested = torch.nested.nested_tensor(parts)
    return nested, {"padding": plan.params["padding"]}


class ToPaddedTensorBenchmark(OperatorBenchmark):
    # Two-phase benchmark over the jagged and strided nested layouts.

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        rows = []
        for shape in list(self.shapes) + _BENCH_CASES:
            if shape and isinstance(shape[0], str):
                row = (shape[0], tuple(shape[1]), tuple(shape[2]))
            else:
                # Split the requested values buffer along its leading axis;
                # preserve all elements and use the native strided nested layout.
                extents = tuple(shape) or (1,)
                count = extents[0]
                lengths = (count // 2, count - count // 2) if count > 1 else (count,)
                row = ("strided", lengths, extents[1:])
            if row not in rows:
                rows.append(row)
        self.shapes = rows


@pytest.mark.to_padded_tensor
def test_to_padded_tensor():
    bench = ToPaddedTensorBenchmark(
        op_name="to_padded_tensor",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.to_padded_tensor,
        gems_op=getattr(flag_gems, "to_padded_tensor", None),
        dtypes=_BENCH_DTYPES,
    )
    bench.run()
