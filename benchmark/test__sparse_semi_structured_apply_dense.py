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

# aten::_sparse_semi_structured_apply_dense(input, threads_masks) selects each
# element of a dense 2-D tensor by its packed mask bit: kept where the bit is 1,
# zero where it is 0. The implementation takes a rank-2 tensor that is either
# RowMajor (stride(1) == 1) or ColMajor (stride(0) == 1), Half/BFloat16 only,
# with tile-aligned extents (rows % 32 == 0, cols % 64 == 0), and a uint8
# (4*ceil(rows/32), 8*ceil(cols/64), 8) mask. Both layout instantiations are
# covered from one tile pair up to 1M elements ((1024, 1024) = 1,048,576).
#
# Case descriptor: ``(rows, cols)`` or ``(rows, cols, layout)`` with layout
# "rowmajor" (default) or "colmajor"; the same descriptor drives listing and
# execution, and it is validated before any tensor is allocated.
_ROWS_MAJOR = "rowmajor"
_COLS_MAJOR = "colmajor"
_LAYOUTS = (_ROWS_MAJOR, _COLS_MAJOR)

_BENCH_SHAPES = [(32, 64), (256, 256), (2048, 64), (1024, 1024), (128, 3840)]

_BENCH_CASES = [
    (rows, cols, layout) for (rows, cols) in _BENCH_SHAPES for layout in _LAYOUTS
]

_TILE_ROWS = 32
_TILE_COLS = 64


def _descriptor(entry):
    """Validate one case descriptor and return ``(rows, cols, layout)``.

    Runs for listing as well as for execution, so an invalid shape is rejected
    before anything is allocated. Extents must be positive because the launch
    grid is derived from them, and tile aligned because each thread covers a full
    8x8 sub-tile while the output is allocated at the exact input size. Unaligned
    and zero-extent shapes are an open native-hazard gap, rejected rather than
    tested.
    """
    if not isinstance(entry, (tuple, list)) or len(entry) not in (2, 3):
        raise ValueError(
            f"case descriptor must be (rows, cols) or (rows, cols, layout), got {entry!r}"
        )
    rows, cols = entry[0], entry[1]
    for name, value in (("rows", rows), ("cols", cols)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer, got {value!r}")
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value!r}")
    if rows % _TILE_ROWS != 0 or cols % _TILE_COLS != 0:
        raise ValueError(
            f"shape must be tile aligned (rows % {_TILE_ROWS} == 0, "
            f"cols % {_TILE_COLS} == 0), got {(rows, cols)}"
        )
    layout = entry[2] if len(entry) == 3 else _ROWS_MAJOR
    if layout not in _LAYOUTS:
        raise ValueError(f"layout must be one of {_LAYOUTS}, got {layout!r}")
    return rows, cols, layout


def _case_fn(descriptor, dtype):
    # ``set_shapes`` feeds each descriptor through case_fn as one case; the plan
    # carries no tensors, so listing allocates nothing and runs no operator.
    del dtype
    rows, cols, layout = _descriptor(descriptor)
    yield base.BenchmarkCasePlan(
        shape={"input": (rows, cols)},
        params={"layout": layout},
        builder_args=(rows, cols, layout),
    )


def _make_mask(rows, cols, device, seed=0):
    """A contract-valid ``threads_masks`` tensor built on the target device."""
    shape = (4 * ((rows + 31) // 32), 8 * ((cols + 63) // 64), 8)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randint(
        0, 256, shape, dtype=torch.uint8, device=device, generator=generator
    )


def _build_inputs_fn(plan, dtype, device):
    rows, cols, layout = plan.builder_args
    inp = utils.generate_tensor_input((rows, cols), dtype, device)
    if layout == _COLS_MAJOR:
        inp = inp.t().contiguous().t()
    return inp, _make_mask(rows, cols, device)


class SparseSemiStructuredApplyDenseBenchmark(OperatorBenchmark):
    """Two-phase GenericBenchmark over aligned 2-D RowMajor/ColMajor inputs."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_BENCH_CASES)


@pytest.mark.sparse_semi_structured_apply_dense
def test__sparse_semi_structured_apply_dense():
    # The operator accepts Half/BFloat16 only; bfloat16 is kept out on a device
    # that does not support it, using the static capability flag rather than a
    # runtime probe.
    dtypes = (
        list(consts.FP16_BF16_DTYPES)
        if flag_gems.runtime.device.support_bf16
        else [torch.float16]
    )
    bench = SparseSemiStructuredApplyDenseBenchmark(
        op_name="_sparse_semi_structured_apply_dense",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._sparse_semi_structured_apply_dense,
        gems_op=getattr(flag_gems, "_sparse_semi_structured_apply_dense", None),
        dtypes=dtypes,
    )
    bench.run()
