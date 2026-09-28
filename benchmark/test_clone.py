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
import os

import pytest
import torch

import flag_gems

from . import base, consts, utils

# bfloat16 has no kernel on every backend; this is a static capability query, so
# case listing stays free of device work and tensor allocations.
_BENCH_DTYPES = [
    dtype
    for dtype in consts.FLOAT_DTYPES
    if dtype is not torch.bfloat16 or flag_gems.runtime.device.support_bf16
]

# The shared scale shapes are all rank <= 3, so these rank-4/rank-5 shapes are
# added for the layout family on top of the scale default (and only then).
_FORMAT_SHAPES = [(2, 3, 512, 512), (2, 2, 64, 128, 64)]

_FORMAT_RANKS = {"channels_last": 4, "channels_last_3d": 5}

_MEMORY_FORMATS = {
    "preserve_format": torch.preserve_format,
    "contiguous_format": torch.contiguous_format,
    "channels_last": torch.channels_last,
    "channels_last_3d": torch.channels_last_3d,
}

# (input layout, requested memory format) pairs. "offset", "strided",
# "transposed" and "expanded" describe the memory layout the candidate has to
# gather from; "preserve_format" and "contiguous_format" are explicit calls and
# the None rows omit the argument, which is the schema default. The
# channels_last rows cover both a dense operand that has to be re-laid into the
# format and one that already carries it.
_LAYOUT_ROWS = [
    ("dense", None),
    ("dense", "preserve_format"),
    ("dense", "contiguous_format"),
    ("offset", None),
    ("strided", None),
    ("transposed", None),
    ("transposed", "preserve_format"),
    ("transposed", "contiguous_format"),
    ("expanded", None),
    ("dense", "channels_last"),
    ("channels_last", "channels_last"),
    ("channels_last", None),
    ("dense", "channels_last_3d"),
    ("channels_last_3d", "channels_last_3d"),
    ("channels_last_3d", None),
]

_BENCHMARK_DIR = os.path.dirname(os.path.abspath(base.__file__))

# The shape file the framework itself selects (Benchmark.DEFAULT_SHAPE_FILES,
# which is also conftest's --shape_file default): the copy inside the benchmark
# package, plus the vendor copies base.py substitutes for set_shapes on the
# kunlunxin and enflame backends. Membership is decided on the fully resolved
# path, so a user file that merely shares the basename stays an exact override.
_FRAMEWORK_SHAPE_NAMES = (os.path.basename(base.Benchmark.DEFAULT_SHAPE_FILES),)
_FRAMEWORK_SHAPE_FILES = frozenset(
    os.path.realpath(os.path.join(root, name))
    for root in (
        _BENCHMARK_DIR,
        os.path.join(
            _BENCHMARK_DIR,
            os.pardir,
            "src",
            "flag_gems",
            "runtime",
            "backend",
            "_kunlunxin",
        ),
        os.path.join(
            _BENCHMARK_DIR,
            os.pardir,
            "src",
            "flag_gems",
            "runtime",
            "backend",
            "_enflame",
        ),
    )
    for name in _FRAMEWORK_SHAPE_NAMES
)


def _is_framework_shape_file(shape_file_path):
    """Whether the framework's own shape file, rather than a user file, applies."""
    if shape_file_path is None:
        return True
    # Only resolved paths are classified: a bare "core_shapes.yaml" is resolved
    # by the filesystem exactly like any other relative path, so a user file
    # with that basename keeps its own location and stays an exact override.
    return os.path.realpath(str(shape_file_path)) in _FRAMEWORK_SHAPE_FILES


def _logical_shape(shape):
    """Validate a shape from the case metadata before any plan is generated.

    Extents must be literal non-negative integers: coercing them would silently
    benchmark a different shape than the caller asked for.
    """
    dims = []
    for dim in shape:
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise ValueError(
                f"clone expects integer extents, got {dim!r} in {tuple(shape)!r}."
            )
        if dim < 0:
            raise ValueError(
                f"clone expects non-negative extents, got {dim!r} in {tuple(shape)!r}."
            )
        dims.append(dim)
    return tuple(dims)


def _rank_matches(layout, rank):
    required = _FORMAT_RANKS.get(layout)
    if required is not None:
        return rank == required
    if layout == "transposed":
        return rank >= 2
    return True


def _layout_rows(shape):
    """The layout rows that are meaningful for this logical shape."""
    if math.prod(shape) == 0:
        # An empty workload has no storage pattern to exercise.
        return [("dense", None)]
    rank = len(shape)
    if rank == 0:
        return [("dense", None)]
    rows = [
        row
        for row in _LAYOUT_ROWS
        if _rank_matches(row[0], rank) and _rank_matches(row[1], rank)
    ]
    if rank == 1:
        # A rank-1 tensor has no transpose and only one non-contiguous pattern.
        return [row for row in rows if row[0] in ("dense", "offset")]
    if not any(dim > 1 for dim in shape):
        return [row for row in rows if row[0] != "expanded"]
    return rows


def _strided_stride(shape):
    """Contiguous strides with one element of padding on the leading axis."""
    stride = [1] * len(shape)
    acc = 1
    for axis in range(len(shape) - 1, -1, -1):
        stride[axis] = acc
        acc *= shape[axis]
    stride[0] += 1
    return tuple(stride)


def _case_fn(shape, dtype):
    del dtype
    logical = _logical_shape(shape)
    for input_layout, memory_format in _layout_rows(logical):
        yield base.BenchmarkCasePlan(
            shape={
                "input": logical,
                "input_layout": input_layout,
                "memory_format": memory_format,
            },
            params={"memory_format": memory_format},
            builder_args=(logical, input_layout, memory_format),
        )


def _layout_input(shape, layout, dtype, device):
    """Generate the input in the requested stride pattern on the given device."""
    if layout == "dense":
        return utils.generate_tensor_input(shape, dtype, device)
    if layout == "channels_last":
        return utils.generate_tensor_input(shape, dtype, device).contiguous(
            memory_format=torch.channels_last
        )
    if layout == "channels_last_3d":
        return utils.generate_tensor_input(shape, dtype, device).contiguous(
            memory_format=torch.channels_last_3d
        )
    if layout == "transposed":
        flat = utils.generate_tensor_input(
            (shape[1], shape[0]) + tuple(shape[2:]), dtype, device
        )
        return flat.transpose(0, 1)
    if layout == "offset":
        flat = utils.generate_tensor_input((math.prod(shape) + 1,), dtype, device)
        return flat[1:].view(shape)
    if layout == "strided":
        flat = utils.generate_tensor_input(
            (math.prod(shape) + shape[0],), dtype, device
        )
        return torch.as_strided(flat, shape, _strided_stride(shape))
    # expanded: a size-1 axis expanded to a larger extent gives that axis
    # stride 0, so the candidate has to gather repeated elements.
    axis = next(index for index, dim in enumerate(shape) if dim > 1)
    base_shape = list(shape)
    base_shape[axis] = 1
    flat = utils.generate_tensor_input(tuple(base_shape), dtype, device)
    return flat.expand(shape)


def _build_layout_inputs(plan, dtype, device):
    shape, input_layout, memory_format = plan.builder_args
    kwargs = (
        {}
        if memory_format is None
        else {"memory_format": _MEMORY_FORMATS[memory_format]}
    )
    return _layout_input(shape, input_layout, dtype, device), kwargs


# All three tests benchmark the single public operator `clone`: the candidate is
# resolved through that one name (the out form passes `out=` to the same
# callable), so `op_name` must stay the operator name for the override lookup and
# the case-list identity to resolve. The shared class default metrics are kept,
# including tflops, even though a copy is memory bound.
@pytest.mark.clone
def test_clone():
    bench = base.UnaryPointwiseBenchmark(
        op_name="clone",
        torch_op=torch.ops.aten.clone,
        gems_op=getattr(flag_gems, "clone", None),
        dtypes=_BENCH_DTYPES,
    )
    bench.run()


@pytest.mark.clone
def test_clone_out():
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="clone",
        torch_op=torch.ops.aten.clone,
        gems_op=getattr(flag_gems, "clone", None),
        dtypes=_BENCH_DTYPES,
    )
    bench.run()


class CloneLayoutBenchmark(base.GenericBenchmark):
    """clone over distinct input stride patterns and requested memory formats."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        # Only the framework's own shape file gets the extra rank-4/rank-5
        # shapes; an explicit --shape_file is the complete workload description.
        if not _is_framework_shape_file(shape_file_path):
            return
        for shape in _FORMAT_SHAPES:
            if shape not in self.shapes:
                self.shapes.append(shape)


@pytest.mark.clone
def test_clone_layout():
    bench = CloneLayoutBenchmark(
        op_name="clone",
        torch_op=torch.ops.aten.clone,
        gems_op=getattr(flag_gems, "clone", None),
        case_fn=_case_fn,
        build_inputs_fn=_build_layout_inputs,
        dtypes=_BENCH_DTYPES,
    )
    bench.run()
