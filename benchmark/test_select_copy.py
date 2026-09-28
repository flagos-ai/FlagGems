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

# select_copy moves one slice per call, so its cost tracks numel(shape) //
# shape[dim] and the read pattern of that slice in a row-major input: a
# leading-axis selection is one contiguous block, a middle-axis selection gives
# contiguous rows separated by gaps, and a trailing-axis selection gathers
# single elements with a constant stride. The six requested shape scales are
# kept, and (1024, 32, 1024) carries both a middle- and a trailing-axis plan.
SELECT_COPY_CASES = [
    ((1024, 32, 1024), 1, 16),  # 1_048_576 copied, 1024-element rows with gaps
    ((1024, 32, 1024), 2, 512),  # 32_768 copied, one element per row
    ((64, 512, 512), 0, 32),  # 262_144 copied, single contiguous block
    ((512, 8, 512), 1, 4),  # 262_144 copied, 512-element rows with gaps
    ((64, 512, 8), 2, 4),  # 32_768 copied, one element per row
    ((1024, 4096), 0, 512),  # 4_096 copied, single contiguous block
    ((16, 7, 57, 32, 29), 1, 3),  # 846_336 copied, 16 contiguous blocks
]

# A bare shape is still valid input: OperatorBenchmark.set_shapes normalizes
# nested lists from a shape file to tuples, so a flat tuple of ints stays a
# bare shape and (shape, dim, index) stays an explicit request.
_BARE_AXES = {}
for _shape, _dim, _index in SELECT_COPY_CASES:
    _BARE_AXES.setdefault(_shape, []).append((_dim, _index))

# consts.FLOAT_DTYPES is not capability-gated; drop bf16 only on backends
# without bfloat16 support instead of dropping the whole float family.
_DTYPES = [
    dtype
    for dtype in consts.FLOAT_DTYPES
    if dtype != torch.bfloat16 or flag_gems.runtime.device.support_bf16
]


def _as_int(value, name):
    """Numeric view of dim/index, used only to validate the request. bool is a
    legal int argument for this schema, so it is accepted here."""
    if isinstance(value, bool):
        return int(value)
    if not isinstance(value, int):
        raise ValueError(f"select_copy {name} must be an integer, got {value!r}")
    return value


def _checked_shape(shape):
    """Reject metadata no select can be built from, before any allocation."""
    if not isinstance(shape, (tuple, list)) or not shape:
        raise ValueError(f"select_copy needs a rank >= 1 shape, got {shape!r}")
    for extent in shape:
        if isinstance(extent, bool) or not isinstance(extent, int) or extent < 0:
            raise ValueError(
                f"shape extents must be non-negative integers, got {shape!r}"
            )
    return tuple(shape)


def _validated_axis(shape, dim, index):
    """Validate one axis request against normalized coordinates and return the
    caller's dim/index unchanged, so an explicit negative or bool request is the
    value the operator actually receives instead of a rewritten positive one."""
    norm_dim = _as_int(dim, "dim")
    norm_index = _as_int(index, "index")
    norm_dim = norm_dim + len(shape) if norm_dim < 0 else norm_dim
    if not 0 <= norm_dim < len(shape):
        raise ValueError(f"dim {dim!r} is out of range for shape {shape}")
    extent = shape[norm_dim]
    norm_index = norm_index + extent if norm_index < 0 else norm_index
    if not 0 <= norm_index < extent:
        raise ValueError(f"index {index!r} is out of range for dim {dim!r} of {shape}")
    return dim, index


def _derived_axis(shape):
    """Axis for a caller-supplied bare shape that is not a table entry. Only a
    nonempty axis can be indexed, so an all-zero shape is rejected before any
    allocation instead of yielding a plan that cannot be built."""
    candidates = [axis for axis, extent in enumerate(shape) if extent > 0]
    if not candidates:
        raise ValueError(f"select_copy needs a nonempty axis, got shape {shape}")
    axis = max(candidates, key=lambda candidate: shape[candidate])
    return axis, shape[axis] // 2


def _case_axes(entry):
    """Resolve one shape entry to (shape, [(dim, index), ...])."""
    if (
        isinstance(entry, (tuple, list))
        and entry
        and isinstance(entry[0], (tuple, list))
    ):
        if len(entry) != 3:
            raise ValueError(
                f"select_copy shape entries are (shape, dim, index), got {entry!r}"
            )
        shape = _checked_shape(entry[0])
        return shape, [_validated_axis(shape, entry[1], entry[2])]
    shape = _checked_shape(entry)
    if shape in _BARE_AXES:
        axes = [_validated_axis(shape, dim, index) for dim, index in _BARE_AXES[shape]]
        return shape, axes
    return shape, [_derived_axis(shape)]


def _case_fn(entry, dtype):
    del dtype
    shape, axes = _case_axes(entry)
    for dim, index in axes:
        yield base.BenchmarkCasePlan(
            shape={"input": shape},
            params={"dim": dim, "index": index},
            builder_args=(shape, dim, index),
        )


def _build_inputs_fn(plan, dtype, device):
    # select_copy takes (input, dim, index); the runner turns this flat tuple
    # into three positional arguments.
    shape, dim, index = plan.builder_args
    inp = utils.generate_tensor_input(shape, dtype, device)
    return inp, dim, index


class SelectCopyBenchmark(OperatorBenchmark):
    """select_copy workloads with explicit (shape, dim, index) requests.

    The inherited shape set (a 2**28-element 1-D tensor and other shapes
    unrelated to select volume) is replaced by SELECT_COPY_CASES; a shape file
    still overrides it, and listing and timing share the same plans.
    """

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=SELECT_COPY_CASES)


@pytest.mark.select_copy
def test_select_copy():
    bench = SelectCopyBenchmark(
        op_name="select_copy",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.select_copy,
        gems_op=getattr(flag_gems, "select_copy", None),
        dtypes=_DTYPES,
    )
    bench.run()


@pytest.mark.select_copy
def test_select_copy_case_metadata():
    """Explicit (shape, dim, index) requests must reach the plan unchanged, so a
    negative or bool coordinate is what the operator is called with, while a
    malformed or out-of-range descriptor fails before any tensor is built."""
    plan = next(iter(_case_fn(((3, 4, 5), -2, -1), torch.float32)))
    assert plan.shape == {"input": (3, 4, 5)}
    assert plan.params == {"dim": -2, "index": -1}
    assert plan.builder_args == ((3, 4, 5), -2, -1)

    plan = next(iter(_case_fn(((3, 4, 5), True, True), torch.float32)))
    assert isinstance(plan.params["dim"], bool)
    assert isinstance(plan.params["index"], bool)
    assert plan.params["dim"] is True and plan.params["index"] is True
    assert plan.builder_args[1] is True and plan.builder_args[2] is True

    plan = next(iter(_case_fn(((3, 0, 5), 0, 1), torch.float32)))
    assert plan.shape == {"input": (3, 0, 5)}
    assert plan.params == {"dim": 0, "index": 1}

    # A bare shape still resolves: a table shape keeps every recorded axis and
    # an unknown shape uses one axis derived from its largest nonempty extent.
    assert [plan.params for plan in _case_fn((1024, 32, 1024), torch.float32)] == [
        {"dim": 1, "index": 16},
        {"dim": 2, "index": 512},
    ]
    plan = next(iter(_case_fn((3, 4, 5), torch.float32)))
    assert plan.shape == {"input": (3, 4, 5)}
    assert plan.params == {"dim": 2, "index": 2}

    # Nearest invalid dim / index values, and a descriptor with an unused field.
    invalid_entries = [
        ((3, 4, 5), 3, 0),
        ((3, 4, 5), -4, 0),
        ((3, 4, 5), 0, 3),
        ((3, 4, 5), 0, -4),
        ((3, 4, 5), 0, 1, 7),
    ]
    for entry in invalid_entries:
        with pytest.raises(ValueError):
            list(_case_fn(entry, torch.float32))
