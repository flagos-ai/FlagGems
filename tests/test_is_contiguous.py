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

# aten::is_contiguous reads size/stride/storage_offset only, so it has no
# kernel, no broadcast operand and no backward, and element values never change
# the verdict. Coverage is layout state (views, offsets, overlapping strides,
# empty and singleton tensors, lazy conj/neg bits) plus the memory_format
# overload, with the 5-range x 7-shape grid labelling the same layouts.

import pytest
import torch

import flag_gems

from . import test_utils as tu

SUPPORTED_DTYPES = list(tu.REQUIRED_DTYPES) + [
    torch.float64,
    torch.bool,
    torch.complex64,  # exercises the lazy conjugate-bit rows
]

# (label, kind, base shape). All rows are metadata-cheap, so both modes collect
# them; quick mode only shrinks the value-range/shape grid and drops the
# positive special-value rows.
_LAYOUT_ROWS = [
    ("plain_2d", "plain", (4, 6)),
    ("scalar", "plain", ()),
    ("empty_1d", "plain", (0,)),
    ("empty_2d", "plain", (0, 3)),
    ("empty_transposed", "transposed", (0, 3)),
    ("transposed", "transposed", (4, 6)),
    ("transposed_singleton", "transposed", (1, 6)),
    ("row_step", "row_step", (8, 6)),
    ("column_step", "column_step", (4, 12)),
    ("offset_slice", "offset_slice", (8, 12)),
    ("unit_offset_1d", "unit_offset_1d", (16,)),
    ("unit_row", "unit_row", (8, 12)),
    ("expanded", "expanded", (1, 1)),
    ("expanded_1d", "expanded_1d", (1,)),
    ("expanded_singleton", "expanded_singleton", (1, 1)),
    ("channels_last", "channels_last", (2, 3, 4, 5)),
    ("channels_last_3d", "channels_last_3d", (2, 3, 4, 5, 6)),
    ("transposed_3d", "transposed_3d", (2, 3, 4)),
    ("transposed_4d", "transposed_4d", (2, 3, 4, 5)),
    ("singleton_trailing", "plain", (1, 4, 1)),
    ("singleton_leading", "plain", (1, 4, 6)),
    ("conj", "conj", (4, 6)),
    ("conj_transposed", "conj_transposed", (4, 6)),
    ("neg_view", "neg_view", (4, 6)),
]

# (label, kind, base shape, memory_format). channels_last only accepts rank 4
# and channels_last_3d only rank 5 natively, so those rows use matching ranks.
_MEMORY_FORMAT_ROWS = [
    ("plain_contiguous", "plain", (4, 6), torch.contiguous_format),
    ("plain_preserve", "plain", (4, 6), torch.preserve_format),
    ("scalar_contiguous", "plain", (), torch.contiguous_format),
    ("empty_contiguous", "plain", (0,), torch.contiguous_format),
    ("transposed_contiguous", "transposed", (4, 6), torch.contiguous_format),
    ("transposed_preserve", "transposed", (4, 6), torch.preserve_format),
    ("column_step_preserve", "column_step", (4, 12), torch.preserve_format),
    ("offset_slice_preserve", "offset_slice", (8, 12), torch.preserve_format),
    ("expanded_contiguous", "expanded", (1, 1), torch.contiguous_format),
    ("unit_offset_preserve", "unit_offset_1d", (16,), torch.preserve_format),
    ("channels_last_query", "channels_last", (2, 3, 4, 5), torch.channels_last),
    (
        "channels_last_3d_query",
        "channels_last_3d",
        (2, 3, 4, 5, 6),
        torch.channels_last_3d,
    ),
    ("plain_channels_last_query", "plain", (2, 3, 4, 5), torch.channels_last),
    (
        "plain_channels_last_3d_query",
        "plain",
        (2, 3, 4, 5, 6),
        torch.channels_last_3d,
    ),
]

# Positive special-value rows are default-only.
_SPECIAL_VALUE_CASES = tu.selected_cases(
    [
        (dtype, scenario, kind)
        for dtype, scenario in tu.special_value_cases(SUPPORTED_DTYPES)
        for kind in ("plain", "step")
    ],
    quick=[],
)


def _input_state(inp):
    return (
        tuple(inp.shape),
        inp.stride(),
        inp.storage_offset(),
        inp.data_ptr(),
        inp.untyped_storage().data_ptr(),
        inp.requires_grad,
        inp.is_conj(),
        inp.is_neg(),
    )


def _make_layout_input(kind, shape, dtype):
    # Values are decorative for this predicate; only the layout matters.
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    if kind == "plain":
        return inp
    if kind in ("transposed", "transposed_singleton", "empty_transposed"):
        return inp.t()
    if kind == "row_step":
        return inp[::2]
    if kind == "column_step":
        return inp[:, ::2]
    if kind == "offset_slice":
        return inp[2:6, 1:5]
    if kind == "unit_offset_1d":
        return inp[5:]
    if kind == "unit_row":
        return inp[3]
    if kind == "expanded":
        return inp.expand(3, 5)
    if kind == "expanded_1d":
        return inp.expand(7)
    if kind == "expanded_singleton":
        return inp.expand(1, 1)
    if kind == "channels_last":
        return inp.to(memory_format=torch.channels_last)
    if kind == "channels_last_3d":
        return inp.to(memory_format=torch.channels_last_3d)
    if kind == "transposed_3d":
        return inp.transpose(0, 2)
    if kind == "transposed_4d":
        return inp.transpose(2, 3)
    if kind == "conj":
        return inp.conj()
    if kind == "conj_transposed":
        return inp.t().conj()
    if kind == "neg_view":
        # Lazy negation bit, set without an arithmetic op.
        return torch._neg_view(inp)
    raise ValueError(f"unknown layout kind: {kind}")


@pytest.mark.is_contiguous
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_is_contiguous(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref = torch.ops.aten.is_contiguous(tu.to_reference(inp))

    res = flag_gems.is_contiguous(inp)

    assert type(res) is bool
    assert res == ref


@pytest.mark.is_contiguous
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
@pytest.mark.parametrize(
    "kind,shape",
    [(kind, shape) for _, kind, shape in _LAYOUT_ROWS],
    ids=[label for label, _, _ in _LAYOUT_ROWS],
)
def test_is_contiguous_layout(kind, shape, dtype):
    inp = _make_layout_input(kind, shape, dtype)
    ref = torch.ops.aten.is_contiguous(tu.to_reference(inp))

    res = flag_gems.is_contiguous(inp)

    assert type(res) is bool
    assert res == ref


@pytest.mark.is_contiguous
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
@pytest.mark.parametrize(
    "kind,shape,memory_format",
    [(kind, shape, fmt) for _, kind, shape, fmt in _MEMORY_FORMAT_ROWS],
    ids=[label for label, _, _, _ in _MEMORY_FORMAT_ROWS],
)
def test_is_contiguous_memory_format(kind, shape, memory_format, dtype):
    inp = _make_layout_input(kind, shape, dtype)
    ref = torch.ops.aten.is_contiguous.memory_format(
        tu.to_reference(inp), memory_format
    )

    res = flag_gems.is_contiguous(inp, memory_format)

    assert type(res) is bool
    assert res == ref


@pytest.mark.is_contiguous
@pytest.mark.parametrize("dtype,scenario,kind", _SPECIAL_VALUE_CASES)
def test_is_contiguous_special_values(dtype, scenario, kind):
    payload = tu.make_special_input(dtype, scenario)
    inp = payload if kind == "plain" else payload[::2]
    ref = torch.ops.aten.is_contiguous(tu.to_reference(inp))

    res = flag_gems.is_contiguous(inp)

    assert type(res) is bool
    assert res == ref


@pytest.mark.is_contiguous
def test_is_contiguous_keeps_input_state():
    inp = tu.make_input(torch.float32, (8, 12), ["-1", "1"])[:, ::2]
    ref = torch.ops.aten.is_contiguous(tu.to_reference(inp))
    state_before = _input_state(inp)

    res = flag_gems.is_contiguous(inp)

    assert _input_state(inp) == state_before
    assert type(res) is bool
    assert res == ref


@pytest.mark.is_contiguous
def test_is_contiguous_rejects_non_tensor():
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.is_contiguous("not a tensor")


@pytest.mark.is_contiguous
def test_is_contiguous_rejects_extra_argument():
    inp = tu.make_input(torch.float32, (4, 6), ["-1", "1"])

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.is_contiguous(inp, torch.contiguous_format, 1)


@pytest.mark.is_contiguous
@pytest.mark.parametrize("memory_format", [None, "x", 1.5, torch.float32])
def test_is_contiguous_rejects_invalid_memory_format(memory_format):
    # Unknown integer memory_format codes are accepted natively, so only
    # non-integer values are negative cases.
    inp = tu.make_input(torch.float32, (4, 6), ["-1", "1"])

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.is_contiguous(inp, memory_format)
