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

import bisect
import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

_DENSE_DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.int64,
]


def _make_offsets(batch_size, max_lengths):
    parent_count = batch_size
    all_offsets = []
    for level, max_length in enumerate(max_lengths):
        lengths = [
            (parent * (level + 2) + level + 1) % (max_length + 1)
            for parent in range(parent_count)
        ]
        if parent_count > 0 and max_length > 0 and sum(lengths) == 0:
            lengths[0] = max_length
        values = [0]
        for length in lengths:
            values.append(values[-1] + length)
        all_offsets.append(values)
        parent_count = values[-1]
    return all_offsets


def _make_dense(shape, dtype):
    numel = math.prod(shape)
    values = torch.arange(numel, dtype=torch.float64).remainder(251).reshape(shape)
    return values.to(dtype)


def _reference(dense, offsets):
    total_L = offsets[-1][-1]
    inner_size = dense.size(-1)
    output = torch.empty((total_L, inner_size), dtype=dense.dtype)
    for output_row in range(total_L):
        tree_offset = output_row
        coordinates = []
        for level in range(len(offsets) - 1, -1, -1):
            parent = bisect.bisect_right(offsets[level], tree_offset) - 1
            coordinates.append(tree_offset - offsets[level][parent])
            tree_offset = parent
        coordinates.reverse()
        output[output_row] = dense[(tree_offset, *coordinates, slice(None))]
    return output


def _run_case(dense_cpu, offsets_cpu, offset_dtype, total_L):
    dense = dense_cpu.to(flag_gems.device)
    offsets = [
        torch.tensor(values, dtype=offset_dtype, device=flag_gems.device)
        for values in offsets_cpu
    ]
    expected = _reference(dense_cpu, offsets_cpu)

    actual = flag_gems._padded_dense_to_jagged_forward(dense, offsets, total_L)

    assert actual.device == dense.device
    assert actual.dtype == dense.dtype
    assert actual.shape == expected.shape
    utils.gems_assert_equal(actual.cpu(), expected)


@pytest.mark.padded_dense_to_jagged_forward
@pytest.mark.parametrize("dtype", _DENSE_DTYPES)
@pytest.mark.parametrize("offset_dtype", [torch.int32, torch.int64])
def test_padded_dense_to_jagged_forward_single(dtype, offset_dtype):
    shape = (7, 9, 17)
    offsets = _make_offsets(shape[0], shape[1:-1])
    _run_case(
        _make_dense(shape, dtype),
        offsets,
        offset_dtype,
        offsets[-1][-1],
    )


@pytest.mark.padded_dense_to_jagged_forward
@pytest.mark.parametrize(
    "shape",
    [
        (3, 4, 3, 12),
        (2, 3, 3, 2, 7),
        (2, 2, 3, 2, 2, 5),
        (2, 2, 2, 2, 2, 2, 3),
    ],
)
@pytest.mark.parametrize("dtype", _DENSE_DTYPES)
@pytest.mark.parametrize("offset_dtype", [torch.int32, torch.int64])
def test_padded_dense_to_jagged_forward_multi(shape, dtype, offset_dtype):
    offsets = _make_offsets(shape[0], shape[1:-1])
    _run_case(
        _make_dense(shape, dtype),
        offsets,
        offset_dtype,
        offsets[-1][-1],
    )


@pytest.mark.padded_dense_to_jagged_forward
def test_padded_dense_to_jagged_forward_infers_terminal_total_L():
    shape = (5, 7, 13)
    offsets = _make_offsets(shape[0], shape[1:-1])
    _run_case(_make_dense(shape, torch.float32), offsets, torch.int64, None)


@pytest.mark.padded_dense_to_jagged_forward
def test_padded_dense_to_jagged_forward_empty_output():
    shape = (4, 0, 8)
    offsets = [[0, 0, 0, 0, 0]]
    _run_case(_make_dense(shape, torch.float32), offsets, torch.int64, 0)


@pytest.mark.padded_dense_to_jagged_forward
def test_padded_dense_to_jagged_forward_noncontiguous_dense():
    shape = (4, 6, 9)
    base = _make_dense((shape[0], shape[1], shape[2] * 2), torch.float32)
    dense_cpu = base[..., ::2]
    assert not dense_cpu.is_contiguous()
    offsets = _make_offsets(shape[0], shape[1:-1])
    _run_case(dense_cpu, offsets, torch.int64, offsets[-1][-1])


@pytest.mark.padded_dense_to_jagged_forward
def test_padded_dense_to_jagged_forward_float64_preserves_bits():
    bit_patterns = torch.tensor(
        [
            0,
            -(2**63),
            0x7FF0000000000000,
            -0x10000000000000,
            0x7FF8000000000001,
            0x7FF123456789ABCD,
            1,
            -1,
            0x3FF0000000000000,
            -0x4010000000000000,
            0x0010000000000000,
            0x0000000000000001,
        ],
        dtype=torch.int64,
    ).reshape(2, 3, 2)
    dense_cpu = bit_patterns.view(torch.float64)
    offsets_cpu = [[0, 2, 3]]
    dense = dense_cpu.to(flag_gems.device)
    offsets = [torch.tensor(offsets_cpu[0], device=flag_gems.device)]
    expected = _reference(dense_cpu, offsets_cpu)

    actual = flag_gems._padded_dense_to_jagged_forward(dense, offsets, 3)

    assert torch.equal(actual.cpu().view(torch.int64), expected.view(torch.int64))


@pytest.mark.padded_dense_to_jagged_forward
@pytest.mark.parametrize("inner_size", [4, 6])
@pytest.mark.parametrize("offset_dtype", [torch.int32, torch.int64])
def test_padded_dense_to_jagged_forward_float16_preserves_bits(
    inner_size, offset_dtype
):
    patterns = torch.tensor(
        [0, -(2**15), 0x7C00, -0x400, 0x7E01, 0x7D55, 1, -1],
        dtype=torch.int16,
    )
    numel = 2 * 3 * inner_size
    bit_patterns = patterns.repeat(math.ceil(numel / patterns.numel()))[:numel]
    bit_patterns = bit_patterns.reshape(2, 3, inner_size)
    dense = bit_patterns.view(torch.float16).to(flag_gems.device)
    offsets = [torch.tensor([0, 2, 3], dtype=offset_dtype, device=flag_gems.device)]
    expected_bits = torch.cat((bit_patterns[0, :2], bit_patterns[1, :1]))

    actual = flag_gems._padded_dense_to_jagged_forward(dense, offsets, 3)

    assert torch.equal(actual.cpu().view(torch.int16), expected_bits)


def _invalid_inputs(case):
    dense = torch.zeros((2, 2, 3, 4), device=flag_gems.device)
    valid = [[0, 2, 3], [0, 1, 3, 4]]
    offsets = [
        torch.tensor(values, dtype=torch.int64, device=flag_gems.device)
        for values in valid
    ]
    total_L = 4

    if case == "empty_list":
        offsets = []
    elif case == "count":
        offsets = offsets[:1]
    elif case == "rank":
        offsets[0] = offsets[0].reshape(1, -1)
    elif case == "empty":
        offsets[0] = torch.empty(0, dtype=torch.int64, device=flag_gems.device)
    elif case == "start":
        offsets[0] = torch.tensor([1, 2, 3], dtype=torch.int64, device=flag_gems.device)
    elif case == "nondecreasing":
        offsets[0] = torch.tensor([0, 2, 1], dtype=torch.int64, device=flag_gems.device)
    elif case == "hierarchy":
        offsets[0] = torch.tensor([0, 2, 4], dtype=torch.int64, device=flag_gems.device)
    elif case == "padded_limit":
        offsets[0] = torch.tensor([0, 3, 3], dtype=torch.int64, device=flag_gems.device)
    elif case == "deep_padded_limit":
        offsets[1] = torch.tensor(
            [0, 4, 4, 4], dtype=torch.int64, device=flag_gems.device
        )
    elif case == "batch":
        offsets[0] = torch.tensor([0, 1], dtype=torch.int64, device=flag_gems.device)
    elif case == "total_L":
        total_L = 5
    else:
        raise AssertionError(f"unknown invalid-input case: {case}")
    return dense, offsets, total_L


@pytest.mark.padded_dense_to_jagged_forward
@pytest.mark.parametrize(
    "case,match",
    [
        ("empty_list", "non-empty list"),
        ("count", "expected 2 offsets"),
        ("rank", "one-dimensional"),
        ("empty", "non-empty"),
        ("start", "start at 0"),
        ("nondecreasing", "nondecreasing"),
        ("hierarchy", "number of segments"),
        ("padded_limit", "segment longer"),
        ("deep_padded_limit", "segment longer"),
        ("batch", "dense.size\\(0\\)"),
        ("total_L", "terminal value"),
    ],
)
def test_padded_dense_to_jagged_forward_validation(case, match):
    dense, offsets, total_L = _invalid_inputs(case)
    with pytest.raises((RuntimeError, TypeError), match=match):
        flag_gems._padded_dense_to_jagged_forward(dense, offsets, total_L)


@pytest.mark.padded_dense_to_jagged_forward
def test_padded_dense_to_jagged_forward_revalidates_mutated_offsets():
    dense, offsets, total_L = _invalid_inputs("total_L")
    total_L = 4
    flag_gems._padded_dense_to_jagged_forward(dense, offsets, total_L)

    offsets[0][1] = -1
    with pytest.raises(RuntimeError, match="nondecreasing"):
        flag_gems._padded_dense_to_jagged_forward(dense, offsets, total_L)
