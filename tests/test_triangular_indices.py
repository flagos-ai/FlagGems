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

from . import accuracy_utils as utils

INT32_MAX = (1 << 31) - 1
INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1

CASES = [
    (0, 0, 0),
    (0, 7, -10),
    (7, 0, 10),
    (1, 1, 0),
    (3, 5, 0),
    (5, 3, 0),
    (4, 7, -2),
    (4, 7, 2),
    (9, 13, -20),
    (9, 13, 20),
    (129, 257, 0),
]


@pytest.mark.parametrize("op_name", ["tril_indices", "triu_indices"])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("row,col,offset", CASES)
def test_triangular_indices(op_name, dtype, row, col, offset):
    torch_op = getattr(torch, op_name)
    gems_op = getattr(flag_gems, op_name)

    reference = torch_op(row, col, offset, dtype=dtype, device="cpu")
    result = gems_op(
        row,
        col,
        offset,
        dtype=dtype,
        layout=torch.strided,
        device=flag_gems.device,
    )

    utils.gems_assert_equal(result.cpu(), reference)
    assert result.shape == reference.shape
    assert result.dtype is dtype
    assert result.is_contiguous()
    assert not result.requires_grad


@pytest.mark.parametrize("op_name", ["tril_indices", "triu_indices"])
def test_triangular_indices_default_options_and_registration(op_name):
    gems_op = getattr(flag_gems, op_name)
    direct = gems_op(5, 7, -1, device=flag_gems.device)

    with flag_gems.use_gems():
        registered = getattr(torch, op_name)(5, 7, -1, device=flag_gems.device)

    reference = getattr(torch, op_name)(5, 7, -1, device="cpu")
    utils.gems_assert_equal(direct.cpu(), reference)
    utils.gems_assert_equal(registered.cpu(), reference)
    assert direct.dtype is torch.int64
    assert direct.is_contiguous()


@pytest.mark.parametrize(
    "op_name,row,col,offset,expected",
    [
        (
            "tril_indices",
            INT64_MAX,
            INT64_MAX,
            1 - INT64_MAX,
            [[INT64_MAX - 1], [0]],
        ),
        ("triu_indices", INT64_MAX, 1, 0, [[0], [0]]),
        ("tril_indices", 1, INT64_MAX, 0, [[0], [0]]),
        (
            "triu_indices",
            1,
            INT64_MAX,
            INT64_MAX - 1,
            [[0], [INT64_MAX - 1]],
        ),
    ],
)
def test_triangular_indices_sparse_int64_extremes(op_name, row, col, offset, expected):
    result = getattr(flag_gems, op_name)(
        row, col, offset, dtype=torch.int64, device=flag_gems.device
    )
    reference = torch.tensor(expected, dtype=torch.int64)
    utils.gems_assert_equal(result.cpu(), reference)


@pytest.mark.parametrize(
    "op_name,offset,expected_size",
    [
        ("tril_indices", INT64_MIN, 0),
        ("tril_indices", INT64_MAX, 20),
        ("triu_indices", INT64_MAX, 0),
        ("triu_indices", INT64_MIN, 20),
    ],
)
def test_triangular_indices_far_offset_fast_paths(op_name, offset, expected_size):
    result = getattr(flag_gems, op_name)(
        4, 5, offset, dtype=torch.int64, device=flag_gems.device
    )
    assert result.shape == (2, expected_size)
    assert result.is_contiguous()


@pytest.mark.parametrize(
    "op_name,row,col,offset,expected",
    [
        (
            "tril_indices",
            INT32_MAX + 1,
            1,
            -INT32_MAX,
            [[INT32_MAX], [0]],
        ),
        ("triu_indices", INT64_MAX, 1, 0, [[0], [0]]),
        ("tril_indices", 1, INT64_MAX, 0, [[0], [0]]),
        (
            "triu_indices",
            1,
            INT32_MAX + 1,
            INT32_MAX,
            [[0], [INT32_MAX]],
        ),
    ],
)
def test_triangular_indices_int32_emitted_value_boundary(
    op_name, row, col, offset, expected
):
    result = getattr(flag_gems, op_name)(
        row, col, offset, dtype=torch.int32, device=flag_gems.device
    )
    reference = torch.tensor(expected, dtype=torch.int32)
    utils.gems_assert_equal(result.cpu(), reference)


@pytest.mark.parametrize(
    "op_name,row,col,offset",
    [
        ("tril_indices", INT32_MAX + 2, 1, -(INT32_MAX + 1)),
        ("triu_indices", 1, INT32_MAX + 2, INT32_MAX + 1),
    ],
)
def test_triangular_indices_rejects_unrepresentable_int32_value(
    op_name, row, col, offset
):
    with pytest.raises(RuntimeError, match="represented as int32"):
        getattr(flag_gems, op_name)(
            row, col, offset, dtype=torch.int32, device=flag_gems.device
        )


@pytest.mark.parametrize("op_name", ["tril_indices", "triu_indices"])
@pytest.mark.parametrize(
    "row,col,offset,kwargs",
    [
        (-1, 2, 0, {}),
        (2, -1, 0, {}),
        (2, 2, 0, {"dtype": torch.float32}),
        (2, 2, 0, {"layout": torch.sparse_coo}),
        (INT64_MAX + 1, 1, 0, {}),
        (1, 1, INT64_MAX + 1, {}),
    ],
)
def test_triangular_indices_invalid_arguments(op_name, row, col, offset, kwargs):
    with pytest.raises((RuntimeError, TypeError)):
        getattr(flag_gems, op_name)(row, col, offset, device=flag_gems.device, **kwargs)


@pytest.mark.parametrize(
    "op_name,row,col,offset",
    [
        ("tril_indices", INT64_MAX, 2, 1),
        ("triu_indices", INT64_MAX, 2, 1 - INT64_MAX),
    ],
)
def test_triangular_indices_rejects_size_overflow(op_name, row, col, offset):
    with pytest.raises(RuntimeError, match="signed 64-bit range"):
        getattr(flag_gems, op_name)(
            row, col, offset, dtype=torch.int64, device=flag_gems.device
        )


@pytest.mark.parametrize(
    "op_name,row,offset",
    [
        ("tril_indices", INT64_MAX // 16 + 1, 0),
        ("triu_indices", INT64_MAX // 16 + 1, 1 - (INT64_MAX // 16 + 1)),
    ],
)
def test_triangular_indices_rejects_allocation_size_overflow(op_name, row, offset):
    with pytest.raises(RuntimeError, match="allocation size"):
        getattr(flag_gems, op_name)(
            row, 1, offset, dtype=torch.int64, device=flag_gems.device
        )
