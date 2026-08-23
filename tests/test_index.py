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

import random
import time

import numpy as np
import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES

INDEX_ACC_SHAPE = (
    # Keep these cases aligned with benchmark/test_index.py so benchmark
    # latency rows have matching correctness coverage.
    ((2**28,), ((2**16,),)),
    ((32, 32), ((8,), (8,))),
    ((32, 32), ((8,), (2, 8))),
    ((32, 32), ((2, 8),)),
    ((1024, 1024), ((64,), (64,))),
    ((512, 512, 512), ((128,), (128,), (128,))),
    ((512, 512, 512), ((2, 128), (2, 128), (2, 128))),
    ((512, 512, 512), ((2, 128), (128,), (128,))),
    ((512, 512, 512), ((2, 128),)),
    (
        (64, 64, 64),
        (
            (2, 8),
            (2, 8),
        ),
    ),
)

INDEX_NONLEADING_ADJACENT_SHAPE = (
    ((1, 4096, 512), (None, (32768,), None)),
    ((4, 512, 128), (None, (4096,), None)),
    ((2, 256, 256, 64), (None, (4096,), (4096,), None)),
    ((2, 128, 128, 128), (None, (2048,), (2048,), (2048,))),
    (
        (1, 128, 128, 64, 8),
        (None, (2048,), (2048,), (2048,), None),
    ),
)

# Make sure every thread has same seed.
random.seed(time.time() // 100)


def gen_indices(input_shape, indices_shape, accumulate):
    """
    Generate indices for torch.ops.aten.index while preserving benchmark shapes.
    """
    indices = []
    for dim, shape in enumerate(indices_shape):
        index = np.random.choice(
            np.arange(input_shape[dim]), size=shape, replace=accumulate
        )
        indices.append(torch.tensor(index, device=flag_gems.device))
    return indices


def gen_optional_indices(input_shape, indices_shape, accumulate):
    indices = []
    for dim, shape in enumerate(indices_shape):
        if shape is None:
            indices.append(None)
            continue
        index = np.random.choice(
            np.arange(input_shape[dim]), size=shape, replace=accumulate
        )
        indices.append(torch.tensor(index, device=flag_gems.device))
    return indices


@pytest.mark.index
@pytest.mark.parametrize("input_shape, indices_shape", INDEX_ACC_SHAPE)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_index(input_shape, indices_shape, dtype):
    inp = torch.randn(
        input_shape, dtype=dtype, device=flag_gems.device, requires_grad=False
    )
    try:
        indices = gen_indices(input_shape, indices_shape, True)
    except Exception:
        return False

    ref_inp = utils.to_reference(inp)
    ref_indices = [utils.to_reference(index) for index in indices]
    try:
        ref_out = torch.ops.aten.index(ref_inp, ref_indices)
    except (IndexError, RuntimeError):
        return False

    out = flag_gems.index(inp, indices)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.index
@pytest.mark.parametrize(
    "input_shape, indices_shape",
    INDEX_NONLEADING_ADJACENT_SHAPE,
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_index_nonleading_adjacent_tensor_indices(input_shape, indices_shape, dtype):
    inp = torch.randn(
        input_shape, dtype=dtype, device=flag_gems.device, requires_grad=False
    )
    indices = gen_optional_indices(input_shape, indices_shape, True)

    ref_inp = utils.to_reference(inp)
    ref_indices = [
        None if index is None else utils.to_reference(index) for index in indices
    ]
    ref_out = torch.ops.aten.index(ref_inp, ref_indices)
    out = flag_gems.index(inp, indices)

    utils.gems_assert_close(out, ref_out, dtype)


# Additional test cases to improve coverage for index operator
@pytest.mark.index
@pytest.mark.parametrize(
    "input_shape, index_pos",
    [
        ((32, 32), 0),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_index_with_none_basic_indexing(input_shape, index_pos, dtype):
    """Test basic indexing with None (ellipsis-like behavior)"""
    inp = torch.randn(input_shape, dtype=dtype, device=flag_gems.device)
    indices = [None] * len(input_shape)

    # Add a single tensor index at the specified position
    idx = torch.randint(0, input_shape[index_pos], (8,), device=flag_gems.device)
    indices[index_pos] = idx

    ref_inp = utils.to_reference(inp)
    ref_indices = [None if idx is None else utils.to_reference(idx) for idx in indices]
    ref_out = torch.ops.aten.index(ref_inp, ref_indices)
    out = flag_gems.index(inp, indices)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.index
@pytest.mark.parametrize(
    "input_shape, indices_idx",
    # 0 in indices_idx means a Tensor
    # 1 in indices_idx means None
    [
        ((1024, 1024), (0, 1)),
        ((16, 16, 16), (1, 0, 0)),
        ((16, 16, 16), (0, 1, 0)),
        ((32, 32, 32), (0, 0, 1)),
        ((32, 32, 32), (1, 1, 0)),
        ((64, 64, 64), (1, 0, 1)),
        ((64, 64, 64), (0, 1, 1)),
        ((12, 12, 12, 12), (1, 0, 0, 0)),
        ((12, 12, 12, 12), (0, 1, 0, 0)),
        ((10, 10, 10, 10), (0, 0, 1, 0)),
        ((10, 10, 10, 10), (0, 0, 0, 1)),
        ((10, 10, 10, 10), (1, 1, 0, 0)),
        ((10, 10, 10, 10), (1, 0, 1, 0)),
        ((16, 16, 16, 16), (1, 0, 0, 1)),
        ((16, 16, 16, 16), (0, 1, 1, 0)),
        ((32, 32, 32, 32), (0, 1, 0, 1)),
        ((32, 32, 32, 32), (0, 0, 1, 1)),
        ((8, 8, 8, 8), (0, 1, 1, 1)),
        ((8, 8, 8, 8), (1, 0, 1, 1)),
        ((8, 8, 8, 8), (1, 1, 0, 1)),
        ((8, 8, 8, 8), (1, 1, 1, 0)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.int64])
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_index_with_none_and_tensor(input_shape, indices_idx, dtype):
    inp = torch.randint(0, 10000, input_shape, dtype=dtype, device=flag_gems.device)
    indices = []
    random_idx_list_len = random.randint(0, min(input_shape) - 1)
    for i, idx_pos in enumerate(indices_idx):
        if idx_pos:
            indices.append(None)
        else:
            dim_len = input_shape[i]
            random_idx = random.randint(0, dim_len - 1)
            indices.append(
                torch.tensor(
                    [random_idx for _ in range(random_idx_list_len)],
                    device=flag_gems.device,
                    dtype=dtype,
                )
            )

    ref_inp = utils.to_reference(inp)
    ref_indices = [utils.to_reference(x) for x in indices]
    result_ref_ = torch.ops.aten.index(ref_inp, ref_indices)
    result_gems_ = flag_gems.index(inp, indices)

    utils.gems_assert_close(result_gems_, result_ref_, dtype)


@pytest.mark.index
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_index_boolean_mask(dtype):
    """Test boolean mask indexing"""

    inp = torch.randn((32, 64), dtype=dtype, device=flag_gems.device)
    mask = torch.rand(32, 64, device=flag_gems.device) > 0.5
    indices = [mask]

    ref_inp = utils.to_reference(inp)
    ref_indices = [utils.to_reference(mask)]
    ref_out = torch.ops.aten.index(ref_inp, ref_indices)
    out = flag_gems.index(inp, indices)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.index
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_index_empty_tensor(dtype):
    """Test index with empty tensor"""

    inp = torch.empty((0, 32), dtype=dtype, device=flag_gems.device)
    idx = torch.empty((0,), dtype=torch.long, device=flag_gems.device)
    indices = [idx, None]

    ref_inp = utils.to_reference(inp)
    ref_indices = [utils.to_reference(idx), None]
    ref_out = torch.ops.aten.index(ref_inp, ref_indices)
    out = flag_gems.index(inp, indices)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.index
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_index_1d_special_case(dtype):
    """Test 1D input special case (uses gather)"""

    inp = torch.randn((128,), dtype=dtype, device=flag_gems.device)
    idx = torch.randint(0, 128, (16,), device=flag_gems.device)
    indices = [idx]

    ref_inp = utils.to_reference(inp)
    ref_indices = [utils.to_reference(idx)]
    ref_out = torch.ops.aten.index(ref_inp, ref_indices)
    out = flag_gems.index(inp, indices)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.index
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_index_error_empty_indices(dtype):
    """Test error handling: empty indices"""

    inp = torch.randn((32, 64), dtype=dtype, device=flag_gems.device)
    indices = []

    with pytest.raises(ValueError, match="at least one index must be provided"):
        flag_gems.index(inp, indices)


@pytest.mark.index
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_index_error_too_many_indices(dtype):
    """Test error handling: too many indices"""

    inp = torch.randn((32, 64), dtype=dtype, device=flag_gems.device)
    idx1 = torch.randint(0, 32, (8,), device=flag_gems.device)
    idx2 = torch.randint(0, 64, (8,), device=flag_gems.device)
    idx3 = torch.randint(0, 32, (8,), device=flag_gems.device)
    indices = [idx1, idx2, idx3]  # Too many for 2D tensor

    with pytest.raises(IndexError, match="too many indices"):
        flag_gems.index(inp, indices)
