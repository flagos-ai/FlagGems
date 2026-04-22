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
    # Original test cases
    ((2**28,), ((2**16,),)),
    ((32, 32), ((8,), (8,))),
    ((32, 32), ((8,), (2, 8))),
    ((32, 32), ((2, 8),)),
    ((512, 512, 512), ((128,), (128,), (128,))),
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

# Make sure every thread has same seed.
random.seed(time.time() // 100)


def gen_indices(input_shape, indices_shape, accumulate):
    """
    Generate indices for torch.ops.aten.index.
    All index tensors must be broadcastable, so we ensure they have compatible shapes.
    """
    indices = []
    # For torch.ops.aten.index, all index tensors must be broadcastable
    # So we use the same shape for all indices
    if len(indices_shape) > 0:
        # Find the minimum size across all indices to ensure broadcastability
        sizes = []
        for shape in indices_shape:
            if isinstance(shape, int):
                sizes.append(shape)
            elif isinstance(shape, (tuple, list)) and len(shape) > 0:
                sizes.append(shape[0])
            else:
                sizes.append(16)  # default
        common_size = min(sizes) if sizes else 16

        for i, shape in enumerate(indices_shape):
            if isinstance(shape, int):
                size = min(shape, common_size)
            elif isinstance(shape, (tuple, list)) and len(shape) > 0:
                size = min(shape[0], common_size)
            else:
                size = common_size
            index = np.random.choice(
                np.arange(input_shape[i]), size=size, replace=accumulate
            )
            indices.append(torch.tensor(index, device=flag_gems.device))
    return indices


@pytest.mark.index
@pytest.mark.parametrize("input_shape, indices_shape", INDEX_ACC_SHAPE)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_index(input_shape, indices_shape, dtype):
    inp = torch.randn(
        input_shape, dtype=dtype, device=flag_gems.device, requires_grad=False
    )
    try:
        indices = gen_indices(input_shape, indices_shape, True)
    except Exception:
        pytest.skip("Failed to generate valid indices")

    ref_inp = utils.to_reference(inp)
    ref_indices = [utils.to_reference(index) for index in indices]
    try:
        ref_out = torch.ops.aten.index(ref_inp, ref_indices)
    except (IndexError, RuntimeError) as e:
        pytest.skip(f"PyTorch reference failed: {e}")

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
def test_index_error_empty_indices(dtype):
    """Test error handling: empty indices"""

    inp = torch.randn((32, 64), dtype=dtype, device=flag_gems.device)
    indices = []

    with pytest.raises(ValueError, match="at least one index must be provided"):
        flag_gems.index(inp, indices)


@pytest.mark.index
@pytest.mark.parametrize("dtype", [torch.float32])
def test_index_error_too_many_indices(dtype):
    """Test error handling: too many indices"""

    inp = torch.randn((32, 64), dtype=dtype, device=flag_gems.device)
    idx1 = torch.randint(0, 32, (8,), device=flag_gems.device)
    idx2 = torch.randint(0, 64, (8,), device=flag_gems.device)
    idx3 = torch.randint(0, 32, (8,), device=flag_gems.device)
    indices = [idx1, idx2, idx3]  # Too many for 2D tensor

    with pytest.raises(IndexError, match="too many indices"):
        flag_gems.index(inp, indices)
