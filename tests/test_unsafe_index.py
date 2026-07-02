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

# Same shape matrix as tests/test_index.py::INDEX_ACC_SHAPE.
UNSAFE_INDEX_ACC_SHAPE = (
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

# Make sure every thread has the same seed.
random.seed(time.time() // 100)


def gen_indices(input_shape, indices_shape, accumulate):
    """
    Generate indices for torch.ops.aten._unsafe_index.
    All index tensors must be broadcastable, so we ensure they have compatible
    shapes (same logic as tests/test_index.py::gen_indices).
    """
    indices = []
    if len(indices_shape) > 0:
        sizes = []
        for shape in indices_shape:
            if isinstance(shape, int):
                sizes.append(shape)
            elif isinstance(shape, (tuple, list)) and len(shape) > 0:
                sizes.append(shape[0])
            else:
                sizes.append(16)
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


@pytest.mark.unsafe_index
@pytest.mark.parametrize("input_shape, indices_shape", UNSAFE_INDEX_ACC_SHAPE)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_unsafe_index(input_shape, indices_shape, dtype):
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
        ref_out = torch.ops.aten._unsafe_index(ref_inp, ref_indices)
    except (IndexError, RuntimeError):
        return False

    with flag_gems.use_gems():
        out = torch.ops.aten._unsafe_index(inp, indices)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.unsafe_index
@pytest.mark.parametrize(
    "input_shape, index_pos",
    [
        ((32, 32), 0),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_unsafe_index_with_none_basic_indexing(input_shape, index_pos, dtype):
    """Basic indexing with a single tensor index and a trailing None (slice)."""
    inp = torch.randn(input_shape, dtype=dtype, device=flag_gems.device)
    indices = [None] * len(input_shape)

    idx = torch.randint(0, input_shape[index_pos], (8,), device=flag_gems.device)
    indices[index_pos] = idx

    ref_inp = utils.to_reference(inp)
    ref_indices = [None if idx is None else utils.to_reference(idx) for idx in indices]
    ref_out = torch.ops.aten._unsafe_index(ref_inp, ref_indices)
    with flag_gems.use_gems():
        out = torch.ops.aten._unsafe_index(inp, indices)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.unsafe_index
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
def test_unsafe_index_with_none_and_tensor(input_shape, indices_idx, dtype):
    """Mixed None/tensor index patterns (contiguous and non-contiguous)."""
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
    result_ref_ = torch.ops.aten._unsafe_index(ref_inp, ref_indices)
    with flag_gems.use_gems():
        result_gems_ = torch.ops.aten._unsafe_index(inp, indices)

    utils.gems_assert_close(result_gems_, result_ref_, dtype)


# _unsafe_index rejects bool masks (unlike index, which accepts them).
@pytest.mark.unsafe_index
@pytest.mark.parametrize("dtype", [torch.float32])
def test_unsafe_index_rejects_bool_mask(dtype):
    """``_unsafe_index`` rejects boolean masks (unlike ``index``, which converts
    them to nonzero).  Calls ``flag_gems._unsafe_index`` directly so the
    gems-side rejection (not aten's front-end check) is exercised.
    """

    inp = torch.randn((32, 64), dtype=dtype, device=flag_gems.device)
    mask = torch.rand(32, 64, device=flag_gems.device) > 0.5
    indices = [mask]

    with pytest.raises(IndexError, match="bool or int8"):
        flag_gems._unsafe_index(inp, indices)


# _unsafe_index rejects int8 index tensors too (like bool masks).
@pytest.mark.unsafe_index
@pytest.mark.parametrize("dtype", [torch.float32])
def test_unsafe_index_rejects_int8_mask(dtype):
    """``_unsafe_index`` rejects int8 index tensors (like bool masks).  Calls
    ``flag_gems._unsafe_index`` directly so the gems-side rejection (not aten's
    front-end check) is exercised.
    """

    inp = torch.randn((32, 64), dtype=dtype, device=flag_gems.device)
    idx = torch.randint(0, 32, (8,), device=flag_gems.device).to(torch.int8)
    indices = [idx]

    with pytest.raises(IndexError, match="bool or int8"):
        flag_gems._unsafe_index(inp, indices)


@pytest.mark.unsafe_index
@pytest.mark.parametrize("dtype", [torch.float32])
def test_unsafe_index_empty_tensor(dtype):
    """Indexing into a zero-size dimension must still produce the right shape."""

    inp = torch.empty((0, 32), dtype=dtype, device=flag_gems.device)
    idx = torch.empty((0,), dtype=torch.long, device=flag_gems.device)
    indices = [idx, None]

    ref_inp = utils.to_reference(inp)
    ref_indices = [utils.to_reference(idx), None]
    ref_out = torch.ops.aten._unsafe_index(ref_inp, ref_indices)
    with flag_gems.use_gems():
        out = torch.ops.aten._unsafe_index(inp, indices)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.unsafe_index
@pytest.mark.parametrize("dtype", [torch.float32])
def test_unsafe_index_1d_special_case(dtype):
    """1-D input indexed by a 1-D index."""

    inp = torch.randn((128,), dtype=dtype, device=flag_gems.device)
    idx = torch.randint(0, 128, (16,), device=flag_gems.device)
    indices = [idx]

    ref_inp = utils.to_reference(inp)
    ref_indices = [utils.to_reference(idx)]
    ref_out = torch.ops.aten._unsafe_index(ref_inp, ref_indices)
    with flag_gems.use_gems():
        out = torch.ops.aten._unsafe_index(inp, indices)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.unsafe_index
@pytest.mark.parametrize("dtype", [torch.float32])
def test_unsafe_index_error_empty_indices(dtype):
    """Error handling: empty indices."""

    inp = torch.randn((32, 64), dtype=dtype, device=flag_gems.device)
    indices = []

    with flag_gems.use_gems():
        with pytest.raises(ValueError, match="at least one index must be provided"):
            torch.ops.aten._unsafe_index(inp, indices)


@pytest.mark.unsafe_index
@pytest.mark.parametrize("dtype", [torch.float32])
def test_unsafe_index_error_too_many_indices(dtype):
    """Error handling: too many indices."""

    inp = torch.randn((32, 64), dtype=dtype, device=flag_gems.device)
    idx1 = torch.randint(0, 32, (8,), device=flag_gems.device)
    idx2 = torch.randint(0, 64, (8,), device=flag_gems.device)
    idx3 = torch.randint(0, 32, (8,), device=flag_gems.device)
    indices = [idx1, idx2, idx3]  # Too many for a 2D tensor

    with flag_gems.use_gems():
        with pytest.raises(IndexError, match="too many indices"):
            torch.ops.aten._unsafe_index(inp, indices)
