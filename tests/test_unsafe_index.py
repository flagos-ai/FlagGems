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
    # Width chosen so cdiv(N, BLOCK_SIZE1) exceeds CUDA's 65535 gridDim.y
    # limit; the launch spills the overflow into grid axis 2 (regression).
    ((2, 65536 * 256 + 256), ((2,),)),
    ((0, 32), ((0,), None)),  # zero-size dim
    ((16, 16), ((),)),  # 0-d index on dim 0 (behaves like an int index)
    ((16, 16), (None, ())),  # 0-d index on dim 1
    ((8, 6, 4), (None, (), None)),  # 0-d index on the middle dim
    ((8, 6, 4), ((), (3,))),  # 0-d index mixed with a tensor index
    # A 0-d index separated from the tensor index by None: aten counts 0-d
    # tensors as advanced indices, so the broadcast subspace goes to the
    # front of the output (out shape (2, 6), not (6, 2)).
    ((4, 6, 3), ((), None, (2,))),
)

torch.manual_seed(0)


def gen_indices(input_shape, indices_shape, accumulate, index_dtype=torch.int64):
    """
    Generate indices for torch.ops.aten._unsafe_index.
    All index tensors must be broadcastable, so we ensure they have compatible
    shapes (same logic as tests/test_index.py::gen_indices).  A None entry
    yields a None index; an empty tuple yields a 0-d index tensor.
    """
    indices = []
    if len(indices_shape) > 0:
        sizes = []
        for shape in indices_shape:
            if shape is None:
                continue
            if isinstance(shape, int):
                sizes.append(shape)
            elif isinstance(shape, (tuple, list)) and len(shape) > 0:
                sizes.append(shape[0])
        common_size = min(sizes) if sizes else 16

        for i, shape in enumerate(indices_shape):
            if shape is None:
                indices.append(None)
                continue
            if isinstance(shape, int):
                size = min(shape, common_size)
            elif isinstance(shape, (tuple, list)) and len(shape) > 0:
                size = min(shape[0], common_size)
            else:
                size = ()  # 0-d index tensor
            n = input_shape[i]
            if n == 0:
                # Zero-size dim: the index tensor must be empty too.
                indices.append(
                    torch.empty(
                        size if isinstance(size, tuple) else (size,),
                        device=flag_gems.device,
                        dtype=index_dtype,
                    )
                )
                continue
            # [-n, n): negative entries exercise the in-kernel wrap.
            if accumulate:
                index = torch.randint(
                    -n, n, size if isinstance(size, tuple) else (size,)
                )
            else:
                pool = torch.randperm(2 * n) - n
                index = pool[:size] if size else pool[:1].squeeze(0)
            indices.append(index.to(device=flag_gems.device, dtype=index_dtype))
    return indices


@pytest.mark.unsafe_index
@pytest.mark.parametrize("input_shape, indices_shape", UNSAFE_INDEX_ACC_SHAPE)
@pytest.mark.parametrize("index_dtype", [torch.int64, torch.int32])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_unsafe_index(input_shape, indices_shape, index_dtype, dtype):
    inp = torch.randn(
        input_shape, dtype=dtype, device=flag_gems.device, requires_grad=False
    )
    try:
        indices = gen_indices(input_shape, indices_shape, True, index_dtype)
    except Exception:
        return False

    ref_inp = utils.to_reference(inp)
    ref_indices = [
        None if index is None else utils.to_reference(index) for index in indices
    ]
    try:
        ref_out = torch.ops.aten._unsafe_index(ref_inp, ref_indices)
    except (IndexError, RuntimeError):
        return False

    out = flag_gems.unsafe_index(inp, indices)

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
    random_idx_list_len = torch.randint(0, min(input_shape), (1,)).item()
    for i, idx_pos in enumerate(indices_idx):
        if idx_pos:
            indices.append(None)
        else:
            dim_len = input_shape[i]
            random_idx = torch.randint(0, dim_len, (1,)).item()
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
    result_gems_ = flag_gems.unsafe_index(inp, indices)

    utils.gems_assert_close(result_gems_, result_ref_, dtype)


# _unsafe_index only accepts int32/int64 index tensors; bool/int8/uint8/float
# etc. are all rejected with the same RuntimeError as aten.
@pytest.mark.unsafe_index
@pytest.mark.parametrize(
    "index_dtype", [torch.bool, torch.int8, torch.uint8, torch.int16, torch.float32]
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_unsafe_index_error_bad_index_dtype(dtype, index_dtype):
    """Bad index dtypes are rejected with aten's exact error (RuntimeError:
    ``_unsafe_index found unexpected index type ...``).  Calls
    ``flag_gems.unsafe_index`` directly so the gems-side rejection (not
    aten's front-end check) is exercised.
    """

    inp = torch.randn((32, 64), dtype=dtype, device=flag_gems.device)
    idx = torch.randint(0, 32, (8,), device=flag_gems.device).to(index_dtype)

    with pytest.raises(RuntimeError, match="found unexpected index type"):
        flag_gems.unsafe_index(inp, [idx])


@pytest.mark.unsafe_index
@pytest.mark.parametrize("dtype", [torch.float32])
def test_unsafe_index_error_non_broadcastable_indices(dtype):
    """Non-broadcastable index tensors raise aten's IndexError."""

    inp = torch.randn((32, 64), dtype=dtype, device=flag_gems.device)
    idx1 = torch.randint(0, 32, (4,), device=flag_gems.device)
    idx2 = torch.randint(0, 64, (3,), device=flag_gems.device)

    with pytest.raises(IndexError, match="could not be broadcast"):
        flag_gems.unsafe_index(inp, [idx1, idx2])


@pytest.mark.unsafe_index
@pytest.mark.parametrize("dtype", [torch.float32])
def test_unsafe_index_error_empty_indices(dtype):
    """Error handling: empty indices."""

    inp = torch.randn((32, 64), dtype=dtype, device=flag_gems.device)
    indices = []

    with pytest.raises(ValueError, match="at least one index must be provided"):
        flag_gems.unsafe_index(inp, indices)


@pytest.mark.unsafe_index
@pytest.mark.parametrize("dtype", [torch.float32])
def test_unsafe_index_error_too_many_indices(dtype):
    """Error handling: too many indices."""

    inp = torch.randn((32, 64), dtype=dtype, device=flag_gems.device)
    idx1 = torch.randint(0, 32, (8,), device=flag_gems.device)
    idx2 = torch.randint(0, 64, (8,), device=flag_gems.device)
    idx3 = torch.randint(0, 32, (8,), device=flag_gems.device)
    indices = [idx1, idx2, idx3]  # Too many for a 2D tensor

    with pytest.raises(IndexError, match="too many indices"):
        flag_gems.unsafe_index(inp, indices)


@pytest.mark.unsafe_index
@pytest.mark.parametrize("dtype", [torch.float32])
def test_unsafe_index_non_contiguous_input(dtype):
    """Non-contiguous input with fewer tensor indices than dims.

    The gather kernel addresses the input through its full strides, so
    non-contiguous inputs must work without any host-side normalization
    (regression test).
    """
    base = torch.randn((64, 64, 64), dtype=dtype, device=flag_gems.device)
    inp = base.permute(1, 2, 0)
    idx = torch.randint(0, 64, (8,), device=flag_gems.device)
    indices = [idx]

    ref_inp = utils.to_reference(inp)
    ref_indices = [utils.to_reference(idx)]
    ref_out = torch.ops.aten._unsafe_index(ref_inp, ref_indices)
    out = flag_gems.unsafe_index(inp, indices)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.unsafe_index
@pytest.mark.parametrize("dtype", [torch.float32])
def test_unsafe_index_non_contiguous_input_with_none(dtype):
    """Non-contiguous input with a trailing None (contiguous subspace, so no
    transpose runs; the input must still be made contiguous)."""
    base = torch.randn((64, 64, 64), dtype=dtype, device=flag_gems.device)
    inp = base.permute(1, 2, 0)
    idx = torch.randint(0, 64, (8,), device=flag_gems.device)
    indices = [idx, None]

    ref_inp = utils.to_reference(inp)
    ref_indices = [utils.to_reference(idx), None]
    ref_out = torch.ops.aten._unsafe_index(ref_inp, ref_indices)
    out = flag_gems.unsafe_index(inp, indices)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.unsafe_index
@pytest.mark.parametrize("dtype", [torch.float32])
def test_unsafe_index_zero_dim_index_negative(dtype):
    """A negative 0-d index wraps like a negative integer index."""
    inp = torch.randn((16, 16), dtype=dtype, device=flag_gems.device)
    indices = [torch.tensor(-1, device=flag_gems.device)]

    ref_inp = utils.to_reference(inp)
    ref_indices = [utils.to_reference(indices[0])]
    ref_out = torch.ops.aten._unsafe_index(ref_inp, ref_indices)
    out = flag_gems.unsafe_index(inp, indices)

    utils.gems_assert_close(out, ref_out, dtype)
