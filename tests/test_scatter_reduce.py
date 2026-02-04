"""Unit tests for scatter_reduce operator."""
import pytest
import torch

import flag_gems
import flag_gems.testing

from .accuracy_utils import PRIMARY_FLOAT_DTYPES, to_reference

# Use only float32 and float16 for scatter_reduce tests
# to avoid bfloat16 precision issues with atomic operations
FLOAT_DTYPES = PRIMARY_FLOAT_DTYPES


def scatter_reduce_assert_close(res, ref, dtype):
    """
    Custom assertion for scatter_reduce with relaxed tolerances.

    Scatter operations use atomic operations which have non-deterministic
    ordering, leading to small precision differences. We use competition-
    compliant tolerances instead of the stricter test defaults.
    """
    # Competition-compliant tolerances
    if dtype == torch.float32:
        atol = 1.3e-6
    elif dtype == torch.float16:
        atol = 1e-3  # Competition standard
    else:
        atol = 1e-4  # Default

    # Note: rtol is calculated internally from RESOLUTION[dtype]
    flag_gems.testing.assert_close(
        res, ref, dtype, atol=atol, equal_nan=False, reduce_dim=1
    )


@pytest.mark.parametrize("shape", [(10,), (20, 30), (15, 20, 25)])
@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize("reduce", ["sum", "prod", "mean", "amax", "amin"])
@pytest.mark.parametrize("include_self", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_scatter_reduce_basic(shape, dim, reduce, include_self, dtype):
    """Test basic scatter_reduce functionality."""
    # Adjust dim for shape
    if dim < 0:
        dim = len(shape) + dim
    if dim >= len(shape):
        dim = len(shape) - 1

    # Create input tensors
    inp = torch.randn(shape, dtype=dtype, device="cuda")

    # Create index tensor with valid indices
    index_shape = list(shape)
    index_shape[dim] = min(5, shape[dim])
    index = torch.randint(0, shape[dim], index_shape, dtype=torch.long, device="cuda")

    # Create source tensor matching index shape
    src = torch.randn(index_shape, dtype=dtype, device="cuda")

    # Reference result
    ref_inp = to_reference(inp, True)
    ref_index = to_reference(index)
    ref_src = to_reference(src, True)
    ref_out = torch.scatter_reduce(
        ref_inp, dim, ref_index, ref_src, reduce, include_self=include_self
    )

    # FlagGems result
    gems_out = flag_gems.scatter_reduce(
        inp, dim, index, src, reduce, include_self=include_self
    )

    # Compare
    scatter_reduce_assert_close(gems_out, ref_out, dtype)


@pytest.mark.parametrize("shape", [(100,), (50, 60)])
@pytest.mark.parametrize("reduce", ["sum", "prod", "mean", "amax", "amin"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_scatter_reduce_large(shape, reduce, dtype):
    """Test scatter_reduce with larger tensors."""
    dim = 0

    inp = torch.randn(shape, dtype=dtype, device="cuda")
    index_shape = list(shape)
    index_shape[dim] = min(20, shape[dim])
    index = torch.randint(0, shape[dim], index_shape, dtype=torch.long, device="cuda")
    src = torch.randn(index_shape, dtype=dtype, device="cuda")

    ref_inp = to_reference(inp, True)
    ref_index = to_reference(index)
    ref_src = to_reference(src, True)
    ref_out = torch.scatter_reduce(
        ref_inp, dim, ref_index, ref_src, reduce, include_self=True
    )

    gems_out = flag_gems.scatter_reduce(inp, dim, index, src, reduce, include_self=True)

    scatter_reduce_assert_close(gems_out, ref_out, dtype)


@pytest.mark.parametrize("shape", [(10, 10), (5, 8, 6)])
@pytest.mark.parametrize("reduce", ["sum", "mean"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_scatter_reduce_all_dims(shape, reduce, dtype):
    """Test scatter_reduce across all dimensions."""
    for dim in range(len(shape)):
        inp = torch.randn(shape, dtype=dtype, device="cuda")
        index_shape = list(shape)
        index_shape[dim] = min(3, shape[dim])
        index = torch.randint(
            0, shape[dim], index_shape, dtype=torch.long, device="cuda"
        )
        src = torch.randn(index_shape, dtype=dtype, device="cuda")

        ref_inp = to_reference(inp, True)
        ref_index = to_reference(index)
        ref_src = to_reference(src, True)
        ref_out = torch.scatter_reduce(
            ref_inp, dim, ref_index, ref_src, reduce, include_self=True
        )

        gems_out = flag_gems.scatter_reduce(
            inp, dim, index, src, reduce, include_self=True
        )

        scatter_reduce_assert_close(gems_out, ref_out, dtype)


@pytest.mark.parametrize("reduce", ["sum", "prod", "mean", "amax", "amin"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_scatter_reduce_edge_cases(reduce, dtype):
    """Test edge cases for scatter_reduce."""
    # Single element
    inp = torch.randn(1, dtype=dtype, device="cuda")
    index = torch.tensor([0], dtype=torch.long, device="cuda")
    src = torch.randn(1, dtype=dtype, device="cuda")

    ref_inp = to_reference(inp, True)
    ref_index = to_reference(index)
    ref_src = to_reference(src, True)
    ref_out = torch.scatter_reduce(
        ref_inp, 0, ref_index, ref_src, reduce, include_self=True
    )

    gems_out = flag_gems.scatter_reduce(inp, 0, index, src, reduce, include_self=True)

    scatter_reduce_assert_close(gems_out, ref_out, dtype)

    # All same index
    inp = torch.randn(10, dtype=dtype, device="cuda")
    index = torch.zeros(5, dtype=torch.long, device="cuda")
    src = torch.randn(5, dtype=dtype, device="cuda")

    ref_inp = to_reference(inp, True)
    ref_index = to_reference(index)
    ref_src = to_reference(src, True)
    ref_out = torch.scatter_reduce(
        ref_inp, 0, ref_index, ref_src, reduce, include_self=True
    )

    gems_out = flag_gems.scatter_reduce(inp, 0, index, src, reduce, include_self=True)

    scatter_reduce_assert_close(gems_out, ref_out, dtype)


@pytest.mark.parametrize("shape", [(10, 10), (5, 8, 6)])
@pytest.mark.parametrize("reduce", ["sum", "amax", "amin"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_scatter_reduce_zeros(shape, reduce, dtype):
    """Test scatter_reduce with zero values."""
    dim = 0

    inp = torch.zeros(shape, dtype=dtype, device="cuda")
    index_shape = list(shape)
    index_shape[dim] = min(3, shape[dim])
    index = torch.randint(0, shape[dim], index_shape, dtype=torch.long, device="cuda")
    src = torch.zeros(index_shape, dtype=dtype, device="cuda")

    ref_inp = to_reference(inp, True)
    ref_index = to_reference(index)
    ref_src = to_reference(src, True)
    ref_out = torch.scatter_reduce(
        ref_inp, dim, ref_index, ref_src, reduce, include_self=True
    )

    gems_out = flag_gems.scatter_reduce(inp, dim, index, src, reduce, include_self=True)

    scatter_reduce_assert_close(gems_out, ref_out, dtype)


@pytest.mark.parametrize("shape", [(10, 10), (5, 8, 6)])
@pytest.mark.parametrize("reduce", ["sum", "prod"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_scatter_reduce_negative(shape, reduce, dtype):
    """Test scatter_reduce with negative values."""
    dim = 0

    inp = torch.randn(shape, dtype=dtype, device="cuda") - 0.5
    index_shape = list(shape)
    index_shape[dim] = min(3, shape[dim])
    index = torch.randint(0, shape[dim], index_shape, dtype=torch.long, device="cuda")
    src = torch.randn(index_shape, dtype=dtype, device="cuda") - 0.5

    ref_inp = to_reference(inp, True)
    ref_index = to_reference(index)
    ref_src = to_reference(src, True)
    ref_out = torch.scatter_reduce(
        ref_inp, dim, ref_index, ref_src, reduce, include_self=True
    )

    gems_out = flag_gems.scatter_reduce(inp, dim, index, src, reduce, include_self=True)

    scatter_reduce_assert_close(gems_out, ref_out, dtype)


@pytest.mark.parametrize("shape", [(20, 30), (10, 15, 20)])
@pytest.mark.parametrize("reduce", ["sum", "mean"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_scatter_reduce_inplace(shape, reduce, dtype):
    """Test in-place scatter_reduce."""
    dim = 0

    inp = torch.randn(shape, dtype=dtype, device="cuda")
    inp_copy = inp.clone()
    index_shape = list(shape)
    index_shape[dim] = min(5, shape[dim])
    index = torch.randint(0, shape[dim], index_shape, dtype=torch.long, device="cuda")
    src = torch.randn(index_shape, dtype=dtype, device="cuda")

    ref_inp = to_reference(inp_copy, True)
    ref_index = to_reference(index)
    ref_src = to_reference(src, True)
    # PyTorch uses Tensor.scatter_reduce_ method, not torch.scatter_reduce_
    ref_out = ref_inp.scatter_reduce_(
        dim, ref_index, ref_src, reduce, include_self=True
    )

    gems_out = flag_gems.scatter_reduce_(
        inp, dim, index, src, reduce, include_self=True
    )

    scatter_reduce_assert_close(gems_out, ref_out, dtype)
    scatter_reduce_assert_close(inp, ref_inp, dtype)


@pytest.mark.parametrize("shape", [(100, 100), (50, 60, 70)])
@pytest.mark.parametrize("reduce", ["sum", "amax", "amin"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_scatter_reduce_large_index(shape, reduce, dtype):
    """Test scatter_reduce with large index tensors."""
    dim = 0

    inp = torch.randn(shape, dtype=dtype, device="cuda")
    index_shape = list(shape)
    index_shape[dim] = min(50, shape[dim])
    index = torch.randint(0, shape[dim], index_shape, dtype=torch.long, device="cuda")
    src = torch.randn(index_shape, dtype=dtype, device="cuda")

    ref_inp = to_reference(inp, True)
    ref_index = to_reference(index)
    ref_src = to_reference(src, True)
    ref_out = torch.scatter_reduce(
        ref_inp, dim, ref_index, ref_src, reduce, include_self=True
    )

    gems_out = flag_gems.scatter_reduce(inp, dim, index, src, reduce, include_self=True)

    scatter_reduce_assert_close(gems_out, ref_out, dtype)


@pytest.mark.parametrize("reduce", ["sum", "prod", "mean", "amax", "amin"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_scatter_reduce_1d(reduce, dtype):
    """Test scatter_reduce on 1D tensors."""
    inp = torch.randn(20, dtype=dtype, device="cuda")
    index = torch.randint(0, 20, (10,), dtype=torch.long, device="cuda")
    src = torch.randn(10, dtype=dtype, device="cuda")

    ref_inp = to_reference(inp, True)
    ref_index = to_reference(index)
    ref_src = to_reference(src, True)
    ref_out = torch.scatter_reduce(
        ref_inp, 0, ref_index, ref_src, reduce, include_self=True
    )

    gems_out = flag_gems.scatter_reduce(inp, 0, index, src, reduce, include_self=True)

    scatter_reduce_assert_close(gems_out, ref_out, dtype)


@pytest.mark.parametrize("reduce", ["sum", "mean"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_scatter_reduce_2d(reduce, dtype):
    """Test scatter_reduce on 2D tensors."""
    inp = torch.randn(30, 40, dtype=dtype, device="cuda")
    index = torch.randint(0, 30, (15, 40), dtype=torch.long, device="cuda")
    src = torch.randn(15, 40, dtype=dtype, device="cuda")

    ref_inp = to_reference(inp, True)
    ref_index = to_reference(index)
    ref_src = to_reference(src, True)
    ref_out = torch.scatter_reduce(
        ref_inp, 0, ref_index, ref_src, reduce, include_self=True
    )

    gems_out = flag_gems.scatter_reduce(inp, 0, index, src, reduce, include_self=True)

    scatter_reduce_assert_close(gems_out, ref_out, dtype)


@pytest.mark.parametrize("reduce", ["sum", "amax", "amin"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_scatter_reduce_3d(reduce, dtype):
    """Test scatter_reduce on 3D tensors."""
    inp = torch.randn(10, 15, 20, dtype=dtype, device="cuda")
    index = torch.randint(0, 10, (5, 15, 20), dtype=torch.long, device="cuda")
    src = torch.randn(5, 15, 20, dtype=dtype, device="cuda")

    ref_inp = to_reference(inp, True)
    ref_index = to_reference(index)
    ref_src = to_reference(src, True)
    ref_out = torch.scatter_reduce(
        ref_inp, 0, ref_index, ref_src, reduce, include_self=True
    )

    gems_out = flag_gems.scatter_reduce(inp, 0, index, src, reduce, include_self=True)

    scatter_reduce_assert_close(gems_out, ref_out, dtype)


@pytest.mark.parametrize("reduce", ["sum", "prod", "mean", "amax", "amin"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_scatter_reduce_4d(reduce, dtype):
    """Test scatter_reduce on 4D tensors."""
    inp = torch.randn(8, 10, 12, 14, dtype=dtype, device="cuda")
    index = torch.randint(0, 8, (4, 10, 12, 14), dtype=torch.long, device="cuda")
    src = torch.randn(4, 10, 12, 14, dtype=dtype, device="cuda")

    ref_inp = to_reference(inp, True)
    ref_index = to_reference(index)
    ref_src = to_reference(src, True)
    ref_out = torch.scatter_reduce(
        ref_inp, 0, ref_index, ref_src, reduce, include_self=True
    )

    gems_out = flag_gems.scatter_reduce(inp, 0, index, src, reduce, include_self=True)

    scatter_reduce_assert_close(gems_out, ref_out, dtype)


@pytest.mark.parametrize("shape", [(64, 64), (128, 128)])
@pytest.mark.parametrize("reduce", ["sum", "mean"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_scatter_reduce_medium_size(shape, reduce, dtype):
    """Test scatter_reduce with medium-sized tensors."""
    dim = 0

    inp = torch.randn(shape, dtype=dtype, device="cuda")
    index_shape = list(shape)
    index_shape[dim] = shape[dim] // 2
    index = torch.randint(0, shape[dim], index_shape, dtype=torch.long, device="cuda")
    src = torch.randn(index_shape, dtype=dtype, device="cuda")

    ref_inp = to_reference(inp, True)
    ref_index = to_reference(index)
    ref_src = to_reference(src, True)
    ref_out = torch.scatter_reduce(
        ref_inp, dim, ref_index, ref_src, reduce, include_self=True
    )

    gems_out = flag_gems.scatter_reduce(inp, dim, index, src, reduce, include_self=True)

    scatter_reduce_assert_close(gems_out, ref_out, dtype)


@pytest.mark.parametrize("shape", [(256, 256), (512, 512)])
@pytest.mark.parametrize("reduce", ["sum", "amax"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_scatter_reduce_large_size(shape, reduce, dtype):
    """Test scatter_reduce with large tensors."""
    dim = 0

    inp = torch.randn(shape, dtype=dtype, device="cuda")
    index_shape = list(shape)
    index_shape[dim] = shape[dim] // 4
    index = torch.randint(0, shape[dim], index_shape, dtype=torch.long, device="cuda")
    src = torch.randn(index_shape, dtype=dtype, device="cuda")

    ref_inp = to_reference(inp, True)
    ref_index = to_reference(index)
    ref_src = to_reference(src, True)
    ref_out = torch.scatter_reduce(
        ref_inp, dim, ref_index, ref_src, reduce, include_self=True
    )

    gems_out = flag_gems.scatter_reduce(inp, dim, index, src, reduce, include_self=True)

    scatter_reduce_assert_close(gems_out, ref_out, dtype)


@pytest.mark.parametrize("reduce", ["sum", "mean", "amax"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_scatter_reduce_empty_tensor(reduce, dtype):
    """Test scatter_reduce with empty tensors."""
    # Empty 1D tensor
    inp = torch.randn(0, dtype=dtype, device="cuda")
    index = torch.randint(0, 1, (0,), dtype=torch.long, device="cuda")
    src = torch.randn(0, dtype=dtype, device="cuda")

    ref_inp = to_reference(inp, True)
    ref_index = to_reference(index)
    ref_src = to_reference(src, True)
    ref_out = torch.scatter_reduce(
        ref_inp, 0, ref_index, ref_src, reduce, include_self=True
    )

    gems_out = flag_gems.scatter_reduce(inp, 0, index, src, reduce, include_self=True)

    scatter_reduce_assert_close(gems_out, ref_out, dtype)

    # Empty 2D tensor
    inp = torch.randn(0, 5, dtype=dtype, device="cuda")
    index = torch.randint(0, 1, (0, 5), dtype=torch.long, device="cuda")
    src = torch.randn(0, 5, dtype=dtype, device="cuda")

    ref_inp = to_reference(inp, True)
    ref_index = to_reference(index)
    ref_src = to_reference(src, True)
    ref_out = torch.scatter_reduce(
        ref_inp, 0, ref_index, ref_src, reduce, include_self=True
    )

    gems_out = flag_gems.scatter_reduce(inp, 0, index, src, reduce, include_self=True)

    scatter_reduce_assert_close(gems_out, ref_out, dtype)


if __name__ == "__main__":
    pytest.main([__file__])
