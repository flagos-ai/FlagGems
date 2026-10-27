import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

# Padded shapes plus the per-component extents fed to the operator. Every case
# keeps the last extent equal to the padded trailing dim (except the explicit
# "trailing_trim" cases), because the nested components must agree on their
# trailing shapes for the result to be a valid nested tensor.
# The ragged dimension is dim 1 by definition of the operator, so the shapes
# differ only in the number of trailing dims, the padded length and the batch.
NON_FUSED_CASES = [
    # 2D padded: no trailing dims at all, lengths vary and one component empty.
    ((3, 5), [[0], [5], [3]]),
    ((4, 8), [[8], [1], [4], [6]]),
    # 3D padded: the canonical (B, T, D) transformer layout.
    ((2, 8, 4), [[3, 4], [8, 4]]),
    ((3, 5, 7), [[2, 7], [5, 7], [1, 7]]),
    # Trailing extents trimmed below the padded size: only the prefix box counts.
    ((2, 6, 5), [[3, 2], [6, 4]]),
    # 4D padded: two trailing dims, both trimmed independently.
    ((2, 4, 3, 5), [[2, 3, 5], [4, 1, 2]]),
    # 5D padded: checks the general N-dimensional index arithmetic.
    ((2, 3, 2, 2, 3), [[1, 2, 2, 3], [3, 1, 2, 2]]),
    # A component whose ragged length fills the padded dim (max-length path).
    ((2, 16, 3), [[16, 3], [16, 3]]),
]


# ``fuse_transform_0213`` permutes the padded tensor to ``[B, E, T * C]``, so the
# fused layout is only well defined when the component row length equals the
# padded ``T * C`` (the standard attention usage): the CUDA reference derives its
# channel step from an integer division by the padded dim-1 size and reads out of
# bounds for trimmed rows, while the CPU generic path applies a prefix box.
# Cases therefore keep row_len == T * C; E and C vary to cover the
# (0<->2)/(1<->3) swap.
FUSED_CASES = [
    ((2, 2, 2, 4), [[2, 8], [1, 8]]),
    ((1, 3, 3, 5), [[3, 15]]),
    ((2, 4, 4, 2), [[2, 8], [4, 8]]),
    ((3, 2, 2, 3), [[2, 6], [1, 6], [2, 6]]),
]


def _assert_nested_equal(res_out, ref_out, dtype):
    """Compare every observable piece of the legacy nested layout."""
    assert res_out.is_nested and ref_out.is_nested
    assert (
        torch.ops.aten._nested_tensor_size(res_out).tolist()
        == torch.ops.aten._nested_tensor_size(ref_out).tolist()
    ), "component sizes differ"
    assert (
        torch.ops.aten._nested_tensor_strides(res_out).tolist()
        == torch.ops.aten._nested_tensor_strides(ref_out).tolist()
    ), "component strides differ"
    assert (
        torch.ops.aten._nested_tensor_storage_offsets(res_out).tolist()
        == torch.ops.aten._nested_tensor_storage_offsets(ref_out).tolist()
    ), "component storage offsets differ"
    utils.gems_assert_close(res_out.values(), ref_out.values(), dtype)
    for res_comp, ref_comp in zip(res_out.unbind(), ref_out.unbind()):
        utils.gems_assert_close(res_comp, ref_comp, dtype)


def _make_sizes(size_rows):
    # ``cpu_nested_shape_example`` is an int64 CPU tensor by definition.
    return torch.tensor(size_rows, dtype=torch.int64)


@pytest.mark.nested_from_padded
@pytest.mark.parametrize("shape,size_rows", NON_FUSED_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_nested_from_padded(shape, size_rows, dtype):
    padded = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    sizes = _make_sizes(size_rows)

    ref_out = torch.ops.aten._nested_from_padded(
        utils.to_reference(padded), sizes.clone()
    )
    res_out = flag_gems._nested_from_padded(padded, sizes)

    _assert_nested_equal(res_out, ref_out, dtype)


@pytest.mark.nested_from_padded
@pytest.mark.skipif(
    cfg.TO_CPU,
    reason=(
        "fuse_transform_0213 only runs on CUDA; the CPU reference falls back to "
        "the generic masked_select path, whose component layout differs"
    ),
)
@pytest.mark.parametrize("shape,size_rows", FUSED_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_nested_from_padded_fuse_transform_0213(shape, size_rows, dtype):
    padded = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    sizes = _make_sizes(size_rows)

    ref_out = torch.ops.aten._nested_from_padded(padded, sizes.clone(), True)
    res_out = flag_gems._nested_from_padded(padded, sizes, True)

    _assert_nested_equal(res_out, ref_out, dtype)


@pytest.mark.nested_from_padded
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
def test_nested_from_padded_int_dtypes(dtype):
    # Integer inputs follow the same index path; the CUDA generic fallback in the
    # reference is only exercised for floats, so cover ints explicitly here.
    padded = torch.randint(-100, 100, (2, 5, 3), dtype=dtype, device=flag_gems.device)
    sizes = _make_sizes([[2, 3], [5, 1]])

    ref_out = torch.ops.aten._nested_from_padded(
        utils.to_reference(padded), sizes.clone()
    )
    res_out = flag_gems._nested_from_padded(padded, sizes)

    assert (
        torch.ops.aten._nested_tensor_size(res_out).tolist()
        == torch.ops.aten._nested_tensor_size(ref_out).tolist()
    )
    utils.gems_assert_equal(res_out.values(), ref_out.values())


@pytest.mark.nested_from_padded
def test_nested_from_padded_non_contiguous_input():
    # The operator makes its input contiguous internally; verify that a strided
    # view of the same values produces the same result as the reference.
    base = torch.randn((2, 5, 6), dtype=torch.float32, device=flag_gems.device)
    padded = base.transpose(1, 2)
    sizes = _make_sizes([[2, 5], [6, 3]])

    ref_out = torch.ops.aten._nested_from_padded(
        utils.to_reference(padded), sizes.clone()
    )
    res_out = flag_gems._nested_from_padded(padded, sizes)

    _assert_nested_equal(res_out, ref_out, torch.float32)


@pytest.mark.nested_from_padded
def test_nested_from_padded_all_padding():
    # Every component can be empty (all padding); the values buffer is then empty.
    padded = torch.randn((2, 4, 3), dtype=torch.float32, device=flag_gems.device)
    sizes = _make_sizes([[0, 3], [0, 3]])

    ref_out = torch.ops.aten._nested_from_padded(
        utils.to_reference(padded), sizes.clone()
    )
    res_out = flag_gems._nested_from_padded(padded, sizes)

    assert res_out.values().numel() == ref_out.values().numel() == 0
    _assert_nested_equal(res_out, ref_out, torch.float32)
