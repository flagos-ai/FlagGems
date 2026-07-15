import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Dimension pairs to swap. Covers first/last, middle pairs, and identity.
TRANSPOSE_DIM_PAIRS = [
    (0, 1),
    (0, -1),
    (-1, -2),
    (1, 2),
    (1, -1),
]


# Shapes covering 0D, 1D, 2D, 3D, 4D and a few non-contiguous strides.
# transpose.int is a view op, so we exercise shapes where swapping
# dimensions is observable (ndim >= 2) plus the degenerate 0D/1D cases.
TRANSPOSE_SHAPES = [
    (),
    (5,),
    (2, 3),
    (128, 256),
    (2, 3, 4),
    (16, 128, 64),
    (2, 3, 4, 5),
    (8, 16, 32, 64),
]


@pytest.mark.transpose
@pytest.mark.parametrize("shape", TRANSPOSE_SHAPES)
@pytest.mark.parametrize("dim_pair", TRANSPOSE_DIM_PAIRS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_transpose(shape, dim_pair, dtype):
    dim0, dim1 = dim_pair
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    # Skip dim pairs that are out of range for low-rank tensors.
    ndim = inp.dim()
    d0 = dim0 + ndim if dim0 < 0 else dim0
    d1 = dim1 + ndim if dim1 < 0 else dim1
    if ndim == 0:
        # 0D: only (0, 0) is valid; skip the rest.
        if not (d0 == 0 and d1 == 0):
            pytest.skip("0D tensor only supports transpose(0, 0)")
    elif ndim == 1:
        # 1D: only (0, 0) is valid.
        if not (d0 == 0 and d1 == 0):
            pytest.skip("1D tensor only supports transpose(0, 0)")
    elif d0 >= ndim or d1 >= ndim:
        pytest.skip(f"dim pair {dim_pair} out of range for {ndim}D tensor")

    ref_out = torch.ops.aten.transpose.int(ref_inp, dim0, dim1)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.transpose.int(inp, dim0, dim1)

    utils.gems_assert_equal(res_out, ref_out)
    # transpose.int returns a view: verify shape/strides match aten.
    assert res_out.shape == ref_out.shape
    assert res_out.stride() == ref_out.stride()
    assert res_out._is_view() == ref_out._is_view()


@pytest.mark.transpose
@pytest.mark.parametrize(
    "shape,dim0,dim1",
    [
        ((2, 3, 4), 1, 1),  # same-dim no-op
        ((2, 3, 4), 0, 0),
        ((5, 5), 0, 1),
        ((5, 5), 1, 0),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_transpose_same_dim(shape, dim0, dim1, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.transpose.int(ref_inp, dim0, dim1)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.transpose.int(inp, dim0, dim1)

    utils.gems_assert_equal(res_out, ref_out)
    assert res_out.shape == ref_out.shape
    assert res_out.stride() == ref_out.stride()


@pytest.mark.transpose
@pytest.mark.parametrize(
    "shape",
    [(2, 3, 4), (8, 16, 32, 64)],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_transpose_non_contiguous(shape, dtype):
    # Verify transpose works correctly on a non-contiguous input.
    # Build the non-contiguous slice on both the test device and the reference
    # device so the two inputs share the same memory layout. Calling
    # ``to_reference`` *after* slicing would densify the CPU copy (cross-device
    # ``.to("cpu")`` materializes a contiguous tensor), which would make the
    # stride comparison below compare mismatched layouts.
    base = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_base = utils.to_reference(base)
    if shape[0] > 1:
        inp = base[::2]
        ref_inp = ref_base[::2]
    else:
        inp = base
        ref_inp = ref_base

    ref_out = torch.ops.aten.transpose.int(ref_inp, 0, -1)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.transpose.int(inp, 0, -1)

    utils.gems_assert_equal(res_out, ref_out)
    assert res_out.shape == ref_out.shape
    assert res_out.stride() == ref_out.stride()


@pytest.mark.transpose
@pytest.mark.parametrize(
    "shape,dim0,dim1",
    [
        ((2, 3), 0, 5),  # dim1 out of range
        ((2, 3), -3, 0),  # dim0 too negative
        ((2, 3, 4), 0, 3),  # dim1 out of range for 3D
    ],
)
def test_transpose_invalid_dims(shape, dim0, dim1):
    inp = torch.randn(shape, device=flag_gems.device)
    with flag_gems.use_gems(), pytest.raises(IndexError):
        torch.ops.aten.transpose.int(inp, dim0, dim1)
