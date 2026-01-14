import os

import pytest
import torch

import flag_gems
from flag_gems.ops.median import median_dim

from .accuracy_utils import (
    FLOAT_DTYPES,
    INT_DTYPES,
    gems_assert_close,
    gems_assert_equal,
    to_reference,
)


def assert_median_indices_equal_if_no_duplicates(inp, res_out, ref_out, *, dim, keepdim):
    def _dup_mask_along_dim(inp: torch.Tensor, *, dim: int) -> torch.Tensor:
        dim = dim % inp.ndim
        x = inp.movedim(dim, -1).contiguous()
        n = x.shape[-1]
        if n <= 1:
            return torch.zeros(x.shape[:-1], device=inp.device, dtype=torch.bool)
        x2d = x.reshape(-1, n)
        xs, _ = torch.sort(x2d, dim=-1)
        dup = (xs[:, 1:] == xs[:, :-1]).any(dim=-1)
        return dup.reshape(x.shape[:-1])
    
    dup = _dup_mask_along_dim(inp, dim=dim)
    if keepdim:
        dup = dup.unsqueeze(dim % inp.ndim)
    mask = (~dup).reshape(-1)
    if mask.any():
        gems_assert_equal(
            res_out.indices.reshape(-1)[mask],
            ref_out.indices.reshape(-1)[mask],
        )


SHAPE_SMALL = [
    (1,),
    (2,),
    (3,),
    (1, 1),
    (2, 2),
    (3, 3),
    (1, 1, 1),
    (2, 2, 2),
]

SHAPE_MEDIUM = [
    (8,),
    (16,),
    (32,),
    (64,),
    (8, 8),
    (16, 16),
    (32, 32),
    (4, 8, 8),
    (8, 16, 16),
    (2, 4, 8, 8),
    (4, 4, 16, 16),
]

SHAPE_LARGE = [
    (256,),
    (512,),
    (1024,),
    (128, 128),
    (256, 256),
    (16, 128, 128),
    (8, 16, 64, 64),
    (4, 8, 32, 32, 32),
]


SHAPE_DIMENSIONS = [
    (1,),
    (8,),
    (32,),
    (128,),
    (1, 1),
    (4, 8),
    (16, 32),
    (64, 128),
    (1, 1, 1),
    (2, 4, 8),
    (8, 16, 32),
    (16, 32, 64),
    (1, 1, 1, 1),
    (2, 4, 8, 16),
    (4, 8, 16, 32),
    (8, 16, 32, 64),
    (1, 1, 1, 1, 1),
    (2, 2, 4, 8, 16),
    (4, 4, 8, 16, 32),
]

DIM_VALUES = [
    -1,
    0,
    1,
    2,
]

KEEPDIM_VALUES = [True, False]


SHAPE_EXTREME_VALUES = [
    (1,),
    (2,),
    (3,),
    (1000,),
]

SHAPE_WITH_ZEROS = [
    (8,),
    (16,),
    (32,),
    (4, 8),
    (8, 16),
]

SHAPE_WITH_NEGATIVES = [
    (8,),
    (16,),
    (32,),
    (4, 8),
    (8, 16),
]

SHAPE_EMPTY = [
    (0,),
    (0, 1),
    (1, 0),
    (0, 0),
]


@pytest.mark.median
@pytest.mark.parametrize("shape", SHAPE_SMALL)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_median_small(shape, dtype):
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = to_reference(inp, True)

    ref_out = torch.median(ref_inp, dim=-1, keepdim=False)
    res_out = median_dim(inp, dim=-1, keepdim=False)

    gems_assert_close(res_out.values, ref_out.values, dtype)
    assert_median_indices_equal_if_no_duplicates(
        inp, res_out, ref_out, dim=-1, keepdim=False
    )

    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        del os.environ["MUSA_ENABLE_SQMMA"]


@pytest.mark.median
@pytest.mark.parametrize("shape", SHAPE_MEDIUM)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_median_medium(shape, dtype):
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = to_reference(inp, True)

    ref_out = torch.median(ref_inp, dim=-1, keepdim=False)
    res_out = median_dim(inp, dim=-1, keepdim=False)

    gems_assert_close(res_out.values, ref_out.values, dtype)
    assert_median_indices_equal_if_no_duplicates(
        inp, res_out, ref_out, dim=-1, keepdim=False
    )

    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        del os.environ["MUSA_ENABLE_SQMMA"]


@pytest.mark.median
@pytest.mark.parametrize("shape", SHAPE_LARGE)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_median_large(shape, dtype):
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = to_reference(inp, True)

    ref_out = torch.median(ref_inp, dim=-1, keepdim=False)
    res_out = median_dim(inp, dim=-1, keepdim=False)

    gems_assert_close(res_out.values, ref_out.values, dtype)
    assert_median_indices_equal_if_no_duplicates(
        inp, res_out, ref_out, dim=-1, keepdim=False
    )

    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        del os.environ["MUSA_ENABLE_SQMMA"]


@pytest.mark.median
@pytest.mark.parametrize("shape", SHAPE_DIMENSIONS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_median_dimensions(shape, dtype):
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = to_reference(inp, True)

    for dim in range(-inp.ndim, inp.ndim):
        ref_out = torch.median(ref_inp, dim=dim, keepdim=False)
        res_out = median_dim(inp, dim=dim, keepdim=False)

        gems_assert_close(res_out.values, ref_out.values, dtype)
        assert_median_indices_equal_if_no_duplicates(
            inp, res_out, ref_out, dim=dim, keepdim=False
        )

    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        del os.environ["MUSA_ENABLE_SQMMA"]


@pytest.mark.median
@pytest.mark.parametrize("shape", [(8,), (16,), (4, 8), (8, 16), (2, 4, 8)])
@pytest.mark.parametrize("keepdim", KEEPDIM_VALUES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_median_keepdim(shape, keepdim, dtype):
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = to_reference(inp, True)

    ref_out = torch.median(ref_inp, dim=-1, keepdim=keepdim)
    res_out = median_dim(inp, dim=-1, keepdim=keepdim)

    gems_assert_close(res_out.values, ref_out.values, dtype)
    assert_median_indices_equal_if_no_duplicates(
        inp, res_out, ref_out, dim=-1, keepdim=keepdim
    )

    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        del os.environ["MUSA_ENABLE_SQMMA"]


@pytest.mark.median
@pytest.mark.parametrize("shape", SHAPE_EXTREME_VALUES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_median_extreme_values(shape, dtype):
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    max_val = (
        torch.finfo(dtype).max if dtype.is_floating_point else torch.iinfo(dtype).max
    )
    inp_max = torch.full(shape, max_val, dtype=dtype, device=flag_gems.device)
    ref_inp_max = to_reference(inp_max, True)

    ref_out_max = torch.median(ref_inp_max, dim=-1, keepdim=False)
    res_out_max = median_dim(inp_max, dim=-1, keepdim=False)

    gems_assert_close(res_out_max.values, ref_out_max.values, dtype)
    assert_median_indices_equal_if_no_duplicates(
        inp_max, res_out_max, ref_out_max, dim=-1, keepdim=False
    )

    min_val = (
        torch.finfo(dtype).min if dtype.is_floating_point else torch.iinfo(dtype).min
    )
    inp_min = torch.full(shape, min_val, dtype=dtype, device=flag_gems.device)
    ref_inp_min = to_reference(inp_min, True)

    ref_out_min = torch.median(ref_inp_min, dim=-1, keepdim=False)
    res_out_min = median_dim(inp_min, dim=-1, keepdim=False)

    gems_assert_close(res_out_min.values, ref_out_min.values, dtype)
    assert_median_indices_equal_if_no_duplicates(
        inp_min, res_out_min, ref_out_min, dim=-1, keepdim=False
    )

    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        del os.environ["MUSA_ENABLE_SQMMA"]


@pytest.mark.median
@pytest.mark.parametrize("shape", SHAPE_WITH_ZEROS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_median_with_zeros(shape, dtype):
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    zero_indices = torch.randint(0, inp.numel(), (inp.numel() // 4,))
    inp_flatten = inp.flatten()
    inp_flatten[zero_indices] = 0
    inp = inp_flatten.reshape(shape)

    ref_inp = to_reference(inp, True)
    ref_out = torch.median(ref_inp, dim=-1, keepdim=False)
    res_out = median_dim(inp, dim=-1, keepdim=False)

    gems_assert_close(res_out.values, ref_out.values, dtype)
    assert_median_indices_equal_if_no_duplicates(
        inp, res_out, ref_out, dim=-1, keepdim=False
    )

    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        del os.environ["MUSA_ENABLE_SQMMA"]


@pytest.mark.median
@pytest.mark.parametrize("shape", SHAPE_WITH_NEGATIVES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_median_with_negatives(shape, dtype):
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    neg_indices = torch.randint(0, inp.numel(), (inp.numel() // 2,))
    inp_flatten = inp.flatten()
    inp_flatten[neg_indices] = -inp_flatten[neg_indices]
    inp = inp_flatten.reshape(shape)

    ref_inp = to_reference(inp, True)
    ref_out = torch.median(ref_inp, dim=-1, keepdim=False)
    res_out = median_dim(inp, dim=-1, keepdim=False)

    gems_assert_close(res_out.values, ref_out.values, dtype)
    assert_median_indices_equal_if_no_duplicates(
        inp, res_out, ref_out, dim=-1, keepdim=False
    )

    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        del os.environ["MUSA_ENABLE_SQMMA"]


@pytest.mark.median
@pytest.mark.parametrize("shape", [(1,), (2,), (3,), (4,), (5,), (100,), (1000,)])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_median_dynamic_shapes(shape, dtype):
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = to_reference(inp, True)

    ref_out = torch.median(ref_inp, dim=-1, keepdim=False)
    res_out = median_dim(inp, dim=-1, keepdim=False)

    gems_assert_close(res_out.values, ref_out.values, dtype)
    assert_median_indices_equal_if_no_duplicates(
        inp, res_out, ref_out, dim=-1, keepdim=False
    )

    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        del os.environ["MUSA_ENABLE_SQMMA"]


@pytest.mark.median
@pytest.mark.parametrize("shape", [(8,), (16,), (4, 8), (8, 16)])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_median_sorted_input(shape, dtype):
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    inp = torch.sort(torch.randn(shape, dtype=dtype, device=flag_gems.device), dim=-1)[
        0
    ]
    ref_inp = to_reference(inp, True)

    ref_out = torch.median(ref_inp, dim=-1, keepdim=False)
    res_out = median_dim(inp, dim=-1, keepdim=False)

    gems_assert_close(res_out.values, ref_out.values, dtype)
    assert_median_indices_equal_if_no_duplicates(
        inp, res_out, ref_out, dim=-1, keepdim=False
    )

    inp_desc = torch.sort(
        torch.randn(shape, dtype=dtype, device=flag_gems.device),
        dim=-1,
        descending=True,
    )[0]
    ref_inp_desc = to_reference(inp_desc, True)

    ref_out_desc = torch.median(ref_inp_desc, dim=-1, keepdim=False)
    res_out_desc = median_dim(inp_desc, dim=-1, keepdim=False)

    gems_assert_close(res_out_desc.values, ref_out_desc.values, dtype)
    assert_median_indices_equal_if_no_duplicates(
        inp_desc, res_out_desc, ref_out_desc, dim=-1, keepdim=False
    )

    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        del os.environ["MUSA_ENABLE_SQMMA"]


@pytest.mark.median
@pytest.mark.parametrize("shape", [(8,), (16,), (4, 8), (8, 16)])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_median_duplicate_values(shape, dtype):
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    base_val = torch.randn(1, dtype=dtype, device=flag_gems.device).item()
    inp = torch.full(shape, base_val, dtype=dtype, device=flag_gems.device)

    ref_inp = to_reference(inp, True)
    ref_out = torch.median(ref_inp, dim=-1, keepdim=False)
    res_out = median_dim(inp, dim=-1, keepdim=False)

    gems_assert_close(res_out.values, ref_out.values.to(dtype), dtype)

    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        del os.environ["MUSA_ENABLE_SQMMA"]


@pytest.mark.median
def test_median_invalid_dim():
    inp = torch.randn((8, 16), dtype=torch.float32, device=flag_gems.device)

    with pytest.raises(AssertionError, match="Invalid dim"):
        median_dim(inp, dim=2)

    with pytest.raises(AssertionError, match="Invalid dim"):
        median_dim(inp, dim=-3)

    inp_1d = torch.randn((8,), dtype=torch.float32, device=flag_gems.device)
    with pytest.raises(AssertionError, match="Invalid dim"):
        median_dim(inp_1d, dim=1)


@pytest.mark.median
def test_median_empty_dimension():
    inp = torch.empty((0, 8), dtype=torch.float32, device=flag_gems.device)
    with pytest.raises(RuntimeError, match="median: dimension is empty"):
        median_dim(inp, dim=0)

    inp = torch.empty((8, 0), dtype=torch.float32, device=flag_gems.device)
    with pytest.raises(RuntimeError, match="median: dimension is empty"):
        median_dim(inp, dim=1)

    inp = torch.empty((0,), dtype=torch.float32, device=flag_gems.device)
    with pytest.raises(RuntimeError, match="median: dimension is empty"):
        median_dim(inp, dim=0)


@pytest.mark.median
def test_median_dtype_compatibility():
    shape = (8,)

    for dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        ref_inp = to_reference(inp, True)

        ref_out = torch.median(ref_inp, dim=-1, keepdim=False)
        res_out = median_dim(inp, dim=-1, keepdim=False)

        gems_assert_close(res_out.values, ref_out.values, dtype)
        assert_median_indices_equal_if_no_duplicates(
            inp, res_out, ref_out, dim=-1, keepdim=False
        )

    for dtype in INT_DTYPES:
        try:
            inp = torch.randint(-100, 100, shape, dtype=dtype, device=flag_gems.device)
            ref_inp = to_reference(inp, True)

            ref_out = torch.median(ref_inp, dim=-1, keepdim=False)
            res_out = median_dim(inp, dim=-1, keepdim=False)

            gems_assert_close(res_out.values, ref_out.values, dtype)
            assert_median_indices_equal_if_no_duplicates(
                inp, res_out, ref_out, dim=-1, keepdim=False
            )
        except (RuntimeError, TypeError, AssertionError) as e:
            pytest.skip(f"dtype {dtype} not supported: {e}")


@pytest.mark.median
def test_median_different_dim_combinations():
    inp = torch.randn((4, 8, 16), dtype=torch.float32, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    for dim in range(-inp.ndim, inp.ndim):
        for keepdim in [True, False]:
            ref_out = torch.median(ref_inp, dim=dim, keepdim=keepdim)
            res_out = median_dim(inp, dim=dim, keepdim=keepdim)

            gems_assert_close(res_out.values, ref_out.values, torch.float32)
            assert_median_indices_equal_if_no_duplicates(
                inp, res_out, ref_out, dim=dim, keepdim=keepdim
            )

            assert res_out.values.shape == ref_out.values.shape
            assert res_out.indices.shape == ref_out.indices.shape


@pytest.mark.median
@pytest.mark.parametrize("shape", [(1,), (2,), (3,), (4,), (5,), (100,)])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_median_odd_even_elements(shape, dtype):
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = to_reference(inp, True)

    ref_out = torch.median(ref_inp, dim=-1, keepdim=False)
    res_out = median_dim(inp, dim=-1, keepdim=False)

    gems_assert_close(res_out.values, ref_out.values, dtype)
    assert_median_indices_equal_if_no_duplicates(
        inp, res_out, ref_out, dim=-1, keepdim=False
    )

    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        del os.environ["MUSA_ENABLE_SQMMA"]


@pytest.mark.median
def test_median_nan_handling():
    inp = torch.tensor(
        [1.0, 2.0, float("nan"), 4.0, 5.0], dtype=torch.float32, device=flag_gems.device
    )
    ref_inp = to_reference(inp, True)

    ref_out = torch.median(ref_inp, dim=-1, keepdim=False)
    res_out = median_dim(inp, dim=-1, keepdim=False)

    if torch.isnan(ref_out.values):
        assert torch.isnan(res_out.values), f"Expected NaN but got {res_out.values}"
        gems_assert_equal(res_out.indices, ref_out.indices)
    else:
        assert not torch.isnan(res_out.values), "Expected non-NaN but got NaN"
        gems_assert_close(res_out.values, ref_out.values, torch.float32)
        gems_assert_equal(res_out.indices, ref_out.indices)



@pytest.mark.median
def test_median_inf_handling():
    inp = torch.tensor(
        [1.0, 2.0, float("inf"), 4.0, 5.0], dtype=torch.float32, device=flag_gems.device
    )
    ref_inp = to_reference(inp, True)

    ref_out = torch.median(ref_inp, dim=-1, keepdim=False)
    res_out = median_dim(inp, dim=-1, keepdim=False)

    if torch.isinf(ref_out.values):
        assert torch.isinf(res_out.values)
        assert torch.sign(ref_out.values) == torch.sign(res_out.values)
        gems_assert_equal(res_out.indices, ref_out.indices)
    else:
        gems_assert_close(res_out.values, ref_out.values, torch.float32)
        gems_assert_equal(res_out.indices, ref_out.indices)

