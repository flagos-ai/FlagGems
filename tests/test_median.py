import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES

MEDIAN_SHAPES = [(7,), (8,), (4, 7), (7, 4), (3, 5, 7), (2, 0, 4), (0, 3)]


@pytest.mark.median
@pytest.mark.parametrize("shape", MEDIAN_SHAPES)
@pytest.mark.parametrize("dim", [0, -1, 1])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + utils.INT_DTYPES)
def test_accuracy_median_dim(shape, dim, keepdim, dtype):
    rank = len(shape)
    if rank == 0:
        pytest.skip("median.dim requires a dimension")
    if dim >= rank or dim < -rank:
        pytest.skip(f"Dimension {dim} is out of bounds for shape {shape}")

    reduced_dim = dim % rank
    if any(size == 0 for size in shape):
        if shape[reduced_dim] == 0:
            with pytest.raises(IndexError):
                with flag_gems.use_gems():
                    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device)
                    torch.median(inp, dim=dim, keepdim=keepdim)
            return
        inp = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    elif dtype in utils.INT_DTYPES:
        inp = torch.randint(-16, 16, shape, device=flag_gems.device).to(dtype)
    else:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.median(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.median(inp, dim=dim, keepdim=keepdim)

    utils.gems_assert_close(res_out.values, ref_out.values, dtype, equal_nan=True)
    utils.gems_assert_equal(res_out.indices, ref_out.indices)


@pytest.mark.median
@pytest.mark.parametrize("keepdim", [True, False])
def test_accuracy_median_dim_tie_break(keepdim):
    inp = torch.tensor(
        [[1.0, 4.0, 2.0, 3.0], [2.0, 1.0, 2.0, 3.0], [2.0, 2.0, 2.0, 2.0]],
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.median(ref_inp, dim=1, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.median(inp, dim=1, keepdim=keepdim)

    utils.gems_assert_close(res_out.values, ref_out.values, inp.dtype)
    utils.gems_assert_equal(res_out.indices, ref_out.indices)


@pytest.mark.median
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_median_dim_with_nan(dtype, keepdim):
    inp = torch.tensor(
        [[1.0, float("nan"), 2.0], [float("nan"), 1.0, 2.0]],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.median(ref_inp, dim=1, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.median(inp, dim=1, keepdim=keepdim)

    utils.gems_assert_close(res_out.values, ref_out.values, dtype, equal_nan=True)
    utils.gems_assert_equal(res_out.indices, ref_out.indices)


@pytest.mark.median
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_median_dim_special_values(dtype, keepdim):
    inp = torch.tensor(
        [
            [0.0, -0.0, 1.0, -1.0, float("inf"), -float("inf"), 2.0],
            [3.0, -5.0, 7.0, 7.0, -2.0, 0.0, 4.0],
        ],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.median(ref_inp, dim=1, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.median(inp, dim=1, keepdim=keepdim)

    utils.gems_assert_close(res_out.values, ref_out.values, dtype, equal_nan=True)
    utils.gems_assert_equal(res_out.indices, ref_out.indices)


@pytest.mark.median
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + utils.INT_DTYPES)
def test_accuracy_median_dim_noncontiguous(dtype, keepdim):
    base_shape = (3, 5, 7)
    if dtype in utils.INT_DTYPES:
        base = torch.randint(-16, 16, base_shape, device=flag_gems.device).to(dtype)
    else:
        base = torch.randn(base_shape, dtype=dtype, device=flag_gems.device)
    inp = base.transpose(1, 2)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.median(ref_inp, dim=1, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.median(inp, dim=1, keepdim=keepdim)

    utils.gems_assert_close(res_out.values, ref_out.values, dtype, equal_nan=True)
    utils.gems_assert_equal(res_out.indices, ref_out.indices)
