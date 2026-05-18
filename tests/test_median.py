import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


# dtypes supported by torch.median
MEDIAN_DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.int32,
    torch.int64,
]

if utils.fp64_is_supported:
    MEDIAN_DTYPES.append(torch.float64)

# Shapes: (batch, size_along_dim)
MEDIAN_SHAPES_1D = [1, 2, 3, 5, 16, 127, 1024, 4096]
MEDIAN_SHAPES_ND = [
    (1,),
    (16,),
    (256,),
    (3, 5),
    (4, 128),
    (8, 64),
    (4, 8, 16),
    (2, 3, 4, 5),
    (16, 64, 64),
]


def make_input(shape, dtype, device):
    if dtype in (torch.int32, torch.int64):
        return torch.randint(-128, 128, shape, dtype=dtype, device=device)
    elif dtype == torch.float16:
        return torch.randn(shape, dtype=torch.float32, device=device).to(dtype)
    elif dtype == torch.bfloat16:
        return torch.randn(shape, dtype=torch.float32, device=device).to(dtype)
    else:
        return torch.randn(shape, dtype=dtype, device=device)


# ── flat median ───────────────────────────────────────────────────────────────


@pytest.mark.median
@pytest.mark.parametrize("n", MEDIAN_SHAPES_1D)
@pytest.mark.parametrize("dtype", MEDIAN_DTYPES)
def test_median_flat_1d(n, dtype):
    inp = make_input((n,), dtype, flag_gems.device)
    ref = torch.median(utils.to_reference(inp))
    with flag_gems.use_gems():
        res = torch.median(inp)
    utils.gems_assert_close(res, ref, dtype)


@pytest.mark.median
@pytest.mark.parametrize("shape", MEDIAN_SHAPES_ND)
@pytest.mark.parametrize("dtype", MEDIAN_DTYPES)
def test_median_flat_nd(shape, dtype):
    inp = make_input(shape, dtype, flag_gems.device)
    ref = torch.median(utils.to_reference(inp))
    with flag_gems.use_gems():
        res = torch.median(inp)
    utils.gems_assert_close(res, ref, dtype)


# ── dim median ────────────────────────────────────────────────────────────────


@pytest.mark.median
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("dim", [0, 1, -1])
@pytest.mark.parametrize("shape", [(4, 7), (8, 64), (4, 8, 16), (2, 3, 4, 5)])
@pytest.mark.parametrize("dtype", MEDIAN_DTYPES)
def test_median_dim(shape, dim, keepdim, dtype):
    ndim = len(shape)
    if dim >= ndim or dim < -ndim:
        pytest.skip(f"dim={dim} out of range for shape {shape}")
    inp = make_input(shape, dtype, flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref = torch.median(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res = torch.median(inp, dim=dim, keepdim=keepdim)
    utils.gems_assert_close(res.values, ref.values, dtype)
    utils.gems_assert_equal(res.indices, ref.indices)


# ── edge cases ────────────────────────────────────────────────────────────────


@pytest.mark.median
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_median_single_element(dtype):
    inp = make_input((1,), dtype, flag_gems.device)
    ref = torch.median(utils.to_reference(inp))
    with flag_gems.use_gems():
        res = torch.median(inp)
    utils.gems_assert_close(res, ref, dtype)


@pytest.mark.median
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_median_single_element_dim(dtype):
    inp = make_input((3, 1, 5), dtype, flag_gems.device)
    ref = torch.median(utils.to_reference(inp), dim=1)
    with flag_gems.use_gems():
        res = torch.median(inp, dim=1)
    utils.gems_assert_close(res.values, ref.values, dtype)
    utils.gems_assert_equal(res.indices, ref.indices)


@pytest.mark.median
def test_median_non_contiguous():
    inp = torch.randn(8, 64, device=flag_gems.device)[::2, ::4]  # non-contiguous
    ref = torch.median(inp.to("cpu").float())
    with flag_gems.use_gems():
        res = torch.median(inp)
    utils.gems_assert_close(res, ref.to(inp.device), torch.float32)


@pytest.mark.median
@pytest.mark.parametrize("dtype", [torch.float32])
def test_median_large(dtype):
    inp = make_input((64, 1024, 64), dtype, flag_gems.device)
    ref = torch.median(utils.to_reference(inp), dim=1)
    with flag_gems.use_gems():
        res = torch.median(inp, dim=1)
    utils.gems_assert_close(res.values, ref.values, dtype)
    utils.gems_assert_equal(res.indices, ref.indices)


@pytest.mark.median
@pytest.mark.parametrize("dtype", [torch.float32])
def test_median_keepdim_shape(dtype):
    inp = make_input((4, 8, 16), dtype, flag_gems.device)
    for dim in range(3):
        ref = torch.median(utils.to_reference(inp), dim=dim, keepdim=True)
        with flag_gems.use_gems():
            res = torch.median(inp, dim=dim, keepdim=True)
        assert res.values.shape == ref.values.shape
        assert res.indices.shape == ref.indices.shape
        utils.gems_assert_close(res.values, ref.values, dtype)
        utils.gems_assert_equal(res.indices, ref.indices)


@pytest.mark.median
def test_median_fp16_special_values():
    inp = torch.tensor(
        [float("inf"), -float("inf"), 0.0, 1.0, -1.0],
        dtype=torch.float16,
        device=flag_gems.device,
    )
    ref = torch.median(inp.float().cpu())
    with flag_gems.use_gems():
        res = torch.median(inp)
    utils.gems_assert_close(res.float(), ref.to(res.device), torch.float32)
