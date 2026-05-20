import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    DIM_LIST = [-1]
    KEEPDIM = [False]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    DIM_LIST = [0, -1]
    KEEPDIM = [True, False]


# --- 4.1.4 coverage matrix ---------------------------------------------------
# - Small / regular / large sizes per the competition requirements:
#     small:    (1, 1), (8, 8)
#     regular:  (64, 64), (256, 256)
#     large:    (1024, 1024)   — exercises the in-block bitonic path edge
# - Both the small-N (bitonic) and large-N (sort + gather) paths are hit by
#   choosing reduction-dim sizes on either side of MAX_BITONIC_N = 1024.
# ---------------------------------------------------------------------------
SMALL_SHAPES = [(1, 1), (8, 8)]
REGULAR_SHAPES = [(64, 64), (256, 256)]
LARGE_SHAPES = [(1024, 1024)]
BITONIC_PATH_SHAPES = SMALL_SHAPES + REGULAR_SHAPES + LARGE_SHAPES
SORT_PATH_SHAPES = [(64, 2048), (16, 8192)]  # N > MAX_BITONIC_N
# Higher-rank shapes (2D-5D) to exercise dim_compress paths.
RANK_SHAPES = [
    (17,),
    (4, 9),
    (3, 5, 7),
    (2, 3, 4, 5),
    (2, 3, 4, 5, 6),
]


@pytest.mark.median
@pytest.mark.parametrize("shape", BITONIC_PATH_SHAPES + SORT_PATH_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + utils.ALL_INT_DTYPES)
def test_median_global(shape, dtype):
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-10000, 10000, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.median(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.median(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.median
@pytest.mark.parametrize("shape", BITONIC_PATH_SHAPES + SORT_PATH_SHAPES)
@pytest.mark.parametrize("keepdim", KEEPDIM)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + utils.ALL_INT_DTYPES)
def test_median_dim(shape, dim, keepdim, dtype):
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-10000, 10000, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    ref_inp = utils.to_reference(inp)

    ref_v, _ = torch.median(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_v, res_i = torch.median(inp, dim=dim, keepdim=keepdim)

    # Values must match exactly (median is a selection, not an arithmetic
    # combination, so there is no numerical error to tolerate).
    utils.gems_assert_equal(res_v, ref_v)

    # Indices may differ when ties exist — PyTorch documents that the
    # returned index is not necessarily the first occurrence. We instead
    # check that the index points to a position whose value equals the
    # reported median.
    norm_dim = dim % inp.ndim
    gather_idx = res_i if keepdim else res_i.unsqueeze(norm_dim)
    gathered = torch.gather(inp, dim=norm_dim, index=gather_idx)
    if not keepdim:
        gathered = gathered.squeeze(norm_dim)
    utils.gems_assert_equal(gathered, res_v)


@pytest.mark.median
@pytest.mark.parametrize("shape", RANK_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_median_dim_ranks(shape, dtype):
    """Cover every legal dim for 1D-5D inputs."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    for dim in range(len(shape)):
        ref_v, _ = torch.median(ref_inp, dim=dim)
        with flag_gems.use_gems():
            res_v, _ = torch.median(inp, dim=dim)
        utils.gems_assert_equal(res_v, ref_v)


@pytest.mark.median
def test_median_negative_dim():
    """dim accepts negative values."""
    inp = torch.randn((4, 5, 6), dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref_v, _ = torch.median(ref_inp, dim=-1, keepdim=True)
    with flag_gems.use_gems():
        res_v, _ = torch.median(inp, dim=-1, keepdim=True)
    utils.gems_assert_equal(res_v, ref_v)


@pytest.mark.median
def test_median_single_element():
    """1-element reduction is a no-op selection."""
    inp = torch.tensor([[42.0]], device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref_v, ref_i = torch.median(ref_inp, dim=1)
    with flag_gems.use_gems():
        res_v, res_i = torch.median(inp, dim=1)
    utils.gems_assert_equal(res_v, ref_v)
    utils.gems_assert_equal(res_i, ref_i)


@pytest.mark.median
def test_median_even_lower():
    """For an even-length row, PyTorch returns the lower of the two middles."""
    inp = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32, device=flag_gems.device
    )
    with flag_gems.use_gems():
        res_v, _ = torch.median(inp, dim=1)
    assert res_v.item() == 2.0


@pytest.mark.median
def test_median_with_inf():
    """+inf in non-median positions must not corrupt the in-block selection."""
    row = torch.tensor(
        [[float("inf"), -1.0, 0.0, 1.0, float("inf")]],
        dtype=torch.float32,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(row)
    ref_v, _ = torch.median(ref_inp, dim=1)
    with flag_gems.use_gems():
        res_v, _ = torch.median(row, dim=1)
    utils.gems_assert_equal(res_v, ref_v)


@pytest.mark.median
def test_median_nan_propagation():
    """Any NaN in the slice ⇒ median is NaN (matches torch.median)."""
    inp = torch.tensor(
        [[1.0, 2.0, float("nan"), 4.0, 5.0]],
        dtype=torch.float32,
        device=flag_gems.device,
    )
    with flag_gems.use_gems():
        res_v, _ = torch.median(inp, dim=1)
    assert torch.isnan(res_v).item()


