import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


# ---------------------------------------------------------------------------
# torch.median(t, dim, keepdim) — values + indices
# ---------------------------------------------------------------------------
@pytest.mark.median
@pytest.mark.parametrize("shape", [(64, 64), (256, 256), (1024, 1024), (20, 320, 15)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize("keepdim", [True, False])
def test_accuracy_median_dim(shape, dtype, dim, keepdim):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_values, ref_idx = torch.median(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_values, res_idx = torch.median(inp, dim=dim, keepdim=keepdim)

    utils.gems_assert_close(res_values, ref_values, dtype)
    # The indices must produce the same value when used to index the input,
    # not necessarily be the same integer (ties are valid for either pick).
    ref_pick = torch.gather(
        ref_inp, dim, ref_idx if keepdim else ref_idx.unsqueeze(dim)
    )
    res_pick = torch.gather(inp, dim, res_idx if keepdim else res_idx.unsqueeze(dim))
    utils.gems_assert_close(res_pick, ref_pick, dtype)


@pytest.mark.median
@pytest.mark.parametrize(
    "shape",
    [(1, 1), (8, 8), (64, 64), (256, 256), (1024, 1024)],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_median_dim_various_sizes(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_v, _ = torch.median(ref_inp, dim=-1)
    with flag_gems.use_gems():
        res_v, _ = torch.median(inp, dim=-1)

    utils.gems_assert_close(res_v, ref_v, dtype)


@pytest.mark.median
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_median_dim_single_element(dtype):
    inp = torch.randn((5, 1, 8), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_v, _ = torch.median(ref_inp, dim=1)
    with flag_gems.use_gems():
        res_v, _ = torch.median(inp, dim=1)

    utils.gems_assert_close(res_v, ref_v, dtype)


# ---------------------------------------------------------------------------
# torch.median(t) — whole-tensor scalar
# ---------------------------------------------------------------------------
@pytest.mark.median
@pytest.mark.parametrize("shape", [(64,), (32, 32), (4, 8, 16), (2, 3, 5, 7)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_median_whole_tensor(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref = torch.median(ref_inp)
    with flag_gems.use_gems():
        res = torch.median(inp)
    utils.gems_assert_close(res, ref, dtype)


# ---------------------------------------------------------------------------
# Lower-median tie-break: for even-length inputs torch returns the lower
# median (index (n-1)//2).  Verify the FlagGems impl follows the same rule.
# ---------------------------------------------------------------------------
@pytest.mark.median
def test_median_lower_tiebreak_even_length():
    # [1, 2, 3, 4] -> lower median = 2
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], device=flag_gems.device)
    with flag_gems.use_gems():
        v, _ = torch.median(x, dim=-1)
    assert v.item() == 2.0


# ---------------------------------------------------------------------------
# Constant input: median is the constant value, on every reduction path.
# ---------------------------------------------------------------------------
@pytest.mark.median
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_median_constant_input(dtype):
    x = torch.full((16, 32), 7.5, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        v_dim, _ = torch.median(x, dim=-1)
        v_full = torch.median(x)
    assert torch.all(v_dim == 7.5)
    assert v_full.item() == 7.5


# ---------------------------------------------------------------------------
# Negative dim, multiple dims of the same size, keepdim shape preservation.
# ---------------------------------------------------------------------------
@pytest.mark.median
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_median_keepdim_shape(dtype):
    x = torch.randn((4, 16, 32), dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        v, i = torch.median(x, dim=1, keepdim=True)
    assert v.shape == (4, 1, 32)
    assert i.shape == (4, 1, 32)


@pytest.mark.median
@pytest.mark.parametrize("dim", [-3, -2, -1, 0, 1, 2])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_median_negative_dim(dim, dtype):
    x = torch.randn((6, 7, 8), dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x, True)
    ref_v, _ = torch.median(ref_x, dim=dim)
    with flag_gems.use_gems():
        res_v, _ = torch.median(x, dim=dim)
    utils.gems_assert_close(res_v, ref_v, dtype)


# ---------------------------------------------------------------------------
# Non-contiguous and transposed inputs.
# ---------------------------------------------------------------------------
@pytest.mark.median
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_median_non_contiguous(dtype):
    full = torch.randn((32, 64), dtype=dtype, device=flag_gems.device)
    inp = full[::2, ::2].contiguous()  # actually contiguous
    inp_nc = full[::2, ::2]  # non-contiguous slice
    ref_v, _ = torch.median(utils.to_reference(inp, True), dim=-1)
    with flag_gems.use_gems():
        res_v, _ = torch.median(inp_nc, dim=-1)
    utils.gems_assert_close(res_v, ref_v, dtype)


@pytest.mark.median
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_median_transposed_input(dtype):
    x = torch.randn((16, 32, 8), dtype=dtype, device=flag_gems.device)
    xt = x.transpose(0, 2)  # non-contiguous
    ref_v, _ = torch.median(utils.to_reference(xt, True), dim=-1)
    with flag_gems.use_gems():
        res_v, _ = torch.median(xt, dim=-1)
    utils.gems_assert_close(res_v, ref_v, dtype)


# ---------------------------------------------------------------------------
# Integer dtypes — torch.median supports int.
# ---------------------------------------------------------------------------
@pytest.mark.median
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_median_integer_dtypes(dtype):
    x = torch.randint(-100, 100, (16, 32), dtype=dtype, device=flag_gems.device)
    ref = utils.to_reference(x, True)
    ref_v, _ = torch.median(ref, dim=-1)
    with flag_gems.use_gems():
        res_v, _ = torch.median(x, dim=-1)
    assert torch.equal(res_v.cpu(), ref_v.cpu())


# ---------------------------------------------------------------------------
# Single-element row.
# ---------------------------------------------------------------------------
@pytest.mark.median
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_median_n_equals_one(dtype):
    x = torch.randn((8, 1), dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        v, i = torch.median(x, dim=-1)
    torch.testing.assert_close(v, x.squeeze(-1))
    assert torch.all(i == 0)
