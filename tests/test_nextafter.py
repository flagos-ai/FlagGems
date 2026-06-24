import numpy as np
import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.nextafter
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_nextafter(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1)
    ref_inp2 = utils.to_reference(inp2)

    ref_out = torch.nextafter(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.nextafter(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.nextafter_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_nextafter_(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1.clone(), True)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = ref_inp1.nextafter_(ref_inp2)
    with flag_gems.use_gems():
        res_out = inp1.nextafter_(inp2)

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(inp1, ref_inp1, dtype)


# --- Edge-case / special-value tests ---

EDGE_DTYPES = utils.FLOAT_DTYPES


def _make_special_tensors(dtype, device, *values):
    """Build tensors of the given dtype from Python float values."""
    return tuple(
        torch.tensor(v, dtype=dtype, device=device) for v in values
    )


@pytest.mark.nextafter
@pytest.mark.parametrize("dtype", EDGE_DTYPES)
@pytest.mark.parametrize(
    "inp, other",
    [
        # same value → return the value unchanged
        (1.0, 1.0),
        (0.0, 0.0),
        (-1.0, -1.0),
        # zeros
        (0.0, 1.0),
        (0.0, -1.0),
        (-0.0, 1.0),
        (-0.0, -1.0),
        # normal positive / negative
        (1.0, 2.0),
        (1.0, 0.5),
        (-1.0, -2.0),
        (-1.0, -0.5),
    ],
)
def test_nextafter_edge_values(dtype, inp, other):
    inp_t = torch.tensor([inp], dtype=dtype, device=flag_gems.device)
    other_t = torch.tensor([other], dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp_t)
    ref_other = utils.to_reference(other_t)

    ref_out = torch.nextafter(ref_inp, ref_other)
    with flag_gems.use_gems():
        res_out = torch.nextafter(inp_t, other_t)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.nextafter
@pytest.mark.parametrize("dtype", EDGE_DTYPES)
def test_nextafter_nan(dtype):
    """nextafter(NaN, y) → NaN; nextafter(x, NaN) → NaN; nextafter(NaN, NaN) → NaN."""
    device = flag_gems.device
    nan_val = float("nan")
    normal_val = 1.0

    cases = [
        ([nan_val], [normal_val]),
        ([normal_val], [nan_val]),
        ([nan_val], [nan_val]),
    ]
    for inp_vals, other_vals in cases:
        inp_t = torch.tensor(inp_vals, dtype=dtype, device=device)
        other_t = torch.tensor(other_vals, dtype=dtype, device=device)
        ref_inp = utils.to_reference(inp_t)
        ref_other = utils.to_reference(other_t)

        ref_out = torch.nextafter(ref_inp, ref_other)
        with flag_gems.use_gems():
            res_out = torch.nextafter(inp_t, other_t)

        # All outputs must be NaN
        assert torch.isnan(ref_out).all()
        assert torch.isnan(res_out).all()


@pytest.mark.nextafter
@pytest.mark.parametrize("dtype", EDGE_DTYPES)
def test_nextafter_infinity(dtype):
    """nextafter(±inf, ...) edge cases."""
    device = flag_gems.device
    finfo = torch.finfo(dtype)
    pos_max = finfo.max - finfo.resolution

    cases = [
        # (+inf, val)  → next value towards val
        (float("inf"), pos_max),
        (float("inf"), float("inf")),
        # (-inf, val)
        (float("-inf"), -pos_max),
        (float("-inf"), float("-inf")),
        # approach inf from below
        (finfo.max, float("inf")),
        (-finfo.max, float("-inf")),
    ]
    for inp, other in cases:
        inp_t = torch.tensor([inp], dtype=dtype, device=device)
        other_t = torch.tensor([other], dtype=dtype, device=device)
        ref_inp = utils.to_reference(inp_t)
        ref_other = utils.to_reference(other_t)

        ref_out = torch.nextafter(ref_inp, ref_other)
        with flag_gems.use_gems():
            res_out = torch.nextafter(inp_t, other_t)

        utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.nextafter
@pytest.mark.parametrize("dtype", EDGE_DTYPES)
def test_nextafter_small_values(dtype):
    """nextafter around zero and for very small subnormal values."""
    device = flag_gems.device
    finfo = torch.finfo(dtype)

    # Smallest positive values
    pos = torch.tensor([0.0, finfo.tiny, finfo.smallest_normal],
                       dtype=dtype, device=device)
    neg = torch.tensor([-0.0, -finfo.tiny, -finfo.smallest_normal],
                       dtype=dtype, device=device)

    for inp_t, other_t in [(pos, neg), (neg, pos)]:
        ref_inp = utils.to_reference(inp_t)
        ref_other = utils.to_reference(other_t)

        ref_out = torch.nextafter(ref_inp, ref_other)
        with flag_gems.use_gems():
            res_out = torch.nextafter(inp_t, other_t)

        utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
