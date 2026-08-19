import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

FLOAT_DTYPES = utils.FLOAT_DTYPES


# Extra shapes probing higher-rank inputs (input has more dims than 2).
HIGHER_RANK_SHAPES = [
    (3, 4, 12),  # input shape (4, 2, 6) -> flattened k = 12
    (2, 8, 64),  # input shape (8, 8, 8) -> flattened k = 64
]


# (m, n, k): coefficients is (m, n), input is (n, k)
MNK_SHAPES = [
    (1, 1, 1),
    (3, 4, 5),
    (8, 16, 32),
    (2, 64, 128),
    (32, 64, 128),
    (5, 17, 3),  # non-power-of-two remainder
]


@pytest.mark.compute_linear_combination
@pytest.mark.parametrize("m, n, k", MNK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compute_linear_combination(m, n, k, dtype):
    inp = torch.randn((n, k), dtype=dtype, device=flag_gems.device)
    coeffs = torch.randn((m, n), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_coeffs = utils.to_reference(coeffs, True)

    ref_out = torch._compute_linear_combination(ref_inp, ref_coeffs)
    with flag_gems.use_gems():
        res_out = torch._compute_linear_combination(inp, coeffs)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=n)


@pytest.mark.compute_linear_combination
@pytest.mark.parametrize("m, n, k", HIGHER_RANK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compute_linear_combination_higher_rank(m, n, k, dtype):
    # input of shape (n, a, b) so the flattened remainder is k = a*b
    a, b = 2, k // 2
    inp = torch.randn((n, a, b), dtype=dtype, device=flag_gems.device)
    coeffs = torch.randn((m, n), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_coeffs = utils.to_reference(coeffs, True)

    ref_out = torch._compute_linear_combination(ref_inp, ref_coeffs)
    with flag_gems.use_gems():
        res_out = torch._compute_linear_combination(inp, coeffs)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=n)


@pytest.mark.compute_linear_combination
@pytest.mark.parametrize("m, n, k", [(2, 8, 64), (4, 16, 32)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_compute_linear_combination_extreme(m, n, k, dtype):
    # All-zero input: result must be zero.
    inp = torch.zeros((n, k), dtype=dtype, device=flag_gems.device)
    coeffs = torch.zeros((m, n), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref_coeffs = utils.to_reference(coeffs)

    ref_out = torch._compute_linear_combination(ref_inp, ref_coeffs)
    with flag_gems.use_gems():
        res_out = torch._compute_linear_combination(inp, coeffs)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=n)


@pytest.mark.compute_linear_combination_out
@pytest.mark.parametrize("m, n, k", MNK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compute_linear_combination_out(m, n, k, dtype):
    inp = torch.randn((n, k), dtype=dtype, device=flag_gems.device)
    coeffs = torch.randn((m, n), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_coeffs = utils.to_reference(coeffs, True)

    # ``aten::_compute_linear_combination.out`` accumulates into ``out`` rather
    # than overwriting it, so the reference buffer must be zero-initialised;
    # ``torch.empty`` would leak stale values into the result.
    ref_out = torch.zeros((m, k), dtype=ref_inp.dtype, device=ref_inp.device)
    torch.ops.aten._compute_linear_combination.out(ref_inp, ref_coeffs, out=ref_out)
    out = torch.zeros((m, k), dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        res_out = torch.ops.aten._compute_linear_combination.out(inp, coeffs, out=out)

    assert res_out is out
    utils.gems_assert_close(out, ref_out, dtype, reduce_dim=n)
