# Copyright 2026 FlagOS Contributors

import pytest
import torch

import flag_gems


def _factor(shape, dtype):
    matrix = torch.randn(shape, device="cuda", dtype=dtype)
    eye = torch.eye(shape[-1], device="cuda", dtype=dtype)
    return torch.linalg.cholesky(matrix @ matrix.mT + 0.5 * eye)


@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cholesky_solve_helper(dtype, upper):
    factor = _factor((2, 5, 5), dtype)
    if upper:
        factor = factor.mT.contiguous()
    rhs = torch.randn((2, 5, 3), device="cuda", dtype=dtype)
    expected = torch.ops.aten._cholesky_solve_helper(rhs, factor, upper)

    with flag_gems.use_gems(include=["_cholesky_solve_helper"]):
        actual = torch.ops.aten._cholesky_solve_helper(rhs, factor, upper)

    torch.testing.assert_close(actual, expected)


def test_cholesky_solve_helper_out():
    factor = _factor((2, 4, 4), torch.float32)
    rhs = torch.randn((2, 4, 2), device="cuda")
    expected = torch.ops.aten._cholesky_solve_helper(rhs, factor, False)
    actual = torch.empty(0, device="cuda")

    with flag_gems.use_gems(
        include=["_cholesky_solve_helper", "_cholesky_solve_helper_out"]
    ):
        result = torch.ops.aten._cholesky_solve_helper.out(
            rhs, factor, False, out=actual
        )

    assert result is actual
    torch.testing.assert_close(actual, expected)
