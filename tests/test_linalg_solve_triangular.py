import pytest
import torch

torch.backends.cuda.matmul.allow_tf32 = False

import flag_gems  # noqa: E402

from . import accuracy_utils as utils  # noqa: E402


def _make_triangular(shape, dtype, device, upper, unitriangular):
    n = shape[-1]
    if len(shape) == 2:
        A = torch.randn(shape, dtype=dtype, device=device)
    else:
        batch_shape = shape[:-2]
        A = torch.randn(batch_shape + (n, n), dtype=dtype, device=device)

    off_diag = 0.1
    if upper:
        A = A.triu(diagonal=1)
    else:
        A = A.tril(diagonal=-1)
    A.mul_(off_diag)

    eye = torch.eye(n, dtype=dtype, device=device)
    batch_dims = [1] * (A.ndim - 2)
    if batch_dims:
        eye = eye.view(*batch_dims, n, n)
    A.add_(eye)

    if unitriangular:
        A.diagonal(0, -2, -1).fill_(1.0)

    return A


@pytest.mark.linalg_solve_triangular
@pytest.mark.parametrize("n", [1, 4, 8, 16, 32, 64, 128, 256, 512])
@pytest.mark.parametrize("k", [1, 3, 16])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_lower_left(n, k, dtype):
    A = _make_triangular(
        (n, n), dtype, flag_gems.device, upper=False, unitriangular=False
    )
    B = torch.randn(n, k, dtype=dtype, device=flag_gems.device)

    ref_A = utils.to_reference(A)
    ref_B = utils.to_reference(B)
    ref_out = torch.linalg.solve_triangular(ref_A, ref_B, upper=False)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.linalg_solve_triangular(A, B, upper=False)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.linalg_solve_triangular
@pytest.mark.parametrize("n", [1, 4, 8, 16, 32, 64, 128, 256])
@pytest.mark.parametrize("k", [1, 3, 16])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_upper_left(n, k, dtype):
    A = _make_triangular(
        (n, n), dtype, flag_gems.device, upper=True, unitriangular=False
    )
    B = torch.randn(n, k, dtype=dtype, device=flag_gems.device)

    ref_A = utils.to_reference(A)
    ref_B = utils.to_reference(B)
    ref_out = torch.linalg.solve_triangular(ref_A, ref_B, upper=True)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.linalg_solve_triangular(A, B, upper=True)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.linalg_solve_triangular
@pytest.mark.parametrize("n", [4, 16, 64, 128])
@pytest.mark.parametrize("k", [1, 8])
@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_right(n, k, upper, dtype):
    A = _make_triangular(
        (k, k), dtype, flag_gems.device, upper=upper, unitriangular=False
    )
    B = torch.randn(n, k, dtype=dtype, device=flag_gems.device)

    ref_A = utils.to_reference(A)
    ref_B = utils.to_reference(B)
    ref_out = torch.linalg.solve_triangular(ref_A, ref_B, upper=upper, left=False)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.linalg_solve_triangular(A, B, upper=upper, left=False)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.linalg_solve_triangular
@pytest.mark.parametrize("n", [4, 16, 64, 128])
@pytest.mark.parametrize("k", [1, 8])
@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_unitriangular(n, k, upper, dtype):
    A = _make_triangular(
        (n, n), dtype, flag_gems.device, upper=upper, unitriangular=True
    )
    B = torch.randn(n, k, dtype=dtype, device=flag_gems.device)

    ref_A = utils.to_reference(A)
    ref_B = utils.to_reference(B)
    ref_out = torch.linalg.solve_triangular(
        ref_A, ref_B, upper=upper, unitriangular=True
    )

    with flag_gems.use_gems():
        res_out = torch.ops.aten.linalg_solve_triangular(
            A, B, upper=upper, unitriangular=True
        )

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.linalg_solve_triangular
@pytest.mark.parametrize("batch_shape", [(3,), (2, 4)])
@pytest.mark.parametrize("n", [8, 32])
@pytest.mark.parametrize("k", [1, 4])
@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_batched(batch_shape, n, k, upper, dtype):
    shape_A = batch_shape + (n, n)
    shape_B = batch_shape + (n, k)
    A = _make_triangular(
        shape_A, dtype, flag_gems.device, upper=upper, unitriangular=False
    )
    B = torch.randn(shape_B, dtype=dtype, device=flag_gems.device)

    ref_A = utils.to_reference(A)
    ref_B = utils.to_reference(B)
    ref_out = torch.linalg.solve_triangular(ref_A, ref_B, upper=upper)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.linalg_solve_triangular(A, B, upper=upper)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.linalg_solve_triangular
@pytest.mark.parametrize("n", [16, 64, 128])
@pytest.mark.parametrize("k", [1, 8])
@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_out_kwarg(n, k, upper, dtype):
    A = _make_triangular(
        (n, n), dtype, flag_gems.device, upper=upper, unitriangular=False
    )
    B = torch.randn(n, k, dtype=dtype, device=flag_gems.device)
    out = torch.empty_like(B)

    ref_A = utils.to_reference(A)
    ref_B = utils.to_reference(B)
    ref_out = torch.linalg.solve_triangular(ref_A, ref_B, upper=upper)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.linalg_solve_triangular(A, B, upper=upper, out=out)

    assert res_out is out
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.linalg_solve_triangular
@pytest.mark.parametrize("n", [16, 64, 128, 256])
@pytest.mark.parametrize("k", [1, 8])
@pytest.mark.parametrize("upper", [False, True])
def test_residual_f64(n, k, upper):
    """残差验证 (float64 确保精度)"""
    dtype = torch.float64
    A = _make_triangular(
        (n, n), dtype, flag_gems.device, upper=upper, unitriangular=False
    )
    B = torch.randn(n, k, dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.linalg_solve_triangular(A, B, upper=upper)

    residual = (A @ res_out - B).abs().max().item()
    assert residual < 1e-6, f"Residual too large: {residual}"


@pytest.mark.linalg_solve_triangular
@pytest.mark.parametrize("dtype", [torch.float32])
def test_empty(dtype):
    A = torch.empty(0, 0, dtype=dtype, device=flag_gems.device)
    B = torch.empty(0, 0, dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.linalg_solve_triangular(A, B, upper=False)

    assert res_out.shape == (0, 0)
    assert res_out.dtype == dtype


@pytest.mark.linalg_solve_triangular
@pytest.mark.parametrize("n", [64, 128, 256, 512, 1024])
@pytest.mark.parametrize("k", [1, 8])
@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_large_n_f64(n, k, upper, dtype):
    """大规模矩阵测试 — 覆盖所有三个内核分派路径"""
    A = _make_triangular(
        (n, n), dtype, flag_gems.device, upper=upper, unitriangular=False
    )
    B = torch.randn(n, k, dtype=dtype, device=flag_gems.device)

    ref_A = utils.to_reference(A)
    ref_B = utils.to_reference(B)
    ref_out = torch.linalg.solve_triangular(ref_A, ref_B, upper=upper)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.linalg_solve_triangular(A, B, upper=upper)

    atol = 1e-4
    if n >= 1024 and dtype == torch.float32:
        # fp32 大 N 精度物理极限 (实测 2026-08-03): n=1024 abs误差~1e-4, 与 torch 同量级
        # (误差比值 0.45-0.99, 残差我们普遍略优)。以 torch 两个官方路径 (GPU cuBLAS vs
        # CPU LAPACK) 的内部差异为容差锚点 ×1.5 余量 — torch 内部自身差异都大于该量级。
        out_torch_cpu = torch.linalg.solve_triangular(A.cpu(), B.cpu(), upper=upper)
        torch_internal_diff = (
            (ref_out.double().cpu() - out_torch_cpu.double()).abs().max().item()
        )
        atol = torch_internal_diff * 1.5

    utils.gems_assert_close(res_out, ref_out, dtype, atol=atol)


@pytest.mark.linalg_solve_triangular
@pytest.mark.parametrize("n", [8, 32, 128, 512, 600])
@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_no_tle_fallback(n, upper, dtype, monkeypatch):
    """非 TLE 平台回退路径冒烟测试: 强制 HAS_TLE=False 走纯 Triton 回退 kernel."""
    import importlib

    import flag_gems.ops  # noqa: F401

    solve_mod = importlib.import_module("flag_gems.ops.linalg_solve_triangular")

    monkeypatch.setattr(solve_mod, "HAS_TLE", False)

    A = _make_triangular(
        (n, n), dtype, flag_gems.device, upper=upper, unitriangular=False
    )
    B = torch.randn(n, n, dtype=dtype, device=flag_gems.device)

    ref_A = utils.to_reference(A)
    ref_B = utils.to_reference(B)
    ref_out = torch.linalg.solve_triangular(ref_A, ref_B, upper=upper)

    res_out = solve_mod.linalg_solve_triangular(A, B, upper=upper)

    utils.gems_assert_close(res_out, ref_out, dtype)
