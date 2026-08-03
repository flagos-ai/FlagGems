import pytest
import torch
import triton

import flag_gems
from flag_gems.ops.rms_norm import (
    _DW_COL_BLOCK_SIZE,
    _DW_ROW_BLOCK_SIZE,
    _DW_TARGET_LAYOUT,
    _DW_TLE_NUM_WARPS,
    _dw_tle_available,
    rms_norm_grad_dw_kernel,
    rms_norm_grad_dw_kernel_tle,
)
from flag_gems.utils.triton_version_utils import HAS_TLE

device = flag_gems.device


def _has_tle_hw():
    if not (HAS_TLE and torch.cuda.is_available()):
        return False
    return torch.cuda.get_device_capability()[0] >= 9


pytestmark = pytest.mark.skipif(
    not _has_tle_hw(),
    reason="requires Triton with TLE support on Hopper+ (capability >= 9)",
)


def _run_dw_kernel(kernel_fn, X, DY, INV_RMS, M, N, extra_kwargs=None):
    extra_kwargs = extra_kwargs or {}
    row_block_num = triton.cdiv(M, _DW_ROW_BLOCK_SIZE)
    col_block_num = triton.cdiv(N, _DW_COL_BLOCK_SIZE)
    DW = torch.empty((row_block_num, N), dtype=torch.float32, device=device)
    grid = (row_block_num, col_block_num)

    kernel_fn[grid](
        X,
        DY,
        INV_RMS,
        DW,
        N,
        1,
        N,
        1,
        M,
        N,
        _DW_ROW_BLOCK_SIZE,
        _DW_COL_BLOCK_SIZE,
        **extra_kwargs,
    )
    return torch.sum(DW, dim=0, dtype=torch.float32)


def _make_dw_inputs(M, N, dtype, seed=0):
    torch.manual_seed(seed)
    X = torch.randn(M, N, dtype=dtype, device=device)
    DY = torch.randn(M, N, dtype=dtype, device=device)
    INV_RMS = torch.rand(M, dtype=torch.float32, device=device) + 0.5
    return X, DY, INV_RMS


DW_SHAPES = [
    (16, 256),
    (1024, 4096),
    (1000, 4096),
    (1024, 4000),
    (17, 300),
    (4096, 4096),
]


@pytest.mark.rms_norm
@pytest.mark.parametrize("M,N", DW_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_dw_tle_matches_baseline(M, N, dtype):
    X, DY, INV_RMS = _make_dw_inputs(M, N, dtype)

    dw_base = _run_dw_kernel(rms_norm_grad_dw_kernel, X, DY, INV_RMS, M, N)
    dw_tle = _run_dw_kernel(
        rms_norm_grad_dw_kernel_tle,
        X,
        DY,
        INV_RMS,
        M,
        N,
        extra_kwargs={
            "TARGET_LAYOUT": _DW_TARGET_LAYOUT,
            "num_warps": _DW_TLE_NUM_WARPS,
        },
    )

    rtol = 1e-3 if dtype == torch.float32 else 1e-2
    atol = 1e-3 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(dw_tle, dw_base, rtol=rtol, atol=atol)


@pytest.mark.rms_norm
def test_dw_tle_available_reflects_hardware():
    x_cuda = torch.zeros(1, device=device)
    assert _dw_tle_available(x_cuda) is True

    x_cpu = torch.zeros(1, device="cpu")
    assert _dw_tle_available(x_cpu) is False


def _torch_rms_norm(x, weight, eps):
    upcast_x = x.to(torch.float32)
    variance = upcast_x.pow(2).mean(-1, keepdim=True)
    hidden_states = upcast_x * torch.rsqrt(variance + eps).to(torch.float32)
    hidden_states = hidden_states.to(x.dtype)
    return weight * hidden_states


@pytest.mark.rms_norm
@pytest.mark.parametrize("M,N", [(1024, 4096), (1000, 4096), (4096, 4096)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_rms_norm_end_to_end_with_tle_dispatch(M, N, dtype):
    torch.manual_seed(0)
    eps = 1e-5

    inp = torch.randn(M, N, dtype=dtype, device=device, requires_grad=True)
    weight = torch.randn(N, dtype=dtype, device=device, requires_grad=True)
    grad_out = torch.randn(M, N, dtype=dtype, device=device)

    ref_inp = inp.detach().clone().float().requires_grad_()
    ref_weight = weight.detach().clone().float().requires_grad_()

    res_out = flag_gems.rms_norm(inp, [N], weight=weight, eps=eps)
    ref_out = _torch_rms_norm(ref_inp, ref_weight, eps).to(dtype)

    res_grad, res_weight_grad = torch.autograd.grad(res_out, (inp, weight), grad_out)
    ref_grad, ref_weight_grad_f32 = torch.autograd.grad(
        ref_out.float(), (ref_inp, ref_weight), grad_out.float()
    )

    out_tol = (
        dict(rtol=1e-2, atol=1e-2)
        if dtype == torch.float16
        else dict(rtol=1e-3, atol=1e-3)
    )
    dw_tol = (
        dict(rtol=2e-2, atol=2e-2)
        if dtype == torch.float16
        else dict(rtol=1e-3, atol=1e-3)
    )

    torch.testing.assert_close(res_out.float(), ref_out.float(), **out_tol)
    torch.testing.assert_close(res_grad.float(), ref_grad.float(), **out_tol)
    torch.testing.assert_close(
        res_weight_grad.float(), ref_weight_grad_f32.float(), **dw_tol
    )
