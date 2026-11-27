# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import numpy as np
import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES


@pytest.mark.rms_norm
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_rms_norm(shape, dtype):
    N = shape[1]
    layer_shape = [
        N,
    ]
    np.random.seed(0)
    np_inp = np.random.uniform(-0.1, 0.1, shape[:2]).astype(np.float32)
    np_grad = np.random.uniform(-0.01, 0.01, shape[:2]).astype(np.float32)
    np_weight = np.random.uniform(-0.1, 0.1, layer_shape).astype(np.float32)

    inp = torch.tensor(np_inp, dtype=dtype, device=flag_gems.device, requires_grad=True)
    weight = torch.tensor(
        np_weight, dtype=dtype, device=flag_gems.device, requires_grad=True
    )

    eps = 1e-5

    ref_inp = utils.to_reference(inp)
    ref_weight = utils.to_reference(weight)

    def _torch_rms_norm(x, weight, eps):
        upcast_x = x.to(torch.float32)
        variance = upcast_x.pow(2).mean(-1, keepdim=True)
        hidden_states = upcast_x * torch.rsqrt(variance + eps).to(torch.float32)
        hidden_states = hidden_states.to(x.dtype)
        return weight * hidden_states

    ref_out = _torch_rms_norm(ref_inp, weight=ref_weight, eps=eps)
    res_out = flag_gems.rms_norm(inp, list(layer_shape), weight=weight, eps=eps)

    res_grad = torch.tensor(
        np_grad, dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    ref_grad = utils.to_reference(res_grad)

    res_grad, res_weight_grad = torch.autograd.grad(res_out, (inp, weight), res_grad)
    ref_grad, ref_weight_grad = torch.autograd.grad(
        ref_out, (ref_inp, ref_weight), ref_grad
    )

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(res_grad, ref_grad, dtype)
    utils.gems_assert_close(res_weight_grad, ref_weight_grad, dtype, reduce_dim=N)


# Ascend coreDim and large-tensor regressions.
MAX_CORE_DIM = 65535


@pytest.fixture(scope="module")
def ascend_rms_norm():
    """Globals of the module that actually launches the kernels.

    The Ascend backend loads its op files under a second, private module name,
    so ``importlib.import_module`` returns a different module object than the
    one ``flag_gems.rms_norm`` closes over; patching the import would be a no-op.
    """
    return flag_gems.rms_norm.__globals__


class _GridRecorder:
    """Stand-in for a launched kernel that records the grid it was given."""

    def __init__(self, kernel):
        self._kernel = kernel
        self.grids = []

    def __getitem__(self, grid):
        if isinstance(grid, int):
            grid = (grid,)
        self.grids.append(tuple(grid))
        return self._kernel[grid]


def _reference(x, weight, dy, eps):
    """RMSNorm forward and backward in fp32, written out explicitly.

    Kept free of autograd so that the large-M cases below do not need the
    saved activations twice.
    """
    xf = x.float()
    rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    y = (xf * rstd) * weight.float()

    dyf = dy.float()
    z = dyf * weight.float()
    proj = (z * xf).sum(-1, keepdim=True) / x.shape[-1]
    dx = (z - xf * rstd.pow(2) * proj) * rstd
    # dL/dw_j = sum_i dy_ij * x_ij * rstd_i; unlike dx this carries no weight_j.
    dw = (dyf * xf * rstd).sum(0)
    return y, dx, dw


def _check(M, N, dtype):
    eps = 1e-5
    torch.manual_seed(0)
    x = torch.randn((M, N), dtype=dtype, device=flag_gems.device).requires_grad_(True)
    weight = torch.randn((N,), dtype=dtype, device=flag_gems.device).requires_grad_(
        True
    )
    dy = torch.randn((M, N), dtype=dtype, device=flag_gems.device)
    ref_y, ref_dx, ref_dw = _reference(
        utils.to_reference(x.detach()),
        utils.to_reference(weight.detach()),
        utils.to_reference(dy),
        eps,
    )

    out = flag_gems.rms_norm(x, [N], weight, eps)
    res_y = out.detach()
    out.backward(dy)
    res_dx, res_dw = x.grad, weight.grad

    utils.gems_assert_close(res_y, ref_y, dtype, reduce_dim=N)
    utils.gems_assert_close(res_dx, ref_dx, dtype, reduce_dim=N)
    # d_weight sums M rows, so its magnitude grows with M and blows past the
    # fp16 range long before 65535 rows; compare it in fp32 instead. A sum of M
    # fp16 products only reorders it, so allow a relative tolerance; a strict
    # 1e-6 rtol would gate on summation order rather than on correctness.
    assert res_dw.dtype == dtype
    torch.testing.assert_close(
        utils.to_cpu(res_dw, ref_dw).float(),
        ref_dw,
        rtol=1e-2,
        atol=1e-2 * ref_dw.abs().max(),
    )
    return res_y, res_dx, res_dw


@pytest.mark.rms_norm
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="Ascend launch-grid regression"
)
@pytest.mark.parametrize(
    "M, N",
    [
        (65536, 512),  # 65536 rows: one row past the limit.
        (786432, 512),  # The reported coreDim, a realistic token count.
        (70000, 1024),
    ],
)
def test_rms_norm_ascend_core_dim_boundary(ascend_rms_norm, M, N):
    assert M > MAX_CORE_DIM

    names = [
        "rms_norm_kernel",
        "rms_norm_grad_dx_kernel",
        "rms_norm_grad_dw_kernel",
    ]
    # On a TensorView-capable build the forward is served by the tiled kernel
    # instead, which has to cap its grid for the same reason. Record it too and
    # require the forward to come from exactly one of the two.
    forward_names = {"rms_norm_kernel"}
    if "rms_norm_tv_kernel" in ascend_rms_norm:
        names.append("rms_norm_tv_kernel")
        forward_names.add("rms_norm_tv_kernel")

    saved = {name: ascend_rms_norm[name] for name in names}
    recs = {name: _GridRecorder(saved[name]) for name in names}
    try:
        ascend_rms_norm.update(recs)
        _check(M, N, torch.float16)
    finally:
        ascend_rms_norm.update(saved)

    forward_launches = sum(len(recs[name].grids) for name in forward_names)
    assert forward_launches == 1, f"forward launched {forward_launches} times"

    for name, rec in recs.items():
        if name not in forward_names:
            assert len(rec.grids) == 1, f"{name} launched {len(rec.grids)} times"
        for grid in rec.grids:
            # NPU collapses a 2D grid to 1D, so the product is the coreDim.
            core_dim = 1
            for dim in grid:
                core_dim *= dim
            assert 0 < core_dim <= MAX_CORE_DIM, f"{name} launched coreDim={core_dim}"


@pytest.mark.rms_norm
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="Ascend launch-grid regression"
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_rms_norm_ascend_core_dim_capped(ascend_rms_norm, monkeypatch, dtype):
    # Cap the grid far below M so every program walks many rows; this covers the
    # in-kernel row loop itself, independent of the real hardware limit.
    monkeypatch.setitem(ascend_rms_norm, "MAX_CORE_DIM", 3)
    _check(1024, 128, dtype)


@pytest.mark.rms_norm
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="Ascend launch-grid regression"
)
@pytest.mark.parametrize("M, N", [(2048, 128), (1, 4096), (3, 1)])
def test_rms_norm_ascend_small_shapes(ascend_rms_norm, M, N):
    # M below the cap must keep the one-row-per-program path intact.
    _check(M, N, torch.float32)


@pytest.mark.rms_norm
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend RMSNorm")
@pytest.mark.parametrize(
    "shape,normalized_shape", [((0, 128), (128,)), ((2, 0), (0,)), ((0, 3, 4), (3, 4))]
)
def test_rms_norm_ascend_review_empty(
    ascend_rms_norm, monkeypatch, shape, normalized_shape
):
    class NoLaunch:
        def __getitem__(self, grid):
            pytest.fail(f"empty RMSNorm launched grid {grid}")

    for name in (
        "rms_norm_kernel",
        "rms_norm_grad_dx_kernel",
        "rms_norm_grad_dw_kernel",
    ):
        monkeypatch.setitem(ascend_rms_norm, name, NoLaunch())
    x = torch.empty(
        shape, device=flag_gems.device, dtype=torch.float16, requires_grad=True
    )
    weight = torch.ones(
        normalized_shape, device=x.device, dtype=torch.float32, requires_grad=True
    )
    y, inv = ascend_rms_norm["rms_norm_forward"](x, normalized_shape, weight)
    assert y.shape == x.shape
    dx, dw = ascend_rms_norm["rms_norm_backward"](
        torch.empty_like(x), x, inv, normalized_shape, weight
    )
    assert dx.shape == x.shape and dw.shape == weight.shape and dw.dtype == weight.dtype
    result = flag_gems.rms_norm(x, normalized_shape, weight)
    result.backward(torch.empty_like(x))
    torch.npu.synchronize()
    torch.testing.assert_close(dw.cpu(), torch.zeros_like(weight.cpu()))
    torch.testing.assert_close(weight.grad.cpu(), torch.zeros_like(weight.cpu()))


@pytest.mark.rms_norm
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend RMSNorm")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "shape,normalized_shape", [((1024, 16), (16,)), ((2, 3, 4), (3, 4))]
)
def test_rms_norm_ascend_review_weight_gradient(dtype, shape, normalized_shape):
    x_cpu = torch.ones(shape, dtype=torch.float64, requires_grad=True)
    w_cpu = torch.ones(normalized_shape, dtype=torch.float64, requires_grad=True)
    ref = torch.nn.functional.rms_norm(x_cpu, normalized_shape, w_cpu, eps=1e-5)
    ref.backward(torch.full_like(ref, 128))
    x = x_cpu.detach().to(device=flag_gems.device, dtype=dtype).requires_grad_()
    weight = w_cpu.detach().to(device=x.device, dtype=torch.float32).requires_grad_()
    result = flag_gems.rms_norm(x, normalized_shape, weight, eps=1e-5)
    result.backward(torch.full_like(result, 128))
    torch.npu.synchronize()
    assert weight.grad.shape == weight.shape and weight.grad.dtype == weight.dtype
    assert torch.isfinite(weight.grad).all()
    utils.gems_assert_close(
        result, ref.detach().float(), dtype, reduce_dim=math.prod(normalized_shape)
    )
    torch.testing.assert_close(
        weight.grad.cpu(), w_cpu.grad.float(), rtol=1e-5, atol=1e-4
    )
    torch.testing.assert_close(
        x.grad.cpu().float(), x_cpu.grad.float(), rtol=1e-2, atol=3e-5
    )
