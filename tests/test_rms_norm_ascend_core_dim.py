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

"""coreDim (launch grid) limit regression for the Ascend rms_norm kernels.

Ascend NPUs reject a launch whose coreDim exceeds UINT16_MAX (65535). rms_norm
launches one program per row, so any shape with more than 65535 rows used to
fail with::

    KernelLaunch failed because value 786432 for parameter coreDim is invalid

The kernels now cap the grid and walk the remaining rows/tiles inside the
kernel, which is what these tests pin down: they assert on the grid that is
actually launched, not only on the numerical result.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

pytestmark = [
    pytest.mark.rms_norm,
    pytest.mark.skipif(
        flag_gems.vendor_name != "ascend", reason="Ascend launch-grid regression"
    ),
]

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
    saved = {name: ascend_rms_norm[name] for name in names}
    recs = {name: _GridRecorder(saved[name]) for name in names}
    try:
        ascend_rms_norm.update(recs)
        _check(M, N, torch.float16)
    finally:
        ascend_rms_norm.update(saved)

    for name, rec in recs.items():
        assert len(rec.grids) == 1, f"{name} launched {len(rec.grids)} times"
        grid = rec.grids[0]
        # NPU collapses a 2D grid to 1D, so the product is the coreDim.
        core_dim = 1
        for dim in grid:
            core_dim *= dim
        assert 0 < core_dim <= MAX_CORE_DIM, f"{name} launched coreDim={core_dim}"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_rms_norm_ascend_core_dim_capped(ascend_rms_norm, monkeypatch, dtype):
    # Cap the grid far below M so every program walks many rows; this covers the
    # in-kernel row loop itself, independent of the real hardware limit.
    monkeypatch.setitem(ascend_rms_norm, "MAX_CORE_DIM", 3)
    _check(1024, 128, dtype)


@pytest.mark.parametrize("M, N", [(2048, 128), (1, 4096), (3, 1)])
def test_rms_norm_ascend_small_shapes(ascend_rms_norm, M, N):
    # M below the cap must keep the one-row-per-program path intact.
    _check(M, N, torch.float32)
