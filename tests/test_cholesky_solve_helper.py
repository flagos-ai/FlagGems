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

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

SUPPORT_FP64 = flag_gems.runtime.device.support_fp64


def _factor(shape, dtype):
    matrix = torch.randn(shape, device=flag_gems.device, dtype=dtype)
    identity = torch.eye(shape[-1], device=flag_gems.device, dtype=dtype)
    return torch.linalg.cholesky(matrix @ matrix.mT + 0.5 * identity)


# Covers unbatched, batched, and multiple-RHS cases. The raw aten helper does
# not broadcast a 2D factor across the rhs batch dim (that happens in the
# higher-level cholesky_solve), so factor and rhs share matching batch dims.
CHOLESKY_SOLVE_CASES = [
    ((5, 3), (5, 5)),
    ((2, 5, 3), (2, 5, 5)),
    ((3, 8, 4), (3, 8, 8)),
]


@pytest.mark.cholesky_solve_helper
@pytest.mark.parametrize("rhs_shape,factor_shape", CHOLESKY_SOLVE_CASES)
@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.parametrize(
    "dtype", [torch.float32] + ([torch.float64] if SUPPORT_FP64 else [])
)
def test_cholesky_solve_helper(rhs_shape, factor_shape, upper, dtype):
    factor = _factor(factor_shape, dtype)
    if upper:
        factor = factor.mT.contiguous()
    rhs = torch.randn(rhs_shape, device=flag_gems.device, dtype=dtype)
    ref_rhs = utils.to_reference(rhs)
    ref_factor = utils.to_reference(factor)
    expected = torch.ops.aten._cholesky_solve_helper(ref_rhs, ref_factor, upper)

    actual = flag_gems._cholesky_solve_helper(rhs, factor, upper)

    utils.gems_assert_close(actual, expected, dtype, reduce_dim=rhs_shape[-2])


@pytest.mark.cholesky_solve_helper
@pytest.mark.parametrize(
    "n,nrhs,batch_shape",
    [
        (1, 1, ()),
        (3, 2, ()),
        (4, 1, ()),
        (7, 3, (2,)),
        (8, 2, ()),
        (9, 5, (2, 3)),
        (16, 4, ()),
        (17, 1, ()),
        (31, 7, ()),
        (32, 4, ()),
        (33, 3, (2,)),
        (63, 9, ()),
        (64, 8, ()),
        (65, 3, ()),
        (96, 2, ()),
    ],
)
@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.parametrize("layout", ["contiguous", "transposed", "sliced"])
def test_cholesky_solve_helper_layouts(n, nrhs, batch_shape, upper, layout):
    # Build the system on CPU so neither the device RNG nor device Cholesky can
    # hide an error in the solver. Poison the unused triangle deliberately.
    generator = torch.Generator().manual_seed(100 * n + nrhs)
    matrix = torch.randn((*batch_shape, n, n), generator=generator)
    factor = torch.linalg.cholesky(matrix @ matrix.mT + 0.5 * torch.eye(n))
    if upper:
        factor = factor.mT.contiguous()
        unused = torch.ones((n, n), dtype=torch.bool).tril(-1)
    else:
        unused = torch.ones((n, n), dtype=torch.bool).triu(1)
    factor[..., unused] = float("nan")
    rhs = torch.randn((*batch_shape, n, nrhs), generator=generator)
    expected = torch.ops.aten._cholesky_solve_helper(rhs, factor, upper)

    if layout == "transposed":
        device_rhs = rhs.mT.contiguous().to(flag_gems.device).mT
        device_factor = factor.mT.contiguous().to(flag_gems.device).mT
    elif layout == "sliced":
        device_rhs = torch.empty(
            (*batch_shape, 2 * n, 2 * nrhs), device=flag_gems.device
        )[..., ::2, ::2]
        device_factor = torch.empty(
            (*batch_shape, 2 * n, 2 * n), device=flag_gems.device
        )[..., ::2, ::2]
        device_rhs.copy_(rhs)
        device_factor.copy_(factor)
    else:
        device_rhs = rhs.to(flag_gems.device)
        device_factor = factor.to(flag_gems.device)

    actual = flag_gems._cholesky_solve_helper(device_rhs, device_factor, upper)
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(device_rhs.cpu(), rhs)
    torch.testing.assert_close(device_factor.cpu(), factor, equal_nan=True)


@pytest.mark.cholesky_solve_helper
@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.parametrize("preexpanded", [False, True])
def test_cholesky_solve_helper_batch_strides(upper, preexpanded):
    generator = torch.Generator().manual_seed(341)
    matrix = torch.randn((2, 1, 9, 9), generator=generator)
    factor = torch.linalg.cholesky(matrix @ matrix.mT + torch.eye(9))
    if upper:
        factor = factor.mT
    rhs = torch.randn((3, 2, 9, 5), generator=generator).transpose(0, 1)
    device_rhs = rhs.transpose(0, 1).to(flag_gems.device).transpose(0, 1)
    device_factor = factor.to(flag_gems.device)
    if preexpanded:
        device_factor = device_factor.expand(2, 3, 9, 9)
    expected = torch.ops.aten._cholesky_solve_helper(
        rhs, factor.expand(2, 3, 9, 9), upper
    )
    actual = flag_gems._cholesky_solve_helper(device_rhs, device_factor, upper)
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-5)


@pytest.mark.cholesky_solve_helper
@pytest.mark.parametrize("shape", [(0, 3), (4, 0), (0, 4, 2)])
def test_cholesky_solve_helper_empty(shape):
    rhs = torch.empty(shape, device=flag_gems.device)
    factor = torch.empty((*shape[:-2], shape[-2], shape[-2]), device=flag_gems.device)
    actual = flag_gems._cholesky_solve_helper(rhs, factor, False)
    assert actual.shape == rhs.shape
    assert actual.dtype == rhs.dtype
    assert actual.device == rhs.device
