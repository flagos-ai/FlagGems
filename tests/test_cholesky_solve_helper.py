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


def _cpu_cholesky(matrix):
    if flag_gems.vendor_name == "mthreads":
        import numpy as np

        return torch.from_numpy(np.linalg.cholesky(matrix.numpy()))
    return torch.linalg.cholesky(matrix)


def _solve_reference(rhs, factor, upper):
    if flag_gems.vendor_name == "mthreads" and rhs.device.type == "cpu":
        import numpy as np

        triangle = np.triu(factor.numpy()) if upper else np.tril(factor.numpy())
        transposed = triangle.swapaxes(-1, -2)
        first, second = (transposed, triangle) if upper else (triangle, transposed)
        return torch.from_numpy(np.linalg.solve(second, np.linalg.solve(first, rhs.numpy())))
    return torch.ops.aten._cholesky_solve_helper(rhs, factor, upper)


def _factor(shape, dtype):
    if flag_gems.vendor_name == "kunlunxin" and dtype == torch.float64:
        pytest.skip("Kunlunxin target storage converts float64 to float32")
    torch.manual_seed(0)
    device = (
        "cpu"
        if flag_gems.vendor_name in ("kunlunxin", "iluvatar", "ascend", "thead")
        else flag_gems.device
    )
    matrix = torch.randn(shape, device=device, dtype=dtype)
    identity = torch.eye(shape[-1], device=device, dtype=dtype)
    factor = torch.linalg.cholesky(matrix @ matrix.mT + 0.5 * identity).to(
        flag_gems.device
    )
    assert factor.dtype == dtype
    return factor


# Covers unbatched, batched, multiple-RHS, and both directions of batch
# broadcasting implemented through torch.broadcast_shapes and expand.
CHOLESKY_SOLVE_CASES = [
    ((5, 3), (5, 5)),
    ((2, 5, 3), (2, 5, 5)),
    ((3, 8, 4), (3, 8, 8)),
    ((2, 5, 3), (5, 5)),
    ((1, 5, 3), (2, 5, 5)),
    ((2, 1, 5, 3), (1, 3, 5, 5)),
]


@pytest.mark.cholesky_solve_helper
@pytest.mark.underscore_cholesky_solve_helper
@pytest.mark.parametrize("rhs_shape,factor_shape", CHOLESKY_SOLVE_CASES)
@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cholesky_solve_helper(rhs_shape, factor_shape, upper, dtype):
    if flag_gems.vendor_name == "enflame" and dtype == torch.float64:
        pytest.skip("Enflame Float64 target storage loses adjacent-double values")
    if (
        flag_gems.vendor_name == "iluvatar"
        and dtype == torch.float64
        and not utils.TO_CPU
    ):
        pytest.skip("Iluvatar native float64 Cholesky solve returns incorrect values")
    factor = _factor(factor_shape, dtype)
    if upper:
        factor = factor.mT.contiguous()
    torch.manual_seed(1)
    rhs_device = (
        "cpu"
        if flag_gems.vendor_name in ("iluvatar", "ascend", "thead")
        and dtype == torch.float64
        else flag_gems.device
    )
    rhs = torch.randn(rhs_shape, device=rhs_device, dtype=dtype).to(flag_gems.device)
    assert rhs.dtype == dtype
    ref_rhs = utils.to_reference(rhs, dtype != torch.float64)
    ref_factor = utils.to_reference(factor, dtype != torch.float64)
    # The private native helper expects batch broadcasting to be done by its
    # caller. Keep the original inputs for the candidate's broadcasting checks.
    batch_shape = torch.broadcast_shapes(ref_rhs.shape[:-2], ref_factor.shape[:-2])
    ref_rhs = ref_rhs.expand(*batch_shape, *ref_rhs.shape[-2:]).contiguous()
    ref_factor = ref_factor.expand(*batch_shape, *ref_factor.shape[-2:]).contiguous()
    expected = _solve_reference(ref_rhs, ref_factor, upper)

    actual = flag_gems._cholesky_solve_helper(rhs, factor, upper)

    assert actual.device == rhs.device
    assert actual.dtype == dtype
    assert actual.shape == (*batch_shape, *rhs.shape[-2:])
    utils.gems_assert_close(actual, expected, dtype, reduce_dim=rhs_shape[-2])


@pytest.mark.cholesky_solve_helper
@pytest.mark.underscore_cholesky_solve_helper
def test_cholesky_solve_helper_incompatible_batch_error():
    torch.manual_seed(0)
    rhs = torch.randn((2, 5, 3), dtype=torch.float32, device=flag_gems.device)
    factor = _factor((3, 5, 5), torch.float32)

    with pytest.raises(RuntimeError):
        flag_gems._cholesky_solve_helper(rhs, factor, False)


@pytest.mark.cholesky_solve_helper
@pytest.mark.underscore_cholesky_solve_helper
def test_cholesky_solve_helper_dtype_error():
    torch.manual_seed(0)
    rhs = torch.randn((5, 2), dtype=torch.float32, device=flag_gems.device)
    factor = _factor((5, 5), torch.float64)

    with pytest.raises(RuntimeError, match="same dtype and device"):
        flag_gems._cholesky_solve_helper(rhs, factor, False)


@pytest.mark.cholesky_solve_helper
@pytest.mark.underscore_cholesky_solve_helper
@pytest.mark.skipif(
    torch.device(flag_gems.device).type == "cpu",
    reason="device-mismatch coverage requires a non-CPU FlagGems target",
)
def test_cholesky_solve_helper_device_error():
    torch.manual_seed(0)
    rhs = torch.randn((5, 2), dtype=torch.float32, device=flag_gems.device)
    factor = torch.eye(5, dtype=torch.float32, device="cpu")

    with pytest.raises(RuntimeError, match="same dtype and device"):
        flag_gems._cholesky_solve_helper(rhs, factor, False)


@pytest.mark.cholesky_solve_helper
@pytest.mark.underscore_cholesky_solve_helper
@pytest.mark.parametrize(
    "rhs_shape,factor_shape",
    [
        ((5,), (5, 5)),
        ((5, 2), (5,)),
        ((5, 2), (5, 4)),
        ((4, 2), (5, 5)),
    ],
)
def test_cholesky_solve_helper_invalid_rank_or_matrix_error(rhs_shape, factor_shape):
    torch.manual_seed(0)
    rhs = torch.randn(rhs_shape, dtype=torch.float32, device=flag_gems.device)
    factor = torch.randn(factor_shape, dtype=torch.float32, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems._cholesky_solve_helper(rhs, factor, False)


@pytest.mark.cholesky_solve_helper
@pytest.mark.underscore_cholesky_solve_helper
def test_cholesky_solve_helper_autograd_rejected():
    factor = _factor((5, 5), torch.float32).requires_grad_(True)
    torch.manual_seed(1)
    rhs = torch.randn(
        (5, 2), dtype=torch.float32, device=flag_gems.device, requires_grad=True
    )

    result = flag_gems._cholesky_solve_helper(rhs, factor, False)

    # The raw helper writes a detached Triton output and has no custom autograd
    # Function, so gradients for either tensor input are explicitly unsupported.
    assert not result.requires_grad
    with pytest.raises(RuntimeError, match="does not require grad"):
        torch.autograd.grad(result.sum(), (rhs, factor))


@pytest.mark.cholesky_solve_helper
@pytest.mark.underscore_cholesky_solve_helper
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
    factor = _cpu_cholesky(matrix @ matrix.mT + 0.5 * torch.eye(n))
    if upper:
        factor = factor.mT.contiguous()
        unused = torch.ones((n, n), dtype=torch.bool).tril(-1)
    else:
        unused = torch.ones((n, n), dtype=torch.bool).triu(1)
    factor[..., unused] = float("nan")
    rhs = torch.randn((*batch_shape, n, nrhs), generator=generator)
    expected = _solve_reference(rhs, factor, upper)

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
    assert actual.device == device_rhs.device
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(device_rhs.cpu(), rhs)
    torch.testing.assert_close(device_factor.cpu(), factor, equal_nan=True)


@pytest.mark.cholesky_solve_helper
@pytest.mark.underscore_cholesky_solve_helper
@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.parametrize("preexpanded", [False, True])
def test_cholesky_solve_helper_batch_strides(upper, preexpanded):
    generator = torch.Generator().manual_seed(341)
    matrix = torch.randn((2, 1, 9, 9), generator=generator)
    factor = _cpu_cholesky(matrix @ matrix.mT + torch.eye(9))
    if upper:
        factor = factor.mT
    rhs = torch.randn((3, 2, 9, 5), generator=generator).transpose(0, 1)
    device_rhs = rhs.transpose(0, 1).to(flag_gems.device).transpose(0, 1)
    device_factor = factor.to(flag_gems.device)
    if preexpanded:
        device_factor = device_factor.expand(2, 3, 9, 9)
    expected = _solve_reference(
        rhs, factor.expand(2, 3, 9, 9), upper
    )
    actual = flag_gems._cholesky_solve_helper(device_rhs, device_factor, upper)
    assert actual.device == device_rhs.device
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-5)


@pytest.mark.cholesky_solve_helper
@pytest.mark.underscore_cholesky_solve_helper
@pytest.mark.parametrize("shape", [(0, 3), (4, 0), (0, 4, 2)])
def test_cholesky_solve_helper_empty(shape):
    rhs = torch.empty(shape, device=flag_gems.device)
    factor = torch.empty((*shape[:-2], shape[-2], shape[-2]), device=flag_gems.device)
    actual = flag_gems._cholesky_solve_helper(rhs, factor, False)
    assert actual.shape == rhs.shape
    assert actual.dtype == rhs.dtype
    assert actual.device == rhs.device
