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
    expected = torch.ops.aten._cholesky_solve_helper(ref_rhs, ref_factor, upper)

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
