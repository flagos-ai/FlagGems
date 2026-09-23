# Copyright 2026, The FlagOS Contributors.
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

# torch.inverse rejects the low precision dtypes, so only fp32/fp64 are covered.
INVERSE_DTYPES = [torch.float32]
if flag_gems.runtime.device.support_fp64:
    INVERSE_DTYPES.append(torch.float64)

# Square matrices covering the register-resident kernel (n <= 64) and the
# delegated path (n > 64).
INVERSE_SHAPES = [
    (2, 2),
    (3, 3),
    (8, 8),
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
]

# Batched stacks, including a non-power-of-two side to exercise identity padding.
INVERSE_BATCH_SHAPES = [
    (2, 3, 3),
    (4, 8, 8),
    (2, 16, 16),
]

# Explicit permutation matrices: the zero pivot in the first column means an
# implementation without partial pivoting cannot invert them at all.
INVERSE_PIVOT_MATRICES = [
    [[0.0, 1.0], [1.0, 0.0]],
    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
    [[0.0, 2.0, 1.0], [1.0, 0.0, 3.0], [4.0, 1.0, 0.0]],
]


def _make_well_conditioned(shape, dtype, device):
    """Identity plus a small perturbation: always invertible, never singular."""
    n = shape[-1]
    eye = torch.eye(n, dtype=dtype, device=device)
    if len(shape) > 2:
        eye = eye.expand(shape).contiguous()
    return eye + 0.05 * torch.randn(shape, dtype=dtype, device=device)


@pytest.mark.inverse
@pytest.mark.parametrize("shape", INVERSE_SHAPES)
@pytest.mark.parametrize("dtype", INVERSE_DTYPES)
def test_inverse(shape, dtype):
    # Inversion is ill conditioned by nature: the reference is upcast to fp64
    # so the comparison measures our error, not the reference's.
    A = _make_well_conditioned(shape, dtype, flag_gems.device)
    ref_A = utils.to_reference(A, upcast=True)

    res_out = flag_gems.inverse(A)
    ref_out = torch.inverse(ref_A).to(dtype)

    assert res_out.shape == A.shape
    utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-3)


@pytest.mark.inverse
@pytest.mark.parametrize("shape", INVERSE_BATCH_SHAPES)
@pytest.mark.parametrize("dtype", INVERSE_DTYPES)
def test_inverse_batch(shape, dtype):
    A = _make_well_conditioned(shape, dtype, flag_gems.device)
    ref_A = utils.to_reference(A, upcast=True)

    res_out = flag_gems.inverse(A)
    ref_out = torch.inverse(ref_A).to(dtype)

    assert res_out.shape == A.shape
    utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-3)


@pytest.mark.inverse
@pytest.mark.parametrize("matrix", INVERSE_PIVOT_MATRICES)
@pytest.mark.parametrize("dtype", INVERSE_DTYPES)
def test_inverse_requires_pivoting(matrix, dtype):
    # Rows must be reordered for these matrices, so this pins down the
    # partial pivoting step rather than just the elimination arithmetic.
    A = torch.tensor(matrix, dtype=dtype, device=flag_gems.device)
    ref_A = utils.to_reference(A, upcast=True)

    res_out = flag_gems.inverse(A)
    ref_out = torch.inverse(ref_A).to(dtype)

    utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-3)


@pytest.mark.inverse
@pytest.mark.parametrize("dtype", INVERSE_DTYPES)
def test_inverse_identity_product(dtype):
    # A @ A^-1 should be the identity; this checks the actual inverse
    # relationship instead of trusting the elementwise reference.
    n = 32
    A = _make_well_conditioned((n, n), dtype, flag_gems.device)

    res_out = flag_gems.inverse(A)

    # Both operands upcast so the residual reflects our error, not the
    # low-precision product.
    resid = utils.to_reference(A, upcast=True) @ utils.to_reference(
        res_out, upcast=True
    )
    resid = resid - torch.eye(n, dtype=resid.dtype, device=resid.device)
    assert resid.abs().max().item() < 1e-4


@pytest.mark.inverse
@pytest.mark.parametrize("dtype", INVERSE_DTYPES)
def test_inverse_noncontiguous(dtype):
    # A transposed view has non-unit row strides; the wrapper must materialize
    # it before the kernel reads the tile.
    A = _make_well_conditioned((64, 64), dtype, flag_gems.device)
    A_t = A.t()

    ref_A = utils.to_reference(A_t, upcast=True)

    res_out = flag_gems.inverse(A_t)
    ref_out = torch.inverse(ref_A).to(dtype)

    assert res_out.shape == A_t.shape
    utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-3)


@pytest.mark.inverse
@pytest.mark.parametrize("dtype", INVERSE_DTYPES)
def test_inverse_empty(dtype):
    A = torch.empty((0, 3, 3), dtype=dtype, device=flag_gems.device)

    res_out = flag_gems.inverse(A)

    assert res_out.shape == A.shape


@pytest.mark.inverse
def test_inverse_unsupported_dtype():
    A = torch.randn((4, 4), dtype=torch.float16, device=flag_gems.device)

    with pytest.raises(AssertionError):
        flag_gems.inverse(A)
