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

# Batched, small and medium square matrices. 64 is the register/tiled boundary
# in the implementation, so it is covered on both sides.
SLOGDET_SHAPES = [(2, 3, 3), (4, 4), (8, 8), (16, 16), (32, 32), (64, 64)]

REAL_DTYPES = [torch.float32, torch.float64]
COMPLEX_DTYPES = [torch.complex64, torch.complex128]


def _tols(dtype, n):
    """Tolerances scaled by matrix size: LU error grows with the pivot count."""
    if dtype in (torch.float32, torch.complex64):
        return {"rtol": 2e-4 * max(1, n / 8), "atol": 2e-4 * max(1, n / 8)}
    return {"rtol": 1e-9, "atol": 1e-9}


@pytest.mark.slogdet
@pytest.mark.parametrize("shape", SLOGDET_SHAPES)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_slogdet(shape, dtype):
    """Accuracy against the ATen reference for real dtypes."""
    A = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_A = utils.to_reference(A)

    ref = torch.linalg.slogdet(ref_A)
    res_sign, res_logabsdet = flag_gems.slogdet(A)

    n = shape[-1]
    torch.testing.assert_close(
        res_sign.cpu(), ref.sign.cpu().to(res_sign.dtype), **_tols(dtype, n)
    )
    torch.testing.assert_close(
        res_logabsdet.cpu(),
        ref.logabsdet.cpu().to(res_logabsdet.dtype),
        **_tols(dtype, n),
    )


@pytest.mark.slogdet
@pytest.mark.parametrize("shape", [(3, 3), (8, 8), (2, 4, 4)])
@pytest.mark.parametrize("dtype", COMPLEX_DTYPES)
def test_slogdet_complex(shape, dtype):
    """Complex input: sign is a unit-modulus complex number, logabsdet is real."""
    A = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_A = utils.to_reference(A)

    ref = torch.linalg.slogdet(ref_A)
    res_sign, res_logabsdet = flag_gems.slogdet(A)

    n = shape[-1]
    assert res_sign.dtype == dtype
    assert res_logabsdet.dtype == (
        torch.float32 if dtype == torch.complex64 else torch.float64
    )
    torch.testing.assert_close(res_sign.cpu(), ref.sign.cpu(), **_tols(dtype, n))
    torch.testing.assert_close(
        res_logabsdet.cpu(), ref.logabsdet.cpu(), **_tols(dtype, n)
    )
    # sign * exp(logabsdet) reconstructs the determinant.
    torch.testing.assert_close(
        (res_sign.cpu() * torch.exp(res_logabsdet.cpu()).to(dtype)),
        (ref.sign.cpu() * torch.exp(ref.logabsdet.cpu()).to(dtype)),
        **_tols(dtype, n),
    )


@pytest.mark.slogdet
@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_slogdet_near_singular_scaled(dtype):
    """A tiny-but-nonzero pivot is not singular.

    diag([1e-11, 1]) has determinant 1e-11: ATen returns a finite logabsdet, so
    classifying it as singular with a fixed 1e-10 threshold is wrong.
    """
    for small in (1e-8, 1e-11, 1e-12, 1e-20):
        vals = torch.tensor([small, 1.0], dtype=dtype, device=flag_gems.device)
        A = torch.diag(vals)
        ref = torch.linalg.slogdet(utils.to_reference(A))
        res_sign, res_logabsdet = flag_gems.slogdet(A)

        assert torch.isfinite(
            res_logabsdet
        ), f"diag([{small}, 1]) is non-singular, got logabsdet={res_logabsdet.item()}"
        assert res_sign.item() == pytest.approx(1.0)
        torch.testing.assert_close(
            res_logabsdet.cpu(),
            ref.logabsdet.cpu().to(res_logabsdet.dtype),
            **_tols(dtype, 2),
        )


@pytest.mark.slogdet
@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("scale", [1e-3, 1e-2, 1e2, 1e3])
def test_slogdet_scaled_identity(dtype, scale):
    """Uniformly scaled matrices stay accurate away from unit magnitude."""
    n = 4
    A = torch.eye(n, dtype=dtype, device=flag_gems.device) * scale
    ref = torch.linalg.slogdet(utils.to_reference(A))
    res_sign, res_logabsdet = flag_gems.slogdet(A)

    torch.testing.assert_close(
        res_logabsdet.cpu(),
        ref.logabsdet.cpu().to(res_logabsdet.dtype),
        rtol=1e-4,
        atol=1e-4,
    )
    assert res_sign.item() == pytest.approx(1.0)


@pytest.mark.slogdet
@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_slogdet_singular(dtype):
    """Exactly singular matrices give sign 0 and logabsdet -inf."""
    cases = {
        "zeros": torch.zeros((4, 4), dtype=dtype, device=flag_gems.device),
        "zero_diag_entry": torch.diag(
            torch.tensor([0.0, 1.0, 2.0], dtype=dtype, device=flag_gems.device)
        ),
        "rank_deficient": torch.tensor(
            [[1.0, 2.0], [2.0, 4.0]], dtype=dtype, device=flag_gems.device
        ),
        "zero_row": torch.tensor(
            [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [4.0, 5.0, 7.0]],
            dtype=dtype,
            device=flag_gems.device,
        ),
        "zero_column": torch.tensor(
            [[1.0, 0.0, 3.0], [2.0, 0.0, 1.0], [4.0, 0.0, 7.0]],
            dtype=dtype,
            device=flag_gems.device,
        ),
    }
    # These are exactly singular in floating point: elimination leaves a hard
    # zero pivot rather than a rounding-level residue. (A matrix like
    # [[1,2,3],[1,2,3],[4,5,7]] is mathematically singular but ATen still
    # reports a finite logabsdet for it, so it is not usable here.)
    for name, A in cases.items():
        ref = torch.linalg.slogdet(utils.to_reference(A))
        res_sign, res_logabsdet = flag_gems.slogdet(A)
        assert res_sign.item() == 0.0, f"{name}: expected sign 0"
        assert res_logabsdet.item() == float("-inf"), f"{name}: expected -inf"
        assert ref.logabsdet.item() == float("-inf"), f"{name}: reference disagrees"


@pytest.mark.slogdet
@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_slogdet_1x1(dtype):
    """1x1 matrices: sign(a), log|a|."""
    for val in (2.5, -2.5, 1.0):
        A = torch.tensor([[val]], dtype=dtype, device=flag_gems.device)
        ref = torch.linalg.slogdet(utils.to_reference(A))
        res_sign, res_logabsdet = flag_gems.slogdet(A)
        torch.testing.assert_close(
            res_sign.cpu(), ref.sign.cpu().to(res_sign.dtype), rtol=1e-6, atol=1e-6
        )
        torch.testing.assert_close(
            res_logabsdet.cpu(),
            ref.logabsdet.cpu().to(res_logabsdet.dtype),
            rtol=1e-6,
            atol=1e-6,
        )


@pytest.mark.slogdet
@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_slogdet_empty_matrix(dtype):
    """A 0x0 matrix is an empty product: sign 1, logabsdet 0."""
    for shape in [(0, 0), (2, 0, 0)]:
        A = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        ref = torch.linalg.slogdet(utils.to_reference(A))
        res_sign, res_logabsdet = flag_gems.slogdet(A)
        assert res_sign.shape == ref.sign.shape
        assert torch.all(res_sign.cpu() == 1)
        assert torch.all(res_logabsdet.cpu() == 0)


@pytest.mark.slogdet
@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_slogdet_empty_batch(dtype):
    """An empty batch returns empty outputs with the right shape."""
    for shape in [(0, 3, 3), (0, 2, 4, 4)]:
        A = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        ref = torch.linalg.slogdet(utils.to_reference(A))
        res_sign, res_logabsdet = flag_gems.slogdet(A)
        assert res_sign.shape == ref.sign.shape
        assert res_logabsdet.shape == ref.logabsdet.shape
        assert res_sign.numel() == 0


@pytest.mark.slogdet
@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("batch_shape", [(2,), (2, 3), (2, 3, 2)])
def test_slogdet_multi_dim_batch(dtype, batch_shape):
    """Multi-dimensional batches reduce over the trailing two dims only."""
    n = 4
    A = torch.randn(*batch_shape, n, n, dtype=dtype, device=flag_gems.device)
    ref = torch.linalg.slogdet(utils.to_reference(A))
    res_sign, res_logabsdet = flag_gems.slogdet(A)

    assert res_sign.shape == batch_shape
    assert res_logabsdet.shape == batch_shape
    torch.testing.assert_close(
        res_sign.cpu(), ref.sign.cpu().to(res_sign.dtype), **_tols(dtype, n)
    )
    torch.testing.assert_close(
        res_logabsdet.cpu(),
        ref.logabsdet.cpu().to(res_logabsdet.dtype),
        **_tols(dtype, n),
    )


@pytest.mark.slogdet
@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_slogdet_non_contiguous(dtype):
    """Transposed and strided views must match, and must not be mutated."""
    n = 6
    base = torch.randn(n, n, dtype=dtype, device=flag_gems.device)

    transposed = base.t()
    before = transposed.clone()
    ref = torch.linalg.slogdet(utils.to_reference(transposed))
    res_sign, res_logabsdet = flag_gems.slogdet(transposed)
    assert not transposed.is_contiguous()
    torch.testing.assert_close(
        res_logabsdet.cpu(),
        ref.logabsdet.cpu().to(res_logabsdet.dtype),
        **_tols(dtype, n),
    )
    assert torch.equal(transposed, before), "input was mutated"

    wide = torch.randn(4, 8, dtype=dtype, device=flag_gems.device)
    strided = wide[:, ::2]
    assert not strided.is_contiguous()
    ref2 = torch.linalg.slogdet(utils.to_reference(strided))
    _, res2 = flag_gems.slogdet(strided)
    torch.testing.assert_close(
        res2.cpu(), ref2.logabsdet.cpu().to(res2.dtype), **_tols(dtype, 4)
    )


@pytest.mark.slogdet
@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_slogdet_nan_inf(dtype):
    """Non-finite entries must not crash and must not fabricate a finite result.

    Inf input is checked against ATen directly. NaN input is only checked for
    non-finiteness: ATen itself is not self-consistent there (for
    [[nan, 0], [0, 1]] the CPU path returns logabsdet=nan while the CUDA path
    returns -inf), so there is no single reference value to match.
    """
    inf_cases = {
        "inf": ([[float("inf"), 0.0], [0.0, 1.0]], 1.0),
        "neg_inf": ([[float("-inf"), 0.0], [0.0, 1.0]], -1.0),
    }
    for name, (vals, want_sign) in inf_cases.items():
        A = torch.tensor(vals, dtype=dtype, device=flag_gems.device)
        res_sign, res_logabsdet = flag_gems.slogdet(A)
        assert res_logabsdet.item() == float(
            "inf"
        ), f"{name}: expected +inf, got {res_logabsdet.item()}"
        assert res_sign.item() == pytest.approx(want_sign)

    nan_cases = {
        "nan_diag": [[float("nan"), 0.0], [0.0, 1.0]],
        "nan_offdiag": [[1.0, float("nan")], [0.0, 1.0]],
    }
    for name, vals in nan_cases.items():
        A = torch.tensor(vals, dtype=dtype, device=flag_gems.device)
        res_sign, res_logabsdet = flag_gems.slogdet(A)
        la = res_logabsdet.item()
        assert not (
            la == la and abs(la) != float("inf")
        ), f"{name}: a NaN input must not produce a finite logabsdet, got {la}"


@pytest.mark.slogdet
@pytest.mark.parametrize("n", [65, 128, 257])
def test_slogdet_large(n):
    """Matrices above the register-tile boundary use the tiled path.

    Rounding both dimensions up to a power of two would make n=1025 build a
    2048x2048 block and exceed Triton's maximum tensor size.
    """
    dtype = torch.float64
    torch.manual_seed(0)
    # A diagonally dominant matrix keeps the LU well conditioned so the
    # comparison tests the tiling rather than float noise.
    A = torch.randn(n, n, dtype=dtype, device=flag_gems.device) + n * torch.eye(
        n, dtype=dtype, device=flag_gems.device
    )
    ref = torch.linalg.slogdet(utils.to_reference(A))
    res_sign, res_logabsdet = flag_gems.slogdet(A)

    assert res_sign.item() == pytest.approx(ref.sign.item())
    torch.testing.assert_close(
        res_logabsdet.cpu(), ref.logabsdet.cpu(), rtol=1e-8, atol=1e-8
    )


@pytest.mark.slogdet
def test_slogdet_large_1025():
    """The exact size called out in review: 1025x1025."""
    n = 1025
    dtype = torch.float64
    torch.manual_seed(0)
    A = torch.randn(n, n, dtype=dtype, device=flag_gems.device) + n * torch.eye(
        n, dtype=dtype, device=flag_gems.device
    )
    ref = torch.linalg.slogdet(utils.to_reference(A))
    res_sign, res_logabsdet = flag_gems.slogdet(A)
    assert res_sign.item() == pytest.approx(ref.sign.item())
    torch.testing.assert_close(
        res_logabsdet.cpu(), ref.logabsdet.cpu(), rtol=1e-7, atol=1e-7
    )


@pytest.mark.slogdet
def test_slogdet_invalid_inputs():
    """Invalid inputs raise, matching the ATen conditions."""
    dev = flag_gems.device
    with pytest.raises(RuntimeError, match="at least 2 dimensions"):
        flag_gems.slogdet(torch.randn(5, dtype=torch.float32, device=dev))
    with pytest.raises(RuntimeError, match="at least 2 dimensions"):
        flag_gems.slogdet(torch.randn((), dtype=torch.float32, device=dev))
    with pytest.raises(RuntimeError, match="square"):
        flag_gems.slogdet(torch.randn(3, 4, dtype=torch.float32, device=dev))
    with pytest.raises(RuntimeError, match="Low precision"):
        flag_gems.slogdet(torch.randn(3, 3, dtype=torch.float16, device=dev))
    with pytest.raises(RuntimeError, match="Low precision"):
        flag_gems.slogdet(torch.randn(3, 3, dtype=torch.bfloat16, device=dev))
    with pytest.raises(RuntimeError, match="floating point or complex"):
        flag_gems.slogdet(torch.ones(3, 3, dtype=torch.int32, device=dev))


@pytest.mark.slogdet
@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_slogdet_backward(dtype):
    """d logabsdet / dA == inv(A)^T, matching ATen."""
    n = 5
    torch.manual_seed(0)
    A = torch.randn(n, n, dtype=dtype, device=flag_gems.device) + n * torch.eye(
        n, dtype=dtype, device=flag_gems.device
    )

    ref_A = utils.to_reference(A).clone().requires_grad_(True)
    ref = torch.linalg.slogdet(ref_A)
    ref.logabsdet.backward()

    gems_A = A.clone().requires_grad_(True)
    _, logabsdet = flag_gems.slogdet(gems_A)
    logabsdet.backward()

    torch.testing.assert_close(
        gems_A.grad.cpu(), ref_A.grad.cpu().to(gems_A.grad.dtype), **_tols(dtype, n)
    )


@pytest.mark.slogdet
@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_slogdet_batched_backward(dtype):
    """Batched backward parity."""
    n = 4
    torch.manual_seed(0)
    A = torch.randn(3, n, n, dtype=dtype, device=flag_gems.device) + n * torch.eye(
        n, dtype=dtype, device=flag_gems.device
    )

    ref_A = utils.to_reference(A).clone().requires_grad_(True)
    ref = torch.linalg.slogdet(ref_A)
    ref.logabsdet.sum().backward()

    gems_A = A.clone().requires_grad_(True)
    _, logabsdet = flag_gems.slogdet(gems_A)
    logabsdet.sum().backward()

    torch.testing.assert_close(
        gems_A.grad.cpu(), ref_A.grad.cpu().to(gems_A.grad.dtype), **_tols(dtype, n)
    )


@pytest.mark.slogdet
@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_slogdet_identity(dtype):
    """The identity has sign 1 and logabsdet 0."""
    for n in (1, 3, 8, 65):
        A = torch.eye(n, dtype=dtype, device=flag_gems.device)
        res_sign, res_logabsdet = flag_gems.slogdet(A)
        assert res_sign.item() == pytest.approx(1.0)
        assert res_logabsdet.item() == pytest.approx(0.0, abs=1e-6)
