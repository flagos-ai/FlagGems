import random
import time

import pytest
import torch

import flag_gems

from .accuracy_utils import gems_assert_close, to_reference

random.seed(time.time() // 100)

device = flag_gems.device

SVD_SHAPES = [
    (1, 1),
    (8, 8),
    (32, 32),
    (5, 3),
    (3, 5),
    (16, 8),
    (32, 64),
    (13, 11),
    (2, 3, 3),
    (2, 3, 8, 4),
    (100, 3, 3),
]

SVD_DTYPES = [torch.float32]
if flag_gems.runtime.device.support_fp64:
    SVD_DTYPES.append(torch.float64)


@pytest.mark.svd
@pytest.mark.parametrize("shape", SVD_SHAPES)
@pytest.mark.parametrize("dtype", SVD_DTYPES)
@pytest.mark.parametrize("some", [True, False])
@pytest.mark.parametrize("compute_uv", [True, False])
def test_svd_accuracy(shape, dtype, some, compute_uv):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, upcast=True)

    ref_result = torch.svd(ref_inp, some=some, compute_uv=compute_uv)

    with flag_gems.use_gems():
        res = torch.svd(inp, some=some, compute_uv=compute_uv)

    gems_assert_close(res.S, ref_result.S, dtype, reduce_dim=shape[-1])

    if res.S.shape[-1] > 1:
        diff = res.S[..., :-1] - res.S[..., 1:]
        assert (diff >= -1e-5).all(), "Singular values are not in descending order!"

    if compute_uv:
        k = min(shape[-2], shape[-1])
        reconstructed = torch.matmul(
            res.U[..., :k] * res.S.unsqueeze(-2),
            res.V[..., :k].transpose(-2, -1),
        )
        gems_assert_close(reconstructed, to_reference(inp), dtype, reduce_dim=shape[-1])


@pytest.mark.svd
@pytest.mark.parametrize("n", [3, 8, 16])
@pytest.mark.parametrize(
    "matrix_type", ["zero", "identity", "diagonal", "rank_deficient"]
)
def test_svd_special_matrices(n, matrix_type):
    device = flag_gems.device
    dtype = torch.float32

    if matrix_type == "zero":
        inp = torch.zeros((n, n), dtype=dtype, device=device)
        expected_S = torch.zeros(n, dtype=dtype, device=device)
    elif matrix_type == "identity":
        inp = torch.eye(n, dtype=dtype, device=device)
        expected_S = torch.ones(n, dtype=dtype, device=device)
    elif matrix_type == "diagonal":
        diag_vals = torch.arange(1, n + 1, dtype=dtype, device=device)
        inp = torch.diag(diag_vals)
        expected_S = diag_vals.flip(0)
    elif matrix_type == "rank_deficient":
        rank = max(1, n // 2)
        U_r = torch.randn(n, rank, dtype=dtype, device=device)
        V_r = torch.randn(n, rank, dtype=dtype, device=device)
        inp = U_r @ V_r.t()
        expected_S = torch.svd(inp)[1]

    with flag_gems.use_gems():
        res = torch.svd(inp, some=True, compute_uv=True)

    gems_assert_close(res.S, to_reference(expected_S), dtype)


@pytest.mark.svd
@pytest.mark.parametrize("shape", [(8, 5), (3, 16, 8), (2, 32, 16)])
def test_svd_non_contiguous(shape):
    inp = torch.randn(shape, dtype=torch.float32, device=flag_gems.device).transpose(
        -2, -1
    )
    ref_inp = to_reference(inp, upcast=True)

    ref_result = torch.svd(ref_inp)
    with flag_gems.use_gems():
        res = torch.svd(inp)
    gems_assert_close(res.S, ref_result.S, torch.float32, reduce_dim=shape[-1])
