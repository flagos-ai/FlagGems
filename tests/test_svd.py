import pytest
import torch

import flag_gems

from .accuracy_utils import gems_assert_close, to_reference

SVD_SHAPES = [
    (1, 1),
    (3, 3),
    (8, 8),
    (16, 16),
    (5, 3),
    (3, 5),
    (16, 8),
    (8, 16),
    (7, 5),
    (7, 3),
    (13, 11),
    (2, 3, 3),
    (4, 5, 3),
    (2, 3, 8, 4),
    (1, 5, 3),
    (10, 8, 8),
    (100, 3, 3),
    (32, 32),
    (64, 32),
    (32, 64),
    (48, 16),
]

SVD_SQUARE_SHAPES = [
    (1, 1),
    (3, 3),
    (8, 8),
    (16, 16),
    (2, 4, 4),
    (3, 8, 8),
    (32, 32),
]

SVD_DTYPES = [torch.float32]
if flag_gems.runtime.device.support_fp64:
    SVD_DTYPES.append(torch.float64)


@pytest.mark.svd
@pytest.mark.parametrize("shape", SVD_SHAPES)
@pytest.mark.parametrize("dtype", SVD_DTYPES)
@pytest.mark.parametrize("compute_uv", [True, False])
def test_accuracy_svd_reduced(shape, dtype, compute_uv):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, upcast=True)

    ref_result = torch.svd(ref_inp, some=True, compute_uv=compute_uv)
    with flag_gems.use_gems():
        res_result = torch.svd(inp, some=True, compute_uv=compute_uv)

    gems_assert_close(res_result.S, ref_result.S, dtype, reduce_dim=shape[-1])

    if compute_uv:
        reconstructed = torch.matmul(
            res_result.U * res_result.S.unsqueeze(-2),
            res_result.V.transpose(-2, -1),
        )
        ref_inp_device = to_reference(inp)
        gems_assert_close(reconstructed, ref_inp_device, dtype, reduce_dim=shape[-1])


@pytest.mark.svd
@pytest.mark.parametrize("shape", SVD_SQUARE_SHAPES)
@pytest.mark.parametrize("dtype", SVD_DTYPES)
@pytest.mark.parametrize("compute_uv", [True, False])
def test_accuracy_svd_full(shape, dtype, compute_uv):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, upcast=True)

    ref_result = torch.svd(ref_inp, some=False, compute_uv=compute_uv)
    with flag_gems.use_gems():
        res_result = torch.svd(inp, some=False, compute_uv=compute_uv)

    gems_assert_close(res_result.S, ref_result.S, dtype, reduce_dim=shape[-1])

    if compute_uv:
        k = min(shape[-2], shape[-1])
        reconstructed = torch.matmul(
            res_result.U[..., :k] * res_result.S.unsqueeze(-2),
            res_result.V[..., :k].transpose(-2, -1),
        )
        ref_inp_device = to_reference(inp)
        gems_assert_close(reconstructed, ref_inp_device, dtype, reduce_dim=shape[-1])


@pytest.mark.svd
@pytest.mark.parametrize("shape", SVD_SHAPES)
@pytest.mark.parametrize("dtype", SVD_DTYPES)
def test_accuracy_svd_descending_order(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        res_result = torch.svd(inp, some=True, compute_uv=True)
    S = res_result.S
    if S.shape[-1] > 1:
        diff = S[..., :-1] - S[..., 1:]
        assert (diff >= -1e-5).all(), f"Singular values not in descending order: {S}"


@pytest.mark.svd
@pytest.mark.parametrize(
    "shape",
    [
        (8, 5),
        (5, 8),
        (3, 16, 8),
        (2, 32, 16),
    ],
)
def test_accuracy_svd_non_contiguous(shape):
    t_shape = list(shape)
    t_shape[-2], t_shape[-1] = t_shape[-1], t_shape[-2]
    inp = torch.randn(t_shape, dtype=torch.float32, device=flag_gems.device)
    inp_t = inp.transpose(-2, -1)
    ref_inp = to_reference(inp_t, upcast=True)

    ref_result = torch.svd(ref_inp, some=True, compute_uv=True)
    with flag_gems.use_gems():
        res_result = torch.svd(inp_t, some=True, compute_uv=True)

    gems_assert_close(res_result.S, ref_result.S, torch.float32, reduce_dim=shape[-1])


@pytest.mark.svd
@pytest.mark.parametrize(
    "shape",
    [(4, 3), (8, 8), (3, 5), (2, 4, 4)],
)
def test_accuracy_svd_zero_matrix(shape):
    inp = torch.zeros(shape, dtype=torch.float32, device=flag_gems.device)
    with flag_gems.use_gems():
        res_result = torch.svd(inp, some=True, compute_uv=True)

    assert (
        res_result.S == 0
    ).all(), f"Expected all zero singular values, got {res_result.S}"


@pytest.mark.svd
@pytest.mark.parametrize("n", [3, 8, 16])
def test_accuracy_svd_identity_matrix(n):
    inp = torch.eye(n, dtype=torch.float32, device=flag_gems.device)
    with flag_gems.use_gems():
        res_result = torch.svd(inp, some=True, compute_uv=True)

    expected_S = to_reference(
        torch.ones(n, dtype=torch.float32, device=flag_gems.device)
    )
    gems_assert_close(res_result.S, expected_S, torch.float32)


@pytest.mark.svd
@pytest.mark.parametrize("n", [3, 8, 16])
def test_accuracy_svd_diagonal_matrix(n):
    diag_vals = torch.arange(1, n + 1, dtype=torch.float32, device=flag_gems.device)
    inp = torch.diag(diag_vals)
    with flag_gems.use_gems():
        res_result = torch.svd(inp, some=True, compute_uv=True)

    expected_S = to_reference(diag_vals.flip(0))
    gems_assert_close(res_result.S, expected_S, torch.float32)


@pytest.mark.svd
@pytest.mark.parametrize("shape", [(8, 8), (16, 8), (8, 16)])
def test_accuracy_svd_rank_deficient(shape):
    m, n = shape
    k = min(m, n)
    rank = max(1, k // 2)
    U_r = torch.randn(m, rank, dtype=torch.float32, device=flag_gems.device)
    V_r = torch.randn(n, rank, dtype=torch.float32, device=flag_gems.device)
    inp = U_r @ V_r.t()
    ref_inp = to_reference(inp, upcast=True)

    ref_result = torch.svd(ref_inp, some=True, compute_uv=True)
    with flag_gems.use_gems():
        res_result = torch.svd(inp, some=True, compute_uv=True)

    gems_assert_close(res_result.S, ref_result.S, torch.float32, reduce_dim=n)


@pytest.mark.svd
@pytest.mark.parametrize(
    "shape",
    [(5, 3), (3, 5), (2, 8, 4), (32, 16), (16, 32)],
)
def test_accuracy_svd_full_non_square(shape):
    inp = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    ref_inp = to_reference(inp, upcast=True)

    ref_result = torch.svd(ref_inp, some=False, compute_uv=True)
    with flag_gems.use_gems():
        res_result = torch.svd(inp, some=False, compute_uv=True)

    gems_assert_close(res_result.S, ref_result.S, torch.float32, reduce_dim=shape[-1])
    k = min(shape[-2], shape[-1])
    reconstructed = torch.matmul(
        res_result.U[..., :k] * res_result.S.unsqueeze(-2),
        res_result.V[..., :k].transpose(-2, -1),
    )
    gems_assert_close(
        reconstructed, to_reference(inp), torch.float32, reduce_dim=shape[-1]
    )
