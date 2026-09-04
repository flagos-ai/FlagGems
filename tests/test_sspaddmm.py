import pytest
import torch

import flag_gems
from flag_gems.ops.sspaddmm import sspaddmm, sspaddmm_out

from . import accuracy_utils as utils

# torch's sspaddmm has no CUDA kernel, so the reference is always computed on
# CPU (in fp64 for accuracy) and compared against the FlagGems dense result.

# (M, K, N) shapes hand-picked to cover small/medium matmul sizes plus
# degenerate empty-dimension cases (0 rows / 0 cols); sspaddmm has no shared
# shape constant, so these are enumerated inline.
MNK_SHAPES = [
    (4, 3, 5),
    (1, 1, 32),
    (16, 8, 20),
    (64, 32, 50),
    (128, 64, 128),
    (0, 3, 5),
    (4, 3, 0),
]

SPARSITIES = [0.3, 0.7]
ALPHA_BETA = [(1.0, 1.0), (2.0, 3.0), (0.0, 1.0), (3.0, 0.0)]

DTYPES = utils.FLOAT_DTYPES


def _make_sparse_coo(m, n, dtype, sparsity):
    dense = torch.randn((m, n), dtype=dtype)
    if m > 0 and n > 0:
        mask = torch.rand((m, n)) > sparsity
        dense = dense * mask
    return dense


def _ref_sspaddmm(inp_d, mat1_d, mat2_d, beta, alpha):
    # Golden reference computed on CPU in fp64, since sspaddmm has no CUDA
    # kernel. This is the self-computed reference for the accuracy comparison.
    ref_out = torch.sspaddmm(
        inp_d.double().to_sparse(),
        mat1_d.double().to_sparse(),
        mat2_d.double(),
        beta=beta,
        alpha=alpha,
    )
    return ref_out.to_dense()


@pytest.mark.sspaddmm
@pytest.mark.parametrize("mnk", MNK_SHAPES)
@pytest.mark.parametrize("sparsity", SPARSITIES)
@pytest.mark.parametrize("alpha_beta", ALPHA_BETA)
@pytest.mark.parametrize("dtype", DTYPES)
def test_sspaddmm(mnk, sparsity, alpha_beta, dtype):
    m, k, n = mnk
    alpha, beta = alpha_beta
    device = flag_gems.device

    inp_d = _make_sparse_coo(m, n, dtype, sparsity)
    mat1_d = _make_sparse_coo(m, k, dtype, sparsity)
    mat2_d = torch.randn((k, n), dtype=dtype)

    ref_dense = _ref_sspaddmm(inp_d, mat1_d, mat2_d, beta, alpha)

    inp = inp_d.to(device).to_sparse()
    mat1 = mat1_d.to(device).to_sparse()
    mat2 = mat2_d.to(device)

    res = sspaddmm(inp, mat1, mat2, beta=beta, alpha=alpha)

    assert res.layout == torch.sparse_coo
    res_dense = res.to_dense().to("cpu")

    utils.gems_assert_close(
        res_dense, ref_dense.to(res_dense.dtype), dtype, reduce_dim=k
    )


@pytest.mark.sspaddmm_out
@pytest.mark.parametrize("mnk", [(16, 8, 20), (64, 32, 50)])
@pytest.mark.parametrize("alpha_beta", [(1.0, 1.0), (2.0, 3.0)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_sspaddmm_out(mnk, alpha_beta, dtype):
    m, k, n = mnk
    alpha, beta = alpha_beta
    device = flag_gems.device

    inp_d = _make_sparse_coo(m, n, dtype, 0.5)
    mat1_d = _make_sparse_coo(m, k, dtype, 0.5)
    mat2_d = torch.randn((k, n), dtype=dtype)

    ref_dense = _ref_sspaddmm(inp_d, mat1_d, mat2_d, beta, alpha)

    inp = inp_d.to(device).to_sparse()
    mat1 = mat1_d.to(device).to_sparse()
    mat2 = mat2_d.to(device)
    out = torch.empty((m, n), dtype=dtype, device=device).to_sparse()

    res = sspaddmm_out(inp, mat1, mat2, beta=beta, alpha=alpha, out=out)

    assert res.layout == torch.sparse_coo
    res_dense = res.to_dense().to("cpu")

    utils.gems_assert_close(
        res_dense, ref_dense.to(res_dense.dtype), dtype, reduce_dim=k
    )
