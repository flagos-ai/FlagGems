import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.linalg_eig
@pytest.mark.parametrize("shape", [(2, 2), (3, 3), (5, 5), (8, 8), (16, 16)])
# linalg_eig computes a general (non-symmetric) eigendecomposition; the Triton
# kernel runs the Francis double-shift QR in float32 -> complex64 output.
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_eig(shape, dtype):
    """linalg.eig accuracy vs. PyTorch reference.

    Eigenvectors are not unique, so we validate the two invariants that are
    well defined: the (unordered) eigenvalue set and the reconstruction
    A @ V == V @ diag(eigenvalues).
    """
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_w, _ = torch.linalg.eig(ref_inp)
    with flag_gems.use_gems():
        res_w, res_v = torch.linalg.eig(inp)

    # eigenvalue set: sort by (real, imag) before comparing
    def _sort_key(w):
        r = torch.view_as_real(w)
        return torch.argsort(r[:, 0] * 1e6 + r[:, 1])

    res_sorted = res_w[_sort_key(res_w)]
    ref_sorted = ref_w[_sort_key(ref_w)]
    utils.gems_assert_close(res_sorted, ref_sorted, res_w.dtype, atol=1e-3)

    # reconstruction residual A @ V - V @ diag(w)
    a_c = inp.to(res_v.dtype)
    recon = a_c @ res_v - res_v @ torch.diag(res_w)
    assert torch.linalg.norm(recon).item() < 1e-2 * (shape[0] ** 0.5 + 1.0)


@pytest.mark.linalg_eig
@pytest.mark.parametrize("batch", [2, 4])
@pytest.mark.parametrize("n", [4, 6])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_eig_batched(batch, n, dtype):
    inp = torch.randn(batch, n, n, dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        res_w, res_v = torch.linalg.eig(inp)

    a_c = inp.to(res_v.dtype)
    recon = a_c @ res_v - res_v @ torch.diag_embed(res_w)
    assert torch.linalg.norm(recon).item() < 1e-2 * (batch * n) ** 0.5
