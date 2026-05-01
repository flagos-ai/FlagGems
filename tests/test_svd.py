import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    DTYPES = [torch.float32]
    SHAPES = [(8, 6)]
    SOME_OPTS = [True]
else:
    DTYPES = [torch.float32, torch.float64, torch.complex64]
    SHAPES = [(8, 6), (2, 5, 7)]
    SOME_OPTS = [True, False]


@pytest.mark.svd
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("some", SOME_OPTS)
def test_svd_matches_cpu(shape, dtype, some):
    if flag_gems.vendor_name == "metax":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref = inp.cpu()

    ref_u, ref_s, ref_v = torch.svd(ref, some=some, compute_uv=True)
    with flag_gems.use_gems():
        u, s, v = torch.svd(inp, some=some, compute_uv=True)

    utils.gems_assert_close(u, ref_u.to(inp.device), dtype, equal_nan=True)
    utils.gems_assert_close(s, ref_s.to(inp.device), dtype, equal_nan=True)
    utils.gems_assert_close(v, ref_v.to(inp.device), dtype, equal_nan=True)


@pytest.mark.svd
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_svd_reconstruction(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        u, s, v = torch.svd(inp, some=True, compute_uv=True)

    k = min(inp.shape[-2], inp.shape[-1])
    ds = torch.diag_embed(s)
    recon = torch.matmul(torch.matmul(u[..., :k], ds), v[..., :k].transpose(-2, -1))

    utils.gems_assert_close(recon, inp, dtype, equal_nan=True)


@pytest.mark.svd
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_linalg_svd_matches_cpu(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref = inp.cpu()

    ref_u, ref_s, ref_vh = torch.linalg.svd(ref, full_matrices=True)
    with flag_gems.use_gems():
        u, s, vh = torch.linalg.svd(inp, full_matrices=True)

    utils.gems_assert_close(u, ref_u.to(inp.device), dtype, equal_nan=True)
    utils.gems_assert_close(s, ref_s.to(inp.device), dtype, equal_nan=True)
    utils.gems_assert_close(vh, ref_vh.to(inp.device), dtype, equal_nan=True)


@pytest.mark.svd
def test_svd_compute_uv_false():
    inp = torch.randn(4, 5, dtype=torch.float32, device=flag_gems.device)
    ref_u, ref_s, ref_v = torch.svd(inp.cpu(), compute_uv=False)
    with flag_gems.use_gems():
        u, s, v = torch.svd(inp, compute_uv=False)

    utils.gems_assert_close(u, ref_u.to(inp.device), torch.float32, equal_nan=True)
    utils.gems_assert_close(s, ref_s.to(inp.device), torch.float32, equal_nan=True)
    utils.gems_assert_close(v, ref_v.to(inp.device), torch.float32, equal_nan=True)
