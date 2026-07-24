import pytest
import torch

import flag_gems
from flag_gems.ops.sparse_sampled_addmm import _broadcast_sparse_csr

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

if QUICK_MODE:
    PYTORCH_MNK = [(5, 5, 5)]
    LARGE_MNK = []
    BATCH_SHAPES = [()]
    SPARSITIES = [0.5]
    ALPHA_BETA = [(1.0, 1.0)]
    INDEX_DTYPES = [torch.int64]
    NONCONTIGUOUS = [False]
else:
    PYTORCH_MNK = [
        (5, 5, 5),
        (5, 5, 0),
        (0, 0, 0),
        (0, 5, 5),
    ]

    LARGE_MNK = [
        (1, 1, 32),
        (15, 160, 1024),
        (495, 5333, 71),
        (2048, 2048, 2048),
    ]
    BATCH_SHAPES = [(), (2,), (2, 3)]
    SPARSITIES = [0.0, 0.5, 1.0]
    ALPHA_BETA = [(1.0, 1.0), (2.0, 0.0)]
    INDEX_DTYPES = [torch.int32, torch.int64]
    NONCONTIGUOUS = [False, True]


BCAST_AND_BATCH = [(False, bs) for bs in BATCH_SHAPES] + [
    (True, bs) for bs in BATCH_SHAPES if bs
]

DTYPES = utils.ALL_FLOAT_DTYPES
_LOW_PREC_DTYPES = (torch.float16, torch.bfloat16)


def _assert_close_dense(res_dense, ref_dense, dtype, reduce_dim, atol):
    utils.gems_assert_close(
        res_dense, ref_dense, dtype, reduce_dim=reduce_dim, atol=atol
    )


def _make_sparse_csr(shape, dtype, device, sparsity=0.5, index_dtype=torch.int64):
    if not shape:
        raise ValueError("shape must be non-empty")
    M, N = shape[-2:]
    dense = torch.randn(shape, dtype=dtype, device=device)
    dense = torch.where(dense == 0, 1.0, dense)

    if sparsity == 0.0:
        mask = torch.ones((M, N), dtype=torch.bool, device=device)
    elif sparsity == 1.0:
        mask = torch.zeros((M, N), dtype=torch.bool, device=device)
    else:
        mask = torch.rand((M, N), device=device) > sparsity
    mask = mask.expand(shape)
    dense = dense * mask

    if torch.device(device).type == "cuda":
        csr = dense.to_sparse_csr()
    else:
        csr_cpu = dense.cpu().to_sparse_csr()
        csr = torch.sparse_csr_tensor(
            csr_cpu.crow_indices().to(device),
            csr_cpu.col_indices().to(device),
            csr_cpu.values().to(device),
            size=csr_cpu.shape,
            dtype=csr_cpu.dtype,
            device=device,
        )
    if csr.crow_indices().dtype != index_dtype:
        csr = torch.sparse_csr_tensor(
            csr.crow_indices().to(index_dtype),
            csr.col_indices().to(index_dtype),
            csr.values(),
            size=csr.shape,
            dtype=csr.dtype,
            device=csr.device,
        )
    return csr


def _csr_to_cpu(csr):
    return torch.sparse_csr_tensor(
        csr.crow_indices().cpu(),
        csr.col_indices().cpu(),
        csr.values().cpu(),
        size=csr.shape,
        dtype=csr.dtype,
    )


def _to_cpu_ref(csr):
    ref_dtype = torch.float64 if utils.fp64_is_supported else torch.float32
    return torch.sparse_csr_tensor(
        csr.crow_indices().cpu(),
        csr.col_indices().cpu(),
        csr.values().cpu().to(ref_dtype),
        size=csr.shape,
    )


@pytest.mark.sparse_sampled_addmm
@pytest.mark.parametrize("M, N, K", PYTORCH_MNK)
@pytest.mark.parametrize("sparsity", SPARSITIES)
@pytest.mark.parametrize("alpha, beta", ALPHA_BETA)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("bcast_input, batch_shape", BCAST_AND_BATCH)
@pytest.mark.parametrize("noncontiguous", NONCONTIGUOUS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_sparse_sampled_addmm(
    M,
    N,
    K,
    batch_shape,
    sparsity,
    alpha,
    beta,
    dtype,
    bcast_input,
    noncontiguous,
    index_dtype,
):
    device = flag_gems.device

    mat1_shape = batch_shape + (M, K)
    mat2_shape = batch_shape + (K, N)
    input_shape = (M, N) if bcast_input else batch_shape + (M, N)

    mat1 = torch.randn(mat1_shape, dtype=dtype, device=device)
    mat2 = torch.randn(mat2_shape, dtype=dtype, device=device)
    if noncontiguous:
        # Transpose and transpose back to get non-contiguous strides.
        mat1 = mat1.transpose(-2, -1).transpose(-2, -1)
        mat2 = mat2.transpose(-2, -1).transpose(-2, -1)
    input = _make_sparse_csr(input_shape, dtype, device, sparsity, index_dtype)

    ref_input = _to_cpu_ref(input)
    ref_mat1 = utils.to_reference(mat1, upcast=True).cpu()
    ref_mat2 = utils.to_reference(mat2, upcast=True).cpu()

    ref_out = torch.sparse.sampled_addmm(
        ref_input, ref_mat1, ref_mat2, alpha=alpha, beta=beta
    )

    with flag_gems.use_gems():
        res_out = torch.sparse.sampled_addmm(input, mat1, mat2, alpha=alpha, beta=beta)

    _assert_close_dense(
        _csr_to_cpu(res_out).to_dense(),
        ref_out.to_dense(),
        dtype,
        K,
        1e-3 if dtype in _LOW_PREC_DTYPES else 1e-4,
    )


@pytest.mark.sparse_sampled_addmm_large
@pytest.mark.parametrize("M, N, K", LARGE_MNK)
@pytest.mark.parametrize("dtype", DTYPES)
def test_sparse_sampled_addmm_large(M, N, K, dtype):
    device = flag_gems.device

    mat1 = torch.randn((M, K), dtype=dtype, device=device)
    mat2 = torch.randn((K, N), dtype=dtype, device=device)
    input = _make_sparse_csr((M, N), dtype, device, sparsity=0.5)

    ref_input = _to_cpu_ref(input)
    ref_mat1 = utils.to_reference(mat1, upcast=True).cpu()
    ref_mat2 = utils.to_reference(mat2, upcast=True).cpu()

    ref_out = torch.sparse.sampled_addmm(ref_input, ref_mat1, ref_mat2)

    with flag_gems.use_gems():
        res_out = torch.sparse.sampled_addmm(input, mat1, mat2)

    _assert_close_dense(
        _csr_to_cpu(res_out).to_dense(),
        ref_out.to_dense(),
        dtype,
        K,
        1e-3 if dtype in _LOW_PREC_DTYPES else 1e-4,
    )


@pytest.mark.sparse_sampled_addmm_out
@pytest.mark.parametrize("M, N, K", [(2, 2, 2), (5, 5, 5), (5, 5, 0)])
@pytest.mark.parametrize("bcast_input, batch_shape", BCAST_AND_BATCH)
@pytest.mark.parametrize("alpha, beta", [(1.0, 1.0), (2.0, 0.0)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_sparse_sampled_addmm_out(
    M, N, K, batch_shape, bcast_input, alpha, beta, dtype
):
    device = flag_gems.device

    mat1_shape = batch_shape + (M, K)
    mat2_shape = batch_shape + (K, N)
    input_shape = (M, N) if bcast_input else batch_shape + (M, N)
    out_shape = batch_shape + (M, N)

    mat1 = torch.randn(mat1_shape, dtype=dtype, device=device)
    mat2 = torch.randn(mat2_shape, dtype=dtype, device=device)
    input = _make_sparse_csr(input_shape, dtype, device, sparsity=0.5)

    ref_input = _to_cpu_ref(input)
    ref_mat1 = utils.to_reference(mat1, upcast=True).cpu()
    ref_mat2 = utils.to_reference(mat2, upcast=True).cpu()

    out = _broadcast_sparse_csr(input, out_shape)
    with flag_gems.use_gems():
        torch.sparse.sampled_addmm(input, mat1, mat2, alpha=alpha, beta=beta, out=out)

    ref_out = _broadcast_sparse_csr(ref_input, out_shape)
    torch.sparse.sampled_addmm(
        ref_input, ref_mat1, ref_mat2, alpha=alpha, beta=beta, out=ref_out
    )

    _assert_close_dense(
        _csr_to_cpu(out).to_dense(),
        ref_out.to_dense(),
        dtype,
        K,
        1e-3 if dtype in _LOW_PREC_DTYPES else 1e-4,
    )
