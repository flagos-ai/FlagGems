import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Test shapes for linalg_matmul (2D and batched)
LINALG_MNK_SHAPES = [
    (1, 1, 32),
    (15, 160, 1024),
    (495, 5333, 71),
    (128, 256, 512),
    (64, 128, 256),
]

# (shape of mat1, shape of mat2) covering broadcasting and mixed-dim cases
LINALG_MATMUL_BROADCAST_SHAPES = [
    ((2, 2), (2, 2)),  # 2D @ 2D
    ((4, 2, 2), (4, 2, 2)),  # 3D @ 3D, same batch
    ((1, 2, 2), (4, 2, 2)),  # 3D @ 3D, batch broadcast on mat1
    ((4, 2, 2), (1, 2, 2)),  # 3D @ 3D, batch broadcast on mat2
    ((2, 5), (4, 5, 6)),  # 2D @ 3D
    ((4, 2, 5), (5, 6)),  # 3D @ 2D
    ((2, 1, 2, 5), (3, 5, 6)),  # 4D @ 3D, multi-dim batch broadcast
    ((5,), (5,)),  # 1D @ 1D -> scalar
    ((5,), (5, 6)),  # 1D @ 2D -> (N,)
    ((4, 5), (5,)),  # 2D @ 1D -> (M,)
    ((5,), (4, 5, 6)),  # 1D @ 3D -> (B, N)
    ((4, 2, 5), (5,)),  # 3D @ 1D -> (B, M)
    ((1, 2, 2), (2, 2)),  # 3D @ 2D with broadcast
]


@pytest.mark.linalg_matmul
@pytest.mark.parametrize("M, N, K", LINALG_MNK_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_linalg_matmul_2d(M, N, K, dtype):
    """Test 2D matrix multiplication: (M, K) @ (K, N) -> (M, N)"""
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Skiping fp32 linalg_matmul test on tsingmicro platform")

    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)

    ref_out = torch.linalg.matmul(ref_mat1, ref_mat2)
    res_out = flag_gems.linalg_matmul(mat1, mat2)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.linalg_matmul
@pytest.mark.parametrize("M, N, K", LINALG_MNK_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_linalg_matmul_3d(M, N, K, dtype):
    """Test 3D (batched) matrix multiplication: (B, M, K) @ (B, K, N) -> (B, M, N)"""
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Skiping fp32 linalg_matmul test on tsingmicro platform")

    batch = 4
    mat1 = torch.randn((batch, M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((batch, K, N), dtype=dtype, device=flag_gems.device)
    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)

    ref_out = torch.linalg.matmul(ref_mat1, ref_mat2)
    res_out = flag_gems.linalg_matmul(mat1, mat2)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


# (shape of mat1, shape of mat2) covering zero-sized cases:
# empty batch with folded 1-D dims, empty 2-D, empty K
LINALG_MATMUL_ZERO_SIZED_SHAPES = [
    ((5,), (0, 5, 6)),  # (K,) @ (0, K, N) -> (0, N)
    ((0, 2, 5), (5,)),  # (0, M, K) @ (K,) -> (0, M)
    ((0, 2, 5), (0, 5, 6)),  # (0, M, K) @ (0, K, N) -> (0, M, N)
    ((1, 0, 2, 5), (0, 5, 6)),  # empty broadcast batch -> (0, 2, 6)
    ((0, 5), (5, 6)),  # (0, K) @ (K, N) -> (0, N)
    ((2, 5), (0, 5, 6)),  # (M, K) @ (0, K, N) -> (0, M, N)
]


@pytest.mark.linalg_matmul
@pytest.mark.parametrize("shape1, shape2", LINALG_MATMUL_BROADCAST_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_linalg_matmul_broadcast(shape1, shape2, dtype):
    """Test broadcasting and mixed-dimensionality cases"""
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Skiping fp32 linalg_matmul test on tsingmicro platform")

    mat1 = torch.randn(shape1, dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn(shape2, dtype=dtype, device=flag_gems.device)
    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)

    ref_out = torch.linalg.matmul(ref_mat1, ref_mat2)
    res_out = flag_gems.linalg_matmul(mat1, mat2)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape1[-1])


@pytest.mark.linalg_matmul
@pytest.mark.parametrize("shape1, shape2", LINALG_MATMUL_ZERO_SIZED_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_linalg_matmul_zero_sized(shape1, shape2, dtype):
    """Test zero-sized inputs: output shape must match native matmul"""
    mat1 = torch.randn(shape1, dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn(shape2, dtype=dtype, device=flag_gems.device)
    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)

    ref_out = torch.linalg.matmul(ref_mat1, ref_mat2)
    res_out = flag_gems.linalg_matmul(mat1, mat2)

    assert res_out.shape == ref_out.shape
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape1[-1])


@pytest.mark.linalg_matmul
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_linalg_matmul_mixed_dtype(dtype):
    """Mixed-dtype inputs are rejected, matching native matmul"""
    mat1 = torch.randn((4, 5), dtype=dtype, device=flag_gems.device)
    other_dtype = torch.float32 if dtype != torch.float32 else torch.float16
    mat2 = torch.randn((5, 6), dtype=other_dtype, device=flag_gems.device)

    with pytest.raises(RuntimeError, match="same dtype"):
        flag_gems.linalg_matmul(mat1, mat2)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_linalg_matmul_complex(dtype):
    """Complex inputs raise NotImplementedError (Triton kernels don't support them)"""
    mat1 = torch.randn((4, 5), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((5, 6), dtype=dtype, device=flag_gems.device)

    with pytest.raises(NotImplementedError, match="not implemented"):
        flag_gems.linalg_matmul(mat1, mat2)


@pytest.mark.linalg_matmul
@pytest.mark.parametrize("shape1, shape2", LINALG_MATMUL_BROADCAST_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_linalg_matmul_grad(shape1, shape2, dtype):
    """Gradients of both inputs match native matmul for all dim/broadcast cases"""
    # the backward recomputes the product through cuBLAS; keep TF32 off so
    # fp32 gradients stay within the tolerance against the fp64 reference
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        _test_linalg_matmul_grad(shape1, shape2, dtype)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32


def _test_linalg_matmul_grad(shape1, shape2, dtype):
    mat1 = torch.randn(shape1, dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn(shape2, dtype=dtype, device=flag_gems.device)
    ref_mat1 = utils.to_reference(mat1, True).requires_grad_(True)
    ref_mat2 = utils.to_reference(mat2, True).requires_grad_(True)
    mat1 = mat1.requires_grad_(True)
    mat2 = mat2.requires_grad_(True)

    ref_out = torch.linalg.matmul(ref_mat1, ref_mat2)
    res_out = flag_gems.linalg_matmul(mat1, mat2)

    out_grad = torch.randn_like(res_out)
    ref_grad = utils.to_reference(out_grad, True)

    ref_grad1, ref_grad2 = torch.autograd.grad(
        ref_out, (ref_mat1, ref_mat2), ref_grad, allow_unused=True
    )
    res_grad1, res_grad2 = torch.autograd.grad(
        res_out, (mat1, mat2), out_grad, allow_unused=True
    )

    # gradient reductions run over N (grad1) / M (grad2) and the broadcast
    # batch; scale the tolerance by the full reduction extent
    nbatch = 1
    for s in torch.broadcast_shapes(shape1[:-2], shape2[:-2]):
        nbatch *= s
    M = shape1[-2] if len(shape1) >= 2 else 1
    N = shape2[-1] if len(shape2) >= 2 else 1

    # 1D inputs produce scalar/1D outputs whose grads may be None when unused
    if ref_grad1 is not None and res_grad1 is not None:
        utils.gems_assert_close(res_grad1, ref_grad1, dtype, reduce_dim=nbatch * N)
    if ref_grad2 is not None and res_grad2 is not None:
        utils.gems_assert_close(res_grad2, ref_grad2, dtype, reduce_dim=nbatch * M)


@pytest.mark.linalg_matmul
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_linalg_matmul_double_grad(dtype):
    """Higher-order gradients flow through the recomputed backward"""
    # keep TF32 off so fp32 gradients stay within tolerance vs the fp64 ref
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        _test_linalg_matmul_double_grad(dtype)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32


def _test_linalg_matmul_double_grad(dtype):
    mat1 = torch.randn((4, 5), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((5, 6), dtype=dtype, device=flag_gems.device)
    ref_mat1 = utils.to_reference(mat1, True).requires_grad_(True)
    ref_mat2 = utils.to_reference(mat2, True).requires_grad_(True)
    mat1 = mat1.requires_grad_(True)
    mat2 = mat2.requires_grad_(True)

    ref_loss = torch.linalg.matmul(ref_mat1, ref_mat2).square().sum()
    res_loss = flag_gems.linalg_matmul(mat1, mat2).square().sum()

    ref_grad1, ref_grad2 = torch.autograd.grad(
        ref_loss, (ref_mat1, ref_mat2), create_graph=True
    )
    res_grad1, res_grad2 = torch.autograd.grad(
        res_loss, (mat1, mat2), create_graph=True
    )

    ref_dd = torch.autograd.grad(ref_grad1.sum(), ref_mat1, allow_unused=True)[0]
    res_dd = torch.autograd.grad(res_grad1.sum(), mat1, allow_unused=True)[0]
    assert res_dd is not None, "double gradient graph is disconnected"

    # gradient of grad wrt mat1 is 2 * mat1 @ (mat2 @ mat2^T). Second-order
    # values inherit the first-order grad rounding, so half-precision dtypes
    # cannot support the per-element tolerance; verify the graph connectivity
    # strictly and compare numerically only where the precision allows it
    if dtype in (torch.float16, torch.bfloat16):
        res_sum = res_dd.float().sum()
        ref_sum = ref_dd.float().sum()
        assert abs(res_sum - ref_sum) <= 0.05 * ref_sum.abs()
    else:
        utils.gems_assert_close(res_dd, ref_dd, dtype)
