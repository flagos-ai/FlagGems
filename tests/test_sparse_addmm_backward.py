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

"""Accuracy tests for the ``_sparse_addmm`` autograd backward.

ATen has no ``_sparse_addmm_backward`` schema: the gradient comes from the
engine-generated ``SparseAddmmBackward0`` formula.  These cases run the whole
autograd chain through the FlagGems entry and compare each input gradient with
a float64 reference built from the same stored pattern.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

# The forward supports fp16/bf16/fp32/fp64.  CUDA's native sparse backward
# rejects fp16/bf16 (``sampled_addmm_out_sparse_csr`` / ``addmm_sparse_cuda``
# are not implemented for Half/BFloat16), so those cases compare against the
# same mathematical reference in float64 instead of native autograd.
DTYPES = utils.FLOAT_DTYPES + [torch.float64]

# Native ATen builds the sparse gradient for CSR and COO only: CSC raises
# ``sparse_addmm_sparse_backward: unsupported combination of layouts``.  The
# FlagGems backward supports all three, and the CSC case is checked against the
# reference rather than against a native gradient.
LAYOUTS = ["csr", "coo", "csc"]

# Native autograd builds the sparse gradient for CSR with
# ``sparse_sampled_addmm`` and for COO with ``addmm_sparse_cuda`` followed by
# ``sparse_mask``.  The COO route is measurably less accurate than a
# float64 sample of the product (up to ~1e-3 in fp32 on the patterns below),
# so only the CSR route is used as the native authority; COO and CSC are
# checked against the float64 reference instead.
_NATIVE_SPARSE_GRAD = {
    (torch.float32, "csr"),
    (torch.float64, "csr"),
}

# Bias broadcast forms exercised explicitly; ``mat2`` is always the second
# input and ``mat1`` the sparse one.
BROADCAST_SHAPES = ["none", "vector", "row", "col", "scalar"]

if QUICK_MODE:
    MNK = [(8, 8, 8)]
    SPARSITIES = [0.5]
    BETA_ALPHA = [(1.0, 1.0)]
else:
    MNK = [(5, 5, 5), (32, 24, 16), (64, 48, 32), (1, 32, 1), (0, 0, 0)]
    SPARSITIES = [0.0, 0.5, 0.9, 1.0]
    BETA_ALPHA = [(1.0, 1.0), (2.0, 0.5), (0.0, 1.0), (1.0, 0.0)]


def _make_sparse(M, K, dtype, device, sparsity, layout):
    dense = torch.randn((M, K), dtype=torch.float32, device=device)
    if sparsity == 0.0:
        mask = torch.ones((M, K), dtype=torch.bool, device=device)
    elif sparsity == 1.0:
        mask = torch.zeros((M, K), dtype=torch.bool, device=device)
    else:
        mask = torch.rand((M, K), device=device) > sparsity
    dense = (dense * mask).to(dtype)
    if layout == "csr":
        return dense.to_sparse_csr()
    if layout == "csc":
        return dense.to_sparse_csc()
    return dense.to_sparse_coo()


def _bias_for(kind, M, N, dtype, device):
    if kind == "vector":
        return torch.randn((N,), dtype=dtype, device=device)
    if kind == "row":
        return torch.randn((1, N), dtype=dtype, device=device)
    if kind == "col":
        return torch.randn((M, 1), dtype=dtype, device=device)
    if kind == "scalar":
        return torch.randn((), dtype=dtype, device=device)
    return torch.randn((M, N), dtype=dtype, device=device)


def _reference_grads(grad_output, bias, mat1, mat2, beta, alpha):
    """First-order gradients in float64 from the stored pattern.

    The sparse gradient keeps ``mat1``'s pattern: the dense product is sampled
    at the stored coordinates, exactly what ATen's ``sparse_sampled_addmm`` /
    ``sparse_mask`` combination produces.
    """
    go = utils.to_reference(grad_output, True)
    ref_bias = utils.to_reference(bias, True)
    ref_mat2 = utils.to_reference(mat2, True)
    dense = utils.to_reference(mat1.to_dense(), True)
    pattern = dense != 0
    grad_self = beta * go
    if ref_bias.shape != grad_self.shape:
        while grad_self.dim() > ref_bias.dim():
            grad_self = grad_self.sum(0)
        for dim in range(ref_bias.dim()):
            if ref_bias.shape[dim] == 1 and grad_self.shape[dim] > 1:
                grad_self = grad_self.sum(dim, keepdim=True)
    grad_mat1 = alpha * (go @ ref_mat2.t())
    grad_mat1 = (grad_mat1 * pattern).to(dense.dtype)
    grad_mat2 = alpha * (dense.t() @ go)
    return grad_self, grad_mat1, grad_mat2, pattern


def _gems_grads(grad_output, bias, mat1, mat2, beta, alpha):
    """Run the FlagGems autograd chain and return the three gradients."""
    res_bias = bias.detach().clone().requires_grad_(True)
    res_mat1 = _detach_requires_grad(mat1)
    res_mat2 = mat2.detach().clone().requires_grad_(True)
    out = flag_gems._sparse_addmm(res_bias, res_mat1, res_mat2, beta=beta, alpha=alpha)
    grads = torch.autograd.grad(out, (res_bias, res_mat1, res_mat2), grad_output)
    return grads, out


def _detach_requires_grad(sparse):
    """Clone a sparse tensor and mark it differentiable without densifying."""
    detached = sparse.detach()
    if not detached.is_leaf:
        detached = detached.clone()
    return detached.requires_grad_(True)


@pytest.mark.sparse_addmm_backward
@pytest.mark.parametrize("M, N, K", MNK)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("sparsity", SPARSITIES)
@pytest.mark.parametrize("beta, alpha", BETA_ALPHA)
@pytest.mark.parametrize("dtype", DTYPES)
def test_sparse_addmm_backward(M, N, K, layout, sparsity, beta, alpha, dtype):
    device = flag_gems.device
    mat1 = _make_sparse(M, K, dtype, device, sparsity, layout)
    mat2 = torch.randn((K, N), dtype=dtype, device=device)
    bias = torch.randn((M, N), dtype=dtype, device=device)
    grad_output = torch.randn((M, N), dtype=dtype, device=device)

    ref_self, ref_mat1, ref_mat2, pattern = _reference_grads(
        grad_output, bias, mat1, mat2, beta, alpha
    )
    grads, out = _gems_grads(grad_output, bias, mat1, mat2, beta, alpha)

    assert type(out.grad_fn).__name__ == "_SparseAddmmFunctionBackward", type(
        out.grad_fn
    ).__name__
    assert grads[1].layout == mat1.layout
    assert grads[1].values().numel() == mat1.values().numel()

    atol = 2e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4

    utils.gems_assert_close(grads[0], ref_self.to(dtype), dtype, atol=atol)
    utils.gems_assert_close(grads[2], ref_mat2.to(dtype), dtype, atol=atol)

    # Compare on the dense grid: stored order is layout-specific (CSR and CSC
    # enumerate entries differently) but the values must match elementwise.
    got_dense = grads[1].to_dense()
    utils.gems_assert_close(got_dense, ref_mat1.to(dtype), dtype, atol=atol)


@pytest.mark.sparse_addmm_backward
@pytest.mark.parametrize("dtype", DTYPES)
def test_sparse_addmm_backward_matches_native_autograd(dtype):
    """Native CSR/COO gradients are the authority for the supported dtypes."""
    device = flag_gems.device
    M, N, K = 32, 24, 16
    for layout in LAYOUTS:
        if (dtype, layout) not in _NATIVE_SPARSE_GRAD:
            continue
        mat1 = _make_sparse(M, K, dtype, device, 0.5, layout)
        mat2 = torch.randn((K, N), dtype=dtype, device=device)
        bias = torch.randn((M, N), dtype=dtype, device=device)
        grad_output = torch.randn((M, N), dtype=dtype, device=device)

        ref_bias = bias.detach().clone().requires_grad_(True)
        ref_mat1 = _detach_requires_grad(mat1)
        ref_mat2 = mat2.detach().clone().requires_grad_(True)
        ref_out = torch.ops.aten._sparse_addmm(
            ref_bias, ref_mat1, ref_mat2, beta=1.5, alpha=0.75
        )
        native = torch.autograd.grad(
            ref_out, (ref_bias, ref_mat1, ref_mat2), grad_output
        )

        grads, _ = _gems_grads(grad_output, bias, mat1, mat2, 1.5, 0.75)
        atol = 1e-4
        utils.gems_assert_close(grads[0], native[0], dtype, atol=atol)
        utils.gems_assert_close(grads[2], native[2], dtype, atol=atol)
        # The layouts enumerate stored entries differently, so compare the
        # gradient on the dense grid rather than raw storage order.
        utils.gems_assert_close(
            grads[1].to_dense(), native[1].to_dense(), dtype, atol=atol
        )


@pytest.mark.sparse_addmm_backward
@pytest.mark.parametrize("bias_kind", BROADCAST_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_addmm_backward_broadcast(bias_kind, dtype):
    """The bias gradient must be reduced back to ``self``'s broadcast shape."""
    device = flag_gems.device
    M, N, K = 16, 16, 16
    mat1 = _make_sparse(M, K, dtype, device, 0.5, "csr")
    mat2 = torch.randn((K, N), dtype=dtype, device=device)
    bias = _bias_for(bias_kind, M, N, dtype, device)
    grad_output = torch.randn((M, N), dtype=dtype, device=device)

    ref_self, _, _, _ = _reference_grads(grad_output, bias, mat1, mat2, 1.5, 0.75)
    grads, _ = _gems_grads(grad_output, bias, mat1, mat2, 1.5, 0.75)

    assert grads[0].shape == bias.shape
    atol = 2e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
    utils.gems_assert_close(grads[0], ref_self.to(dtype), dtype, atol=atol)


@pytest.mark.sparse_addmm_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_addmm_backward_needs_input_grad(dtype):
    """Only requested inputs get a gradient; the others stay ``None``."""
    device = flag_gems.device
    M, N, K = 16, 16, 16
    mat1 = _make_sparse(M, K, dtype, device, 0.5, "csr")
    mat2 = torch.randn((K, N), dtype=dtype, device=device)
    bias = torch.randn((M, N), dtype=dtype, device=device)
    grad_output = torch.randn((M, N), dtype=dtype, device=device)

    from flag_gems.ops._sparse_addmm_backward import sparse_addmm_backward

    grad_self, grad_mat1, grad_mat2 = sparse_addmm_backward(
        grad_output, bias, mat1, mat2, beta=1.0, alpha=1.0
    )
    assert grad_self is not None and grad_mat1 is not None and grad_mat2 is not None

    grad_self, grad_mat1, grad_mat2 = sparse_addmm_backward(
        grad_output,
        bias,
        mat1,
        mat2,
        beta=1.0,
        alpha=1.0,
        needs_input_grad=(True, False, False),
    )
    assert grad_self is not None
    assert grad_mat1 is None and grad_mat2 is None


@pytest.mark.sparse_addmm_backward
@pytest.mark.parametrize("beta, alpha", [(0.0, 1.0), (1.0, 0.0), (0.0, 0.0)])
def test_sparse_addmm_backward_zero_scalars(beta, alpha):
    """``beta``/``alpha`` zero still produce the correctly-scaled gradients."""
    device = flag_gems.device
    dtype = torch.float32
    M, N, K = 16, 16, 16
    mat1 = _make_sparse(M, K, dtype, device, 0.5, "csr")
    mat2 = torch.randn((K, N), dtype=dtype, device=device)
    bias = torch.randn((M, N), dtype=dtype, device=device)
    grad_output = torch.randn((M, N), dtype=dtype, device=device)

    ref_self, ref_mat1, ref_mat2, pattern = _reference_grads(
        grad_output, bias, mat1, mat2, beta, alpha
    )
    grads, _ = _gems_grads(grad_output, bias, mat1, mat2, beta, alpha)

    utils.gems_assert_close(grads[0], ref_self.to(dtype), dtype, atol=1e-5)
    utils.gems_assert_close(grads[2], ref_mat2.to(dtype), dtype, atol=1e-5)
    utils.gems_assert_close(grads[1].values(), ref_mat1.to(dtype)[pattern], dtype)


@pytest.mark.sparse_addmm_backward
@pytest.mark.parametrize("dtype", [torch.float32])
def test_sparse_addmm_backward_dispatch(dtype):
    """``torch.ops.aten._sparse_addmm`` under ``use_gems()`` takes the bridge."""
    device = flag_gems.device
    M, N, K = 32, 24, 16
    mat1 = _make_sparse(M, K, dtype, device, 0.5, "csr")
    mat2 = torch.randn((K, N), dtype=dtype, device=device)
    bias = torch.randn((M, N), dtype=dtype, device=device, requires_grad=True)
    grad_output = torch.randn((M, N), dtype=dtype, device=device)

    # Only the sparse keys are registered here (the CI-compliant way to
    # activate dispatch: use_gems() is banned in test files), so the aten call
    # lands on the FlagGems device entry and the autograd bridge owns the
    # differentiable path.
    flag_gems.only_enable(include=["_sparse_addmm"])
    out = torch.ops.aten._sparse_addmm(bias, mat1, mat2, beta=1.5, alpha=0.75)
    assert "_SparseAddmmFunctionBackward" in type(out.grad_fn).__name__
    grads = torch.autograd.grad(out, (bias,), grad_output)

    assert torch.equal(grads[0], 1.5 * grad_output)


@pytest.mark.sparse_addmm_backward
@pytest.mark.parametrize("dtype", [torch.float32])
def test_sparse_addmm_backward_single_kernel_attribution(dtype):
    """The FlagGems backward launches Triton kernels, not ATen's sparse ones."""
    from torch.profiler import ProfilerActivity, profile

    device = flag_gems.device
    M, N, K = 256, 128, 192
    mat1 = _make_sparse(M, K, dtype, device, 0.9, "csr")
    mat2 = torch.randn((K, N), dtype=dtype, device=device)
    bias = torch.randn((M, N), dtype=dtype, device=device)
    grad_output = torch.randn((M, N), dtype=dtype, device=device)

    grads, out = _gems_grads(grad_output, bias, mat1, mat2, 1.0, 1.0)
    res_bias = bias.detach().clone().requires_grad_(True)
    res_mat1 = _detach_requires_grad(mat1)
    res_mat2 = mat2.detach().clone().requires_grad_(True)
    out = flag_gems._sparse_addmm(res_bias, res_mat1, res_mat2)

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        torch.autograd.grad(out, (res_bias, res_mat1, res_mat2), grad_output)
        flag_gems.runtime.torch_device_fn.synchronize()

    kernel_names = [event.key for event in prof.key_averages() if event.device_time > 0]
    assert kernel_names, "the backward launched no device kernel"
    # The sparse work must be FlagGems' Triton kernels, never cuSPARSE.
    assert not any("cusparse" in name.lower() for name in kernel_names), kernel_names
    assert any("sparse_addmm" in name for name in kernel_names), kernel_names


@pytest.mark.sparse_addmm_backward
@pytest.mark.parametrize("dtype", [torch.float32])
def test_sparse_addmm_backward_empty_pattern(dtype):
    """A pattern with no stored entries still yields shaped, zero gradients."""
    device = flag_gems.device
    M, N, K = 16, 8, 12
    crow = torch.zeros(M + 1, dtype=torch.int64, device=device)
    mat1 = torch.sparse_csr_tensor(
        crow,
        torch.zeros(0, dtype=torch.int64, device=device),
        torch.zeros(0, dtype=dtype, device=device),
        size=(M, K),
        dtype=dtype,
        device=device,
    )
    mat2 = torch.randn((K, N), dtype=dtype, device=device)
    bias = torch.randn((M, N), dtype=dtype, device=device)
    grad_output = torch.randn((M, N), dtype=dtype, device=device)

    grads, _ = _gems_grads(grad_output, bias, mat1, mat2, 1.0, 1.0)

    assert grads[1].layout is torch.sparse_csr
    assert grads[1].values().numel() == 0
    assert grads[1].shape == (M, K)
    assert torch.equal(grads[2], torch.zeros((K, N), dtype=dtype, device=device))
    assert torch.equal(grads[0], grad_output)


@pytest.mark.sparse_addmm_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_addmm_backward_inference_unchanged(dtype):
    """A forward without gradients keeps the plain (non-autograd) entry."""
    device = flag_gems.device
    M, N, K = 16, 16, 16
    mat1 = _make_sparse(M, K, dtype, device, 0.5, "csr")
    mat2 = torch.randn((K, N), dtype=dtype, device=device)
    bias = torch.randn((M, N), dtype=dtype, device=device)

    out = flag_gems._sparse_addmm(bias, mat1, mat2, beta=1.5, alpha=0.75)
    assert out.grad_fn is None
    ref = 1.5 * utils.to_reference(bias, True) + 0.75 * (
        utils.to_reference(mat1.to_dense(), True) @ utils.to_reference(mat2, True)
    )
    atol = 2e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
    utils.gems_assert_close(out, ref.to(dtype), dtype, atol=atol)
