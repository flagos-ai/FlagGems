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

import importlib
import random

import numpy as np
import pytest
import torch
import triton

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

if QUICK_MODE:
    MNK_SHAPES = [
        (1, 1, 32),
    ]
    FLOAT_DTYPES = [torch.float32]
else:
    MNK_SHAPES = [
        (1, 1, 32),
        (15, 160, 1024),
        (495, 5333, 71),
    ]
    FLOAT_DTYPES = utils.FLOAT_DTYPES


MK_SHAPES = (
    [(1, 32)]
    if QUICK_MODE
    else [
        (1, 32),
        (7, 33),
        (31, 65),
        (160, 1024),
        (257, 96),
        (1023, 255),
        (5333, 71),
    ]
)


def _mm_atol_base():
    """On MetaX C550, the hardware matmul unit has inherent precision limits in
    fp16/bf16 (confirmed by native torch.mm showing the same error vs fp64 ref).
    Use a larger atol base to accommodate hardware accumulation precision."""
    if flag_gems.vendor_name == "metax":
        return 3e-4
    return 1e-4


# Issue #2833: fails at (1, 1, 2)
@pytest.mark.mm
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("b_column_major", [True, False])
def test_mm(M, N, K, dtype, b_column_major):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Issue #2834: Skipping fp32 mm test on tsingmicro platform")

    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    if b_column_major:
        mat2 = torch.randn((N, K), dtype=dtype, device=flag_gems.device).t()
    else:
        mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)

    ref_out = torch.mm(ref_mat1, ref_mat2)
    res_out = flag_gems.mm(mat1, mat2)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K, atol=_mm_atol_base())


@pytest.mark.mm
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_mm_broadcast_stride_zero(dtype):
    """Regression test: broadcast tensors (stride=0) must not crash TMA path."""
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Issue #3794: not working ")
    torch.manual_seed(0)
    M, K, N = 128, 256, 256

    # Simulate the stride=(0,0) tensor that autograd produces from sum().backward():
    # scalar expand -> all strides are 0
    a = torch.randn((), dtype=dtype, device=flag_gems.device).expand(M, K)
    b = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    assert a.stride() == (0, 0)

    ref_a = utils.to_reference(a.contiguous(), True)
    ref_b = utils.to_reference(b, True)

    ref_out = torch.mm(ref_a, ref_b)
    res_out = flag_gems.mm(a, b)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K, atol=_mm_atol_base())


@pytest.mark.mm
def test_mm_out_vllm_tma_column_major_weight():
    """Regression test for vLLM Inductor mm_out with a column-major BF16 weight."""
    torch.manual_seed(0)
    M, K, N = 4096, 4096, 3328
    dtype = torch.bfloat16

    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    mat2_storage = torch.randn((N, K), dtype=dtype, device=flag_gems.device)
    mat2 = mat2_storage.t()
    out = torch.empty((M, N), dtype=dtype, device=flag_gems.device)

    assert mat2.shape == (K, N)
    assert mat2.stride() == (1, K)

    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)
    ref_out = torch.empty((M, N), dtype=ref_mat1.dtype, device=ref_mat1.device)
    torch.mm(ref_mat1, ref_mat2, out=ref_out)

    flag_gems.mm_out(mat1, mat2, out=out)

    utils.gems_assert_close(out, ref_out, dtype, reduce_dim=K, atol=_mm_atol_base())


@pytest.mark.mm
@pytest.mark.skipif(
    not hasattr(
        getattr(getattr(triton, "tools", None), "tensor_descriptor", None),
        "TensorDescriptor",
    )
    or flag_gems.vendor_name != "nvidia"
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] < 9,
    reason="Host TMA TensorDescriptor and Hopper GPU are required for this regression test.",
)
def test_mm_kernel_general_host_tma_vllm_column_major_weight_compile_error():
    """Reproduce the vLLM TMA descriptor compile error for a column-major BF16 weight."""
    from triton.tools.tensor_descriptor import TensorDescriptor

    from flag_gems.runtime.backend._nvidia.hopper.ops.mm import (
        mm_kernel_general_host_tma,
    )

    torch.manual_seed(0)
    M, K, N = 64, 4096, 3328
    dtype = torch.bfloat16

    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    mat2_storage = torch.randn((N, K), dtype=dtype, device=flag_gems.device)
    mat2 = mat2_storage.t()
    out = torch.empty((M, N), dtype=dtype, device=flag_gems.device)

    assert mat2.shape == (K, N)
    assert mat2.stride() == (1, K)

    dummy_block = [1, 1]
    a_desc = TensorDescriptor(mat1, mat1.shape, mat1.stride(), dummy_block)
    b_desc = TensorDescriptor(mat2, mat2.T.shape, mat2.T.stride(), dummy_block)
    c_desc = TensorDescriptor(out, out.shape, out.stride(), dummy_block)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    mm_kernel_general_host_tma.fn.fn[grid](
        a_desc,
        b_desc,
        c_desc,
        M,
        N,
        K,
        mat1.stride(0),
        mat1.stride(1),
        mat2.stride(0),
        mat2.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=64,
        BLOCK_N=128,
        BLOCK_K=64,
        GROUP_M=8,
        A_ROW_MAJOR=True,
        B_ROW_MAJOR=False,
        dtype="bfloat16",
        num_warps=4,
        num_stages=2,
    )


@pytest.mark.mm
@pytest.mark.mm_self_transpose
@pytest.mark.parametrize("M, K", MK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_mm_self_transpose(M, K, dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip(
            "Issue #2834: Skipping fp32 mm self-transpose test on tsingmicro platform"
        )

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)

    mat = torch.randn((K, M), dtype=dtype, device=flag_gems.device).t()
    ref_mat = utils.to_reference(mat, True)

    ref_out = torch.mm(ref_mat, ref_mat.t())
    res_out = flag_gems.mm(mat, mat.t())

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K, atol=_mm_atol_base())


@pytest.mark.mm_out
@pytest.mark.parametrize("M, K", MK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_mm_out_self_transpose(M, K, dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip(
            "Issue #2834: Skipping fp32 mm.out self-transpose test on tsingmicro platform"
        )

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)

    mat = torch.randn((K, M), dtype=dtype, device=flag_gems.device).t()
    out = torch.empty((M, M), dtype=dtype, device=flag_gems.device)
    ref_mat = utils.to_reference(mat, True)
    ref_out = utils.to_reference(out, True)

    torch.mm(ref_mat, ref_mat.t(), out=ref_out)
    flag_gems.mm_out(mat, mat.t(), out=out)

    utils.gems_assert_close(out, ref_out, dtype, reduce_dim=K, atol=_mm_atol_base())


# ---------------------------------------------------------------------------
# Backward coverage
# ---------------------------------------------------------------------------
# ``torch.mm`` has no independently registered ATen backward schema: autograd
# derives it from the forward (``mm_mat1_backward`` / ``mm_mat2_backward``,
# i.e. two GEMMs).  FlagGems exposes the same two products through
# ``flag_gems.mm_backward`` and attaches them to ``flag_gems.mm`` with an
# ``autograd.Function``.  The cases below follow ``BlasBenchmark``'s M/N/K
# inputs and cover non-square shapes, a column-major right operand, one-sided
# and two-sided gradients, and zero-length dimensions.

BACKWARD_MNK_SHAPES = (
    [(1, 1, 32)]
    if QUICK_MODE
    else [
        (1, 1, 32),
        (15, 160, 1024),
        (495, 5333, 71),
    ]
)

# A zero-length M, N or K changes which gradient is empty and which is an
# all-zero reduction over no terms; ATen returns them without a kernel launch.
BACKWARD_DEGENERATE_MNK_SHAPES = [
    (0, 8, 16),
    (8, 0, 16),
    (8, 16, 0),
]

GRAD_MODES = ["both", "mat1", "mat2"]


def _make_mm_backward_inputs(M, N, K, dtype, b_column_major, grad_mode):
    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    if b_column_major:
        mat2_storage = torch.randn((N, K), dtype=dtype, device=flag_gems.device)
    else:
        mat2_storage = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    if grad_mode in ("both", "mat1"):
        mat1.requires_grad_(True)
    if grad_mode in ("both", "mat2"):
        mat2_storage.requires_grad_(True)
    mat2 = mat2_storage.t() if b_column_major else mat2_storage
    return mat1, mat2


def _grad_inputs(mat1, mat2, needs_input_grad):
    return tuple(
        x for x, need in zip((mat1, mat2), needs_input_grad, strict=True) if need
    )


@pytest.mark.mm_backward
@pytest.mark.parametrize("M, N, K", BACKWARD_MNK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("b_column_major", [False, True])
@pytest.mark.parametrize("grad_mode", GRAD_MODES)
def test_mm_backward(M, N, K, dtype, b_column_major, grad_mode):
    """Full autograd chain: forward, then gradients of every needed input."""
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Issue #2834: Skipping fp32 mm test on tsingmicro platform")

    mat1, mat2 = _make_mm_backward_inputs(M, N, K, dtype, b_column_major, grad_mode)
    needs_input_grad = (
        grad_mode in ("both", "mat1"),
        grad_mode in ("both", "mat2"),
    )
    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)

    ref_out = torch.mm(ref_mat1, ref_mat2)
    res_out = flag_gems.mm(mat1, mat2)

    assert res_out.shape == ref_out.shape
    assert res_out.dtype == dtype
    if any(needs_input_grad):
        # Prove the candidate really built its own backward node instead of
        # silently falling back to a detached forward.
        assert res_out.grad_fn is not None
        # Generic and Hopper entries both expose a Function-based node; a
        # plain "MmBackward0" would mean the ATen derivative did the work.
        assert type(res_out.grad_fn).__name__.endswith("MmFunctionBackward")

    out_grad = torch.randn_like(res_out)
    ref_grad = utils.to_reference(out_grad, True)

    ref_inputs = _grad_inputs(ref_mat1, ref_mat2, needs_input_grad)
    res_inputs = _grad_inputs(mat1, mat2, needs_input_grad)
    ref_grads = torch.autograd.grad(ref_out, ref_inputs, ref_grad)
    res_grads = torch.autograd.grad(res_out, res_inputs, out_grad)

    for res_grad, ref_grad_out in zip(res_grads, ref_grads, strict=True):
        assert res_grad.shape == ref_grad_out.shape
        assert res_grad.dtype == dtype
    if needs_input_grad[0]:
        utils.gems_assert_close(
            res_grads[0],
            ref_grads[0],
            dtype,
            reduce_dim=max(N, 1),
            atol=_mm_atol_base(),
        )
    if needs_input_grad[1]:
        utils.gems_assert_close(
            res_grads[-1],
            ref_grads[-1],
            dtype,
            reduce_dim=max(M, 1),
            atol=_mm_atol_base(),
        )


@pytest.mark.mm_backward
@pytest.mark.parametrize(
    "M, N, K", BACKWARD_MNK_SHAPES + BACKWARD_DEGENERATE_MNK_SHAPES
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize(
    "needs_input_grad", [(True, True), (True, False), (False, True)]
)
def test_mm_backward_entry(M, N, K, dtype, needs_input_grad):
    """The explicit ``mm_backward`` entry used by the KernelGen catalog."""
    torch.manual_seed(0)
    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    grad_output = torch.randn((M, N), dtype=dtype, device=flag_gems.device)

    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)
    ref_grad_output = utils.to_reference(grad_output, True)
    ref_grad_a = (
        torch.mm(ref_grad_output, ref_mat2.t()) if needs_input_grad[0] else None
    )
    ref_grad_b = (
        torch.mm(ref_mat1.t(), ref_grad_output) if needs_input_grad[1] else None
    )

    grad_a, grad_b = flag_gems.mm_backward(grad_output, mat1, mat2, needs_input_grad)

    if needs_input_grad[0]:
        assert grad_a.shape == (M, K) and grad_a.dtype == dtype
        assert grad_a.is_contiguous()
        utils.gems_assert_close(
            grad_a, ref_grad_a, dtype, reduce_dim=max(N, 1), atol=_mm_atol_base()
        )
    else:
        assert grad_a is None
    if needs_input_grad[1]:
        assert grad_b.shape == (K, N) and grad_b.dtype == dtype
        assert grad_b.is_contiguous()
        utils.gems_assert_close(
            grad_b, ref_grad_b, dtype, reduce_dim=max(M, 1), atol=_mm_atol_base()
        )
    else:
        assert grad_b is None


@pytest.mark.mm_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_mm_backward_use_gems(dtype):
    """``torch.mm`` under ``use_gems()`` must differentiate through FlagGems."""
    M, N, K = 128, 256, 192
    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    # require grad before the reference copy so the fp64 reference keeps its
    # gradient history through the cast.
    mat1.requires_grad_(True)
    mat2.requires_grad_(True)
    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)
    out_grad = torch.randn((M, N), dtype=dtype, device=flag_gems.device)
    ref_grad = utils.to_reference(out_grad, True)

    with flag_gems.use_gems():
        res_out = torch.mm(mat1, mat2)
        res_grad_a, res_grad_b = torch.autograd.grad(res_out, (mat1, mat2), (out_grad,))

    ref_out = torch.mm(ref_mat1, ref_mat2)
    ref_grad_a, ref_grad_b = torch.autograd.grad(
        ref_out, (ref_mat1, ref_mat2), (ref_grad,)
    )

    assert res_out.grad_fn is not None
    utils.gems_assert_close(
        res_grad_a,
        ref_grad_a,
        dtype,
        reduce_dim=N,
        atol=_mm_atol_base(),
    )
    utils.gems_assert_close(
        res_grad_b,
        ref_grad_b,
        dtype,
        reduce_dim=M,
        atol=_mm_atol_base(),
    )


@pytest.mark.mm_backward
@pytest.mark.parametrize("M, N, K", BACKWARD_MNK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("path", ["fused", "tuned"])
def test_mm_backward_selectable_path(monkeypatch, M, N, K, dtype, path):
    """Both launch strategies must produce the same gradients.

    ``mm_backward`` picks one of them from the problem size; the limit is
    monkeypatched here so the small CPU-reference cases cover both branches
    instead of only the host-bound one.
    """
    # ``flag_gems.ops.mm_backward`` the attribute is the function, so resolve
    # the implementation module through importlib instead of attribute access.
    backward_module = importlib.import_module("flag_gems.ops.mm_backward")

    limit = 0 if path == "tuned" else float("inf")
    monkeypatch.setitem(backward_module._HOST_BOUND_LIMIT, dtype, limit)

    torch.manual_seed(0)
    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    grad_output = torch.randn((M, N), dtype=dtype, device=flag_gems.device)
    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)
    ref_grad_output = utils.to_reference(grad_output, True)
    ref_grad_a = torch.mm(ref_grad_output, ref_mat2.t())
    ref_grad_b = torch.mm(ref_mat1.t(), ref_grad_output)

    grad_a, grad_b = flag_gems.mm_backward(grad_output, mat1, mat2, (True, True))

    assert (M, K) == grad_a.shape and (K, N) == grad_b.shape
    utils.gems_assert_close(
        grad_a, ref_grad_a, dtype, reduce_dim=max(N, 1), atol=_mm_atol_base()
    )
    utils.gems_assert_close(
        grad_b, ref_grad_b, dtype, reduce_dim=max(M, 1), atol=_mm_atol_base()
    )


@pytest.mark.mm_backward
@pytest.mark.skipif(
    not torch.cuda.is_available() or flag_gems.device != "cuda",
    reason="CUDA profiler activity is required to attribute the backward kernels.",
)
def test_mm_backward_launches_flag_gems_kernels():
    """Attribute the timed kernels: the backward must run FlagGems Triton code."""
    from torch.profiler import ProfilerActivity, profile

    M, N, K = 64, 96, 80
    dtype = torch.float32
    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    out_grad = torch.randn((M, N), dtype=dtype, device=flag_gems.device)

    mat1.requires_grad_(True)
    mat2.requires_grad_(True)
    out = flag_gems.mm(mat1, mat2)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        torch.autograd.grad(out, (mat1, mat2), (out_grad,))
    kernels = {
        event.key: event.count
        for event in prof.key_averages()
        if event.self_device_time_total > 0
    }
    # The backward is either the fused single-launch kernel or the tuned
    # forward kernels of the GPU-bound path; both are FlagGems Triton code.
    backward_kernels = {
        name: count
        for name, count in kernels.items()
        if "_mm_backward_kernel" in name or "mm_kernel" in name
    }
    assert backward_kernels, kernels
    assert all(count > 0 for count in backward_kernels.values()), backward_kernels
