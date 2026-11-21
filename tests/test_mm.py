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
from .gemm_utils import GEMM_DTYPES, accuracy_dtype_cases, accuracy_shapes

MNK_SHAPES = accuracy_shapes("mm", QUICK_MODE)


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
@pytest.mark.parametrize(
    "dtype, M, N, K, b_column_major", accuracy_dtype_cases("mm", QUICK_MODE)
)
def test_mm(M, N, K, dtype, b_column_major):
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
@pytest.mark.parametrize("dtype", GEMM_DTYPES)
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
@pytest.mark.parametrize("dtype", GEMM_DTYPES)
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
@pytest.mark.parametrize("dtype", GEMM_DTYPES)
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


# Public mm and mm_out accuracy regressions on Ascend.
@pytest.mark.mm
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="Ascend mm accuracy regression"
)
@pytest.mark.parametrize(
    "M, N, K",
    [
        (4080, 8224, 32),
        (4096, 8192, 33),
        (4352, 10240, 64),
        (4096, 4096, 65),
    ],
)
def test_mm_ascend_large_shapes(M, N, K):
    """Check public mm on large outputs with short reductions."""
    torch.manual_seed(0)
    a = torch.randn((M, K), dtype=torch.float16, device=flag_gems.device)
    b = torch.randn((K, N), dtype=torch.float16, device=flag_gems.device)
    ref = torch.mm(utils.to_reference(a).float(), utils.to_reference(b).float())
    out = flag_gems.mm(a, b)
    utils.gems_assert_close(out, ref, a.dtype, reduce_dim=K)


@pytest.mark.mm
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="Ascend mm accuracy regression"
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("K", [0, 33, 64, 71, 129])
def test_mm_ascend_transposed_inputs_and_k_tails(dtype, K):
    """Cover transposed inputs, irregular dimensions and an empty reduction."""
    torch.manual_seed(0)
    M, N = 145, 97
    a = torch.randn((K, M), dtype=dtype, device=flag_gems.device).t()
    b = torch.randn((N, K), dtype=dtype, device=flag_gems.device).t()
    ref = torch.mm(utils.to_reference(a).float(), utils.to_reference(b).float())
    out = flag_gems.mm(a, b)
    utils.gems_assert_close(out, ref, dtype, reduce_dim=max(K, 1))


@pytest.mark.mm
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="Ascend mm accuracy regression"
)
@pytest.mark.parametrize("use_out", [False, True])
def test_mm_ascend_large_shape_entrypoints(use_out):
    """Check public mm and mm_out with a large output and transposed weight."""
    torch.manual_seed(0)
    M, N, K = 4352, 10240, 33
    a = torch.randn((M, K), dtype=torch.bfloat16, device=flag_gems.device)
    b = torch.randn((N, K), dtype=torch.bfloat16, device=flag_gems.device).t()
    ref = torch.mm(utils.to_reference(a).float(), utils.to_reference(b).float())
    if use_out:
        out = torch.empty((M, N), dtype=a.dtype, device=a.device)
        assert flag_gems.mm_out(a, b, out=out) is out
    else:
        out = flag_gems.mm(a, b)
    utils.gems_assert_close(out, ref, a.dtype, reduce_dim=K)


@pytest.mark.mm
@pytest.mark.mm_out
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend mm")
@pytest.mark.parametrize(
    "M,N,K", [(0, 32, 128), (16, 0, 128), (0, 16, 2048), (0, 0, 0)]
)
@pytest.mark.parametrize("use_out", [False, True])
def test_mm_ascend_review_empty(monkeypatch, M, N, K, use_out):
    ops = flag_gems.mm.__globals__

    class NoLaunch:
        def __getitem__(self, grid):
            pytest.fail(f"empty mm launched grid {grid}")

    for name in list(ops):
        if name.endswith("_kernel"):
            monkeypatch.setitem(ops, name, NoLaunch())
    a = torch.empty((M, K), dtype=torch.bfloat16, device=flag_gems.device)
    b = torch.empty((K, N), dtype=a.dtype, device=a.device)
    if use_out:
        out = torch.empty(0, dtype=a.dtype, device=a.device)
        result = flag_gems.mm_out(a, b, out=out)
        assert result is out
    else:
        result = flag_gems.mm(a, b)
    torch.npu.synchronize()
    assert result.shape == (M, N) and result.numel() == 0


@pytest.mark.mm_out
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend mm")
@pytest.mark.parametrize("layout", ["empty", "wrong_shape", "strided", "alias"])
def test_mm_ascend_review_out_resize(layout):
    a = torch.randn((33, 128), device=flag_gems.device)
    b = torch.randn((128, 17), device=a.device)
    expected = a.cpu() @ b.cpu()
    if layout == "strided":
        out = torch.empty((17, 33), device=a.device).t()
    elif layout == "alias":
        out = a
    else:
        out = torch.empty((0,) if layout == "empty" else (1, 3), device=a.device)
    old_stride = out.stride()
    assert flag_gems.mm_out(a, b, out=out) is out
    torch.npu.synchronize()
    if layout == "strided":
        assert out.stride() == old_stride
    utils.gems_assert_close(out, expected, torch.float32, reduce_dim=128)


@pytest.mark.mm
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend mm")
@pytest.mark.parametrize(
    "M,N,K,dtype,kernel_name",
    [
        (37, 1, 129, torch.float16, "_mm_narrow_n_kernel"),
        (641, 33, 3, torch.float16, "_mm_rectangular_kernel"),
        (257, 16, 2048, torch.bfloat16, "_mm_packed_narrow_kernel"),
        (81, 4, 8704, torch.bfloat16, "_mm_packed_narrow_kernel"),
        (145, 97, 257, torch.float32, "mm_kernel"),
    ],
)
def test_mm_ascend_review_capped_routes(monkeypatch, M, N, K, dtype, kernel_name):
    ops = flag_gems.mm.__globals__
    monkeypatch.setitem(ops, "_MAX_CORE_DIM", 2)
    kernel = ops[kernel_name]
    grids = []

    class CheckedKernel:
        def __getitem__(self, grid):
            def checked(meta):
                resolved = grid(meta) if callable(grid) else grid
                assert resolved == (2,)
                grids.append(resolved)
                return resolved

            return kernel[checked]

    monkeypatch.setitem(ops, kernel_name, CheckedKernel())
    a = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    b = torch.randn((K, N), dtype=dtype, device=a.device)
    expected = a.cpu().float() @ b.cpu().float()
    result = flag_gems.mm(a, b)
    torch.npu.synchronize()
    assert grids
    utils.gems_assert_close(result, expected, dtype, reduce_dim=K)


@pytest.mark.mm
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend mm")
def test_mm_ascend_review_narrow_real_core_dim():
    M, K = 262144, 128
    a = torch.ones((M, K), dtype=torch.float16, device=flag_gems.device)
    b = torch.ones((K, 1), dtype=a.dtype, device=a.device)
    result = flag_gems.mm(a, b)
    torch.npu.synchronize()
    torch.testing.assert_close(
        result.cpu(), torch.full((M, 1), K, dtype=a.dtype), rtol=0, atol=0
    )


@pytest.mark.mm
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend mm")
@pytest.mark.parametrize("family", ["wide", "large"])
def test_mm_ascend_review_panel_loops(family):
    ops = flag_gems.mm.__globals__
    M, N, K = 145, 512 if family == "wide" else 513, 256
    a = torch.randn((M, K), dtype=torch.bfloat16, device=flag_gems.device)
    b = torch.randn((K, N), dtype=a.dtype, device=a.device)
    packed = torch.empty(((N + 255) // 256, K, 256), dtype=a.dtype, device=a.device)
    result = torch.empty((M, N), dtype=a.dtype, device=a.device)
    with ops["_blockified"]():
        ops[f"_mm_pack_{family}_panels_kernel"][(2,)](
            b, packed, N, K, 256, 64, multibuffer=False
        )
        ops[f"_mm_{family}_panel_cube_kernel"][(2,)](
            a,
            packed,
            result,
            M,
            N,
            K,
            128,
            256,
            128,
            multibuffer=True,
            unit_flag=True,
            sync_solver=False,
        )
    torch.npu.synchronize()
    expected = a.cpu().float() @ b.cpu().float()
    utils.gems_assert_close(result, expected, a.dtype, reduce_dim=K)


@pytest.mark.mm
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend mm")
def test_mm_ascend_review_broadcast_materialize(monkeypatch):
    ops = flag_gems.mm.__globals__
    monkeypatch.setitem(ops, "_MAX_CORE_DIM", 2)
    x = torch.arange(129, dtype=torch.float32).to(torch.bfloat16).to(flag_gems.device)
    x = x.expand(73, 129)
    result = ops["_mm_materialize_broadcast"](x)
    torch.npu.synchronize()
    torch.testing.assert_close(result.cpu(), x.cpu(), rtol=0, atol=0)


@pytest.mark.mm
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend mm")
@pytest.mark.parametrize(
    "M,N,K",
    [
        (262144, 1, 128),
        (8388608, 1, 1),
        (4194304, 16, 2048),
        (1048576, 4, 8704),
        (16, 1048576, 2048),
        (2048, 1048576, 2048),
    ],
)
def test_mm_ascend_review_specialized_launch_cap(monkeypatch, M, N, K):
    from contextlib import nullcontext

    ops = flag_gems.mm.__globals__
    grids = []

    class Capture:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                assert not callable(grid)
                grids.append(grid)

            return launch

    for name in list(ops):
        if name.endswith("_kernel"):
            monkeypatch.setitem(ops, name, Capture())
    monkeypatch.setattr(ops["torch_device_fn"], "device", lambda *args: nullcontext())
    a = torch.empty((M, K), device="meta", dtype=torch.bfloat16)
    b = torch.empty((K, N), device="meta", dtype=a.dtype)
    result = flag_gems.mm(a, b)
    assert result.shape == (M, N)
    assert grids and any(grid == (65535,) for grid in grids)
    assert all(0 < grid[0] <= 65535 for grid in grids)


@pytest.mark.mm_out
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend mm")
@pytest.mark.parametrize("invalid", ["dtype", "device"])
def test_mm_ascend_review_out_validation(invalid):
    a = torch.ones((3, 128), device=flag_gems.device)
    b = torch.ones((128, 5), device=a.device)
    out = torch.empty(
        0,
        device="cpu" if invalid == "device" else a.device,
        dtype=torch.float16 if invalid == "dtype" else a.dtype,
    )
    with pytest.raises(RuntimeError, match=invalid):
        flag_gems.mm_out(a, b, out=out)
    assert out.shape == (0,)


# ---------------------------------------------------------------------------
# Backward coverage
# ---------------------------------------------------------------------------
# ``torch.mm`` has no independently registered ATen backward schema: autograd
# derives it from the forward (``mm_mat1_backward`` / ``mm_mat2_backward``,
# i.e. two GEMMs).  FlagGems exposes the same two products through
# the Hopper backend helper and attaches them to ``flag_gems.mm`` with an
# ``autograd.Function``.  The cases below follow ``BlasBenchmark``'s M/N/K
# inputs and cover non-square shapes, a column-major right operand, one-sided
# and two-sided gradients, and zero-length dimensions.

mm_backward = getattr(
    importlib.import_module(flag_gems.mm.__module__), "mm_backward", None
)
BACKWARD_DTYPES = [torch.float32] if QUICK_MODE else utils.FLOAT_DTYPES

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
@pytest.mark.skipif(
    mm_backward is None, reason="Hopper mm backward is unavailable on this backend"
)
@pytest.mark.parametrize("M, N, K", BACKWARD_MNK_SHAPES)
@pytest.mark.parametrize("dtype", BACKWARD_DTYPES)
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
        assert type(res_out.grad_fn).__name__ == "MmFunctionBackward"
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
@pytest.mark.skipif(
    mm_backward is None, reason="Hopper mm backward is unavailable on this backend"
)
@pytest.mark.parametrize(
    "M, N, K", BACKWARD_MNK_SHAPES + BACKWARD_DEGENERATE_MNK_SHAPES
)
@pytest.mark.parametrize("dtype", BACKWARD_DTYPES)
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

    grad_a, grad_b = mm_backward(grad_output, mat1, mat2, needs_input_grad)

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
@pytest.mark.skipif(
    mm_backward is None, reason="Hopper mm backward is unavailable on this backend"
)
@pytest.mark.parametrize("dtype", BACKWARD_DTYPES)
def test_mm_backward_autograd(dtype):
    """The direct FlagGems entry must build the Hopper backward node."""
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

    res_out = flag_gems.mm(mat1, mat2)
    res_grad_a, res_grad_b = torch.autograd.grad(res_out, (mat1, mat2), (out_grad,))

    ref_out = torch.mm(ref_mat1, ref_mat2)
    ref_grad_a, ref_grad_b = torch.autograd.grad(
        ref_out, (ref_mat1, ref_mat2), (ref_grad,)
    )

    assert type(res_out.grad_fn).__name__ == "MmFunctionBackward"
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
@pytest.mark.skipif(
    mm_backward is None, reason="Hopper mm backward is unavailable on this backend"
)
@pytest.mark.parametrize("M, N, K", BACKWARD_MNK_SHAPES)
@pytest.mark.parametrize("dtype", BACKWARD_DTYPES)
@pytest.mark.parametrize("path", ["fused", "tuned"])
def test_mm_backward_selectable_path(monkeypatch, M, N, K, dtype, path):
    """Both launch strategies must produce the same gradients.

    ``mm_backward`` picks one of them from the problem size; the limit is
    monkeypatched here so the small CPU-reference cases cover both branches
    instead of only the host-bound one.
    """
    # Resolve the active Hopper module to exercise both backward launch paths.
    backward_module = importlib.import_module(flag_gems.mm.__module__)

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

    grad_a, grad_b = mm_backward(grad_output, mat1, mat2, (True, True))

    assert (M, K) == grad_a.shape and (K, N) == grad_b.shape
    utils.gems_assert_close(
        grad_a, ref_grad_a, dtype, reduce_dim=max(N, 1), atol=_mm_atol_base()
    )
    utils.gems_assert_close(
        grad_b, ref_grad_b, dtype, reduce_dim=max(M, 1), atol=_mm_atol_base()
    )


@pytest.mark.mm_backward
@pytest.mark.skipif(
    mm_backward is None, reason="Hopper mm backward is unavailable on this backend"
)
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
