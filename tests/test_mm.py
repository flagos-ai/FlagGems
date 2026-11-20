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

import inspect
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


# Deduplicated FP32 matrix signatures from ESM-2 and GenScore inference.
# Values are seeded reconstructions, not captured model tensors.
HYGON_SCIENTIFIC_MM_SHAPES = [
    (303, 320, 320),
    (303, 33, 320),
    (29403, 1, 120),
    (1220, 128, 128),
    (2806, 128, 128),
    (1220, 256, 128),
    (1220, 128, 256),
    (2806, 256, 128),
    (2806, 128, 256),
    (4819, 128, 128),
    (126636, 128, 128),
    (4819, 256, 128),
    (4819, 128, 256),
    (126636, 256, 128),
    (126636, 128, 256),
]


def _hygon_mm_storage(shape, stride, offset, dtype, seed):
    extent = (
        offset + 1 + sum(max(dim - 1, 0) * step for dim, step in zip(shape, stride))
    )
    generator = torch.Generator().manual_seed(seed)
    backing = (torch.randn(extent, generator=generator) * 0.125).to(dtype)
    reference = backing.as_strided(shape, stride, offset)
    device_backing = backing.to(flag_gems.device)
    device_view = device_backing.as_strided(shape, stride, offset)
    return reference, backing, device_view, device_backing


def _assert_hygon_mm(
    reference_a, backing_a, a, storage_a, reference_b, backing_b, b, storage_b
):
    metadata = [
        (x.shape, x.stride(), x.storage_offset(), x.dtype, x.device) for x in (a, b)
    ]
    expected = torch.mm(reference_a, reference_b)
    actual = flag_gems.mm(a, b)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.device == a.device
    if a.dtype in (torch.float16, torch.bfloat16):
        utils.gems_assert_close(
            actual.cpu(),
            expected,
            a.dtype,
            equal_nan=True,
            reduce_dim=reference_a.shape[1],
            atol=_mm_atol_base(),
        )
    else:
        tolerance = (1e-7, 1e-9) if a.dtype == torch.float64 else (1e-4, 1e-5)
        torch.testing.assert_close(
            actual.cpu(), expected, rtol=tolerance[0], atol=tolerance[1], equal_nan=True
        )
    assert actual.numel() == 0 or all(actual.data_ptr() != x.data_ptr() for x in (a, b))
    for value, saved in zip((a, b), metadata):
        assert (
            value.shape,
            value.stride(),
            value.storage_offset(),
            value.dtype,
            value.device,
        ) == saved
    # The backing arrays are 1D stride-one tensors, avoiding size-one view traps.
    assert torch.equal(storage_a.cpu().view(torch.uint8), backing_a.view(torch.uint8))
    assert torch.equal(storage_b.cpu().view(torch.uint8), backing_b.view(torch.uint8))
    return actual.cpu(), expected


@pytest.mark.mm
@pytest.mark.skipif(
    flag_gems.vendor_name != "hygon", reason="Hygon schedule regression"
)
@pytest.mark.parametrize("M,N,K", HYGON_SCIENTIFIC_MM_SHAPES)
def test_hygon_mm_scientific_shapes(M, N, K):
    left = _hygon_mm_storage((M, K), (K, 1), 0, torch.float32, 73)
    right = _hygon_mm_storage((K, N), (1, K), 0, torch.float32, 91)
    _assert_hygon_mm(*left, *right)


@pytest.mark.mm
@pytest.mark.skipif(
    flag_gems.vendor_name != "hygon", reason="Hygon schedule regression"
)
@pytest.mark.parametrize(
    "shape, column_major, expected_route",
    [
        ((303, 320, 320), True, "_mm_tiled"),
        ((29403, 1, 120), True, "_mm_vector"),
        ((1024, 1024, 1024), True, "_mm_default"),
        ((303, 320, 320), False, "_mm_default"),
    ],
)
def test_hygon_mm_scientific_route(shape, column_major, expected_route, monkeypatch):
    provider = inspect.getmodule(flag_gems.mm)
    m, n, k = shape
    calls = []

    class LaunchSpy:
        def __init__(self, name):
            self.name = name
            self.kernel = getattr(provider, name)

        def __getitem__(self, grid):
            launch = self.kernel[grid]

            def run(*args, **kwargs):
                calls.append(self.name)
                return launch(*args, **kwargs)

            return run

    default = provider._mm_default

    def observe_default(a, b):
        calls.append("_mm_default")
        return default(a, b)

    for name in ("_mm_tiled", "_mm_vector"):
        monkeypatch.setattr(provider, name, LaunchSpy(name))
    monkeypatch.setattr(provider, "_mm_default", observe_default)
    cpu_a = torch.randn((m, k), dtype=torch.float32)
    a = cpu_a.to(flag_gems.device)
    if column_major:
        rhs = torch.randn((n, k), dtype=torch.float32)
        cpu_b, b = rhs.t(), rhs.to(flag_gems.device).t()
    else:
        cpu_b = torch.randn((k, n), dtype=torch.float32)
        b = cpu_b.to(flag_gems.device)
    actual = flag_gems.mm(a, b)
    expected = torch.mm(cpu_a, cpu_b)
    assert calls == [expected_route]
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-4)


@pytest.mark.mm
@pytest.mark.skipif(
    flag_gems.vendor_name != "hygon", reason="Hygon schedule regression"
)
@pytest.mark.parametrize(
    "M,N,K", [(0, 5, 7), (7, 0, 5), (7, 5, 0), (7, 5, 1), (17, 33, 65)]
)
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_hygon_mm_edges_and_offset_strides(M, N, K, dtype):
    left = _hygon_mm_storage((M, K), (max(2 * K + 3, 1), 2), 3, dtype, 4)
    right = _hygon_mm_storage((K, N), (2, max(2 * K + 5, 1)), 5, dtype, 8)
    _assert_hygon_mm(*left, *right)


@pytest.mark.mm
@pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="Hygon rank-one semantics")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_hygon_mm_rank_one_specials(dtype):
    values = torch.tensor(
        [
            -float("inf"),
            -1.0,
            -0.0,
            0.0,
            torch.finfo(dtype).tiny / 8,
            1.0,
            float("inf"),
            float("nan"),
        ],
        dtype=dtype,
    )
    a, b = values[:, None], values[None, :]
    expected = torch.mm(a, b)
    actual = flag_gems.mm(a.to(flag_gems.device), b.to(flag_gems.device)).cpu()
    tolerance = (1e-7, 1e-9) if dtype == torch.float64 else (1e-4, 1e-5)
    torch.testing.assert_close(
        actual, expected, rtol=tolerance[0], atol=tolerance[1], equal_nan=True
    )
    zeros = expected == 0
    assert torch.equal(torch.signbit(actual[zeros]), torch.signbit(expected[zeros]))


@pytest.mark.mm
@pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="Hygon FP32 schedules")
@pytest.mark.parametrize("M,N,K", [(17, 1, 120), (17, 33, 65)])
def test_hygon_mm_schedule_specials(M, N, K):
    left = list(_hygon_mm_storage((M, K), (K, 1), 0, torch.float32, 11))
    right = list(_hygon_mm_storage((K, N), (1, K), 0, torch.float32, 13))
    left[0][0, 0] = float("inf")
    left[0][1, 0] = -float("inf")
    left[0][2, 0] = float("nan")
    left[0][3, :] = -0.0
    right[0][0, :] = 1.0
    # Refresh the device backing after setting the CPU reference special values.
    left[3] = left[1].to(flag_gems.device)
    left[2] = left[3].as_strided((M, K), (K, 1), 0)
    right[3] = right[1].to(flag_gems.device)
    right[2] = right[3].as_strided((K, N), (1, K), 0)
    _assert_hygon_mm(*left, *right)
