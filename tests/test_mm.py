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
    with flag_gems.use_gems():
        res_out = torch.mm(mat1, mat2)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K, atol=_mm_atol_base())


def _cuda_hopper_w8a8_fp8_available():
    tensor_descriptor = getattr(
        getattr(triton, "tools", None), "tensor_descriptor", None
    )
    fn = getattr(flag_gems, "mm_w8a8_fp8", None)
    return (
        fn is not None
        and getattr(fn, "__module__", "") != "flag_gems.ops.mm_w8a8_fp8"
        and flag_gems.device == "cuda"
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] >= 9
        and hasattr(torch, "float8_e4m3fn")
        and hasattr(tensor_descriptor, "TensorDescriptor")
    )


def _thead_w8a8_fp8_available():
    return (
        getattr(flag_gems, "vendor_name", None) == "thead"
        and hasattr(torch, "float8_e4m3fn")
        and hasattr(flag_gems, "mm_w8a8_fp8")
    )


def _mm_w8a8_fp8_available():
    return _thead_w8a8_fp8_available() or _cuda_hopper_w8a8_fp8_available()


def _mm_w8a8_fp8_reference(a, b):
    fp8_dtype = torch.float8_e4m3fn
    fp8_info = torch.finfo(fp8_dtype)

    a_fp32 = a.float()
    a_scale = a_fp32.abs().amax(dim=1).clamp_min(1e-10) / fp8_info.max
    a_fp8 = (a_fp32 / a_scale[:, None]).clamp(fp8_info.min, fp8_info.max).to(fp8_dtype)

    b_fp32 = b.float()
    b_scale = b_fp32.abs().amax(dim=0).clamp_min(1e-10) / fp8_info.max
    b_fp8 = (b_fp32 / b_scale[None, :]).clamp(fp8_info.min, fp8_info.max).to(fp8_dtype)

    return torch.mm(a_fp8.float(), b_fp8.float()) * a_scale[:, None] * b_scale[None, :]


def _mm_w8a8_int8_reference(a, b):
    a_fp32 = a.float()
    a_scale = a_fp32.abs().amax(dim=1).clamp_min(1e-8) / 127.0
    a_q = torch.round(a_fp32 / a_scale[:, None]).clamp(-127, 127)

    b_fp32 = b.float()
    b_scale = b_fp32.abs().amax(dim=0).clamp_min(1e-8) / 127.0
    b_q = torch.round(b_fp32 / b_scale[None, :]).clamp(-127, 127)

    return (a_q @ b_q) * a_scale[:, None] * b_scale[None, :]


@pytest.mark.mm_w8a8_fp8
@pytest.mark.parametrize(
    "M, N, K",
    [
        (1, 16, 16),
        (16, 1, 128),
        (256, 1, 2048),
        (2, 32, 32),
        (8, 64, 64),
        (16, 128, 64),
        (32, 128, 128),
        (64, 256, 128),
        (128, 256, 256),
        (192, 512, 512),
        (256, 768, 1024),
        (512, 1024, 1024),
        # Qwen3.5-35B-A3B-p32768d1024 families from FlagGems#3821
        (16, 1, 2048),
        (16, 64, 2048),
        (16, 256, 2048),
        (16, 1024, 2048),
        (16, 2048, 512),
        (16, 2048, 4096),
        (16, 9216, 2048),
        (16, 12288, 2048),
        (1, 248320, 2048),
    ],
)
@pytest.mark.skipif(
    not _mm_w8a8_fp8_available(),
    reason="mm_w8a8_fp8 requires THead/PPU or CUDA Hopper FP8 TMA support",
)
def test_mm_w8a8_fp8(M, N, K):
    dtype = torch.bfloat16
    torch.manual_seed(0)

    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    reference = (
        _mm_w8a8_int8_reference
        if flag_gems.vendor_name == "thead"
        else _mm_w8a8_fp8_reference
    )
    ref_out = utils.to_reference(reference(mat1, mat2), True)

    res_out = flag_gems.mm_w8a8_fp8(mat1, mat2, out_dtype=dtype)
    out = torch.empty((M, N), dtype=dtype, device=flag_gems.device)
    res_out_reused = flag_gems.mm_w8a8_fp8_out(mat1, mat2, out=out)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)
    utils.gems_assert_close(res_out_reused, ref_out, dtype, reduce_dim=K)


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
    with flag_gems.use_gems():
        res_out = torch.mm(a, b)

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

    with flag_gems.use_gems():
        torch.mm(mat1, mat2, out=out)

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
    with flag_gems.use_gems():
        res_out = torch.mm(mat, mat.t())

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
    with flag_gems.use_gems():
        torch.mm(mat, mat.t(), out=out)

    utils.gems_assert_close(out, ref_out, dtype, reduce_dim=K, atol=_mm_atol_base())
