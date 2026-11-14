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
def test_hygon_mm_scientific_path_does_not_fallback(monkeypatch):
    provider = inspect.getmodule(flag_gems.mm)

    def unexpected_fallback(*args, **kwargs):
        pytest.fail("Hygon FP32 scientific mm unexpectedly used _mm_default")

    monkeypatch.setattr(provider, "_mm_default", unexpected_fallback)
    for M, N, K in [(4096, 64, 64), (4096, 32, 64)]:
        cpu_a = torch.randn((M, K), dtype=torch.float32)
        cpu_b = torch.randn((K, N), dtype=torch.float32)
        actual = flag_gems.mm(cpu_a.to(flag_gems.device), cpu_b.to(flag_gems.device))
        expected = torch.mm(cpu_a, cpu_b)
        torch.testing.assert_close(
            actual.cpu(), expected, rtol=1e-4, atol=1e-5, equal_nan=True
        )


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
