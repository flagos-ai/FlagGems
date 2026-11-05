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

import pytest
import torch
import triton
from packaging import version

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

IS_KUNLUNXIN = flag_gems.vendor_name == "kunlunxin"

if QUICK_MODE:
    MNK_SHAPES = [
        (1, 1, 32),
    ]
    VECTOR_BIAS_MNK_SHAPES = [(64, 256, 184)]
    FLOAT_DTYPES = [torch.float32]
else:
    MNK_SHAPES = [
        (1, 1, 32),
        (15, 160, 1024),
        (495, 5333, 71),
    ]
    VECTOR_BIAS_MNK_SHAPES = [
        (64, 256, 184),
        (128, 512, 512),
        (256, 256, 1024),
        (512, 1024, 1024),
        (1024, 512, 1536),
    ]
    FLOAT_DTYPES = utils.FLOAT_DTYPES


_ADDMM_LAYOUT_BIAS_VENDORS = (
    "ascend",
    "nvidia",
    "hygon",
    "thead",
    "mthreads",
    "metax",
)
_ADDMM_BETA_ZERO_VENDORS = ("nvidia", "hygon", "thead", "metax")

if IS_KUNLUNXIN:
    _ADDMM_LAYOUT_BIAS_VENDORS += ("kunlunxin",)
    _ADDMM_BETA_ZERO_VENDORS += ("kunlunxin",)

# Extend this set as vendor implementations gain equivalent layout and bias support.
_addmm_layout_bias_only = pytest.mark.skipif(
    flag_gems.vendor_name not in _ADDMM_LAYOUT_BIAS_VENDORS,
    reason="Issue #5385: AddMM layout and bias coverage is pending on this backend",
)

_addmm_beta_zero_only = pytest.mark.skipif(
    flag_gems.vendor_name not in _ADDMM_BETA_ZERO_VENDORS,
    reason="Issue #5755: this backend does not yet preserve the AddMM beta-zero contract",
)


@pytest.mark.addmm
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("b_column_major", [True, False])
def test_addmm(monkeypatch, M, N, K, scalar, dtype, b_column_major):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Issue #3794: not working")

    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    if b_column_major:
        mat2 = torch.randn((N, K), dtype=dtype, device=flag_gems.device).t()
    else:
        mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    bias1 = torch.randn((N,), dtype=dtype, device=flag_gems.device)
    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)
    ref_bias1 = utils.to_reference(bias1, True)

    alpha = beta = scalar

    ref_out1 = torch.addmm(ref_bias1, ref_mat1, ref_mat2, alpha=alpha, beta=beta)
    res_out1 = flag_gems.addmm(bias1, mat1, mat2, alpha=alpha, beta=beta)

    utils.gems_assert_close(res_out1, ref_out1, dtype, reduce_dim=K)

    bias2 = torch.randn((M, N), dtype=dtype, device=flag_gems.device)
    ref_bias2 = utils.to_reference(bias2, True)

    ref_out2 = torch.addmm(ref_bias2, ref_mat1, ref_mat2, alpha=alpha, beta=beta)
    res_out2 = flag_gems.addmm(bias2, mat1, mat2, alpha=alpha, beta=beta)

    utils.gems_assert_close(res_out2, ref_out2, dtype, reduce_dim=K)


@pytest.mark.addmm
@_addmm_beta_zero_only
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_addmm_beta_zero_ignores_bias(dtype):
    M, N, K = 17, 19, 23
    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    bias = torch.full((N,), float("nan"), dtype=dtype, device=flag_gems.device)

    ref_out = torch.addmm(
        utils.to_reference(bias, True),
        utils.to_reference(mat1, True),
        utils.to_reference(mat2, True),
        beta=0,
    )
    result = flag_gems.addmm(bias, mat1, mat2, beta=0)

    assert torch.isfinite(result).all()
    utils.gems_assert_close(result, ref_out, dtype, reduce_dim=K)


@pytest.mark.addmm_vector_bias
@pytest.mark.addmm
@_addmm_layout_bias_only
@pytest.mark.parametrize("M, N, K", VECTOR_BIAS_MNK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("b_column_major", [True, False])
def test_addmm_vector_bias_shapes(M, N, K, dtype, b_column_major):
    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    if b_column_major:
        mat2 = torch.randn((N, K), dtype=dtype, device=flag_gems.device).t()
    else:
        mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((N,), dtype=dtype, device=flag_gems.device)

    ref_out = torch.addmm(
        utils.to_reference(bias, True),
        utils.to_reference(mat1, True),
        utils.to_reference(mat2, True),
    )
    result = flag_gems.addmm(bias, mat1, mat2)

    utils.gems_assert_close(result, ref_out, dtype, reduce_dim=K)


@pytest.mark.addmm
@_addmm_layout_bias_only
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("b_column_major", [True, False])
def test_addmm_scalar_bias(dtype, b_column_major):
    M, N, K = 17, 29, 31
    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    if b_column_major:
        mat2 = torch.randn((N, K), dtype=dtype, device=flag_gems.device).t()
    else:
        mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((1, 1), dtype=dtype, device=flag_gems.device)

    ref_out = torch.addmm(
        utils.to_reference(bias, True),
        utils.to_reference(mat1, True),
        utils.to_reference(mat2, True),
    )
    result = flag_gems.addmm(bias, mat1, mat2)

    utils.gems_assert_close(result, ref_out, dtype, reduce_dim=K)


@pytest.mark.addmm_vector_bias
@pytest.mark.addmm
@_addmm_layout_bias_only
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("b_column_major", [True, False])
def test_addmm_padded_vector_bias(dtype, b_column_major):
    M, N, K, padding = 64, 128, 64, 7
    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    if b_column_major:
        storage = torch.randn((N, K + padding), dtype=dtype, device=flag_gems.device)
        mat2 = storage[:, :K].t()
    else:
        storage = torch.randn((K, N + padding), dtype=dtype, device=flag_gems.device)
        mat2 = storage[:, :N]
    bias = torch.randn((N,), dtype=dtype, device=flag_gems.device)

    ref_out = torch.addmm(
        utils.to_reference(bias, True),
        utils.to_reference(mat1, True),
        utils.to_reference(mat2, True),
    )
    result = flag_gems.addmm(bias, mat1, mat2)

    utils.gems_assert_close(result, ref_out, dtype, reduce_dim=K)


@pytest.mark.addmm_out
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_addmm_out(M, N, K, scalar, dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Issue #3794: not working")

    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    bias1 = torch.randn((N,), dtype=dtype, device=flag_gems.device)
    out = torch.empty((M, N), dtype=dtype, device=flag_gems.device)
    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)
    ref_bias1 = utils.to_reference(bias1, True)
    ref_out = utils.to_reference(out, True)

    alpha = beta = scalar

    torch.addmm(ref_bias1, ref_mat1, ref_mat2, alpha=alpha, beta=beta, out=ref_out)
    flag_gems.addmm_out(bias1, mat1, mat2, alpha=alpha, beta=beta, out=out)

    utils.gems_assert_close(out, ref_out, dtype, reduce_dim=K)

    bias2 = torch.randn((M, N), dtype=dtype, device=flag_gems.device)
    ref_bias2 = utils.to_reference(bias2, True)

    torch.addmm(ref_bias2, ref_mat1, ref_mat2, alpha=alpha, beta=beta, out=ref_out)
    flag_gems.addmm_out(bias2, mat1, mat2, alpha=alpha, beta=beta, out=out)

    utils.gems_assert_close(out, ref_out, dtype, reduce_dim=K)


@pytest.mark.addmm_out
@_addmm_beta_zero_only
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_addmm_out_beta_zero_ignores_bias(dtype):
    M, N, K = 17, 19, 23
    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    bias = torch.full((N,), float("nan"), dtype=dtype, device=flag_gems.device)
    out = torch.empty((M, N), dtype=dtype, device=flag_gems.device)

    ref_out = torch.addmm(
        utils.to_reference(bias, True),
        utils.to_reference(mat1, True),
        utils.to_reference(mat2, True),
        beta=0,
    )
    flag_gems.addmm_out(bias, mat1, mat2, beta=0, out=out)

    assert torch.isfinite(out).all()
    utils.gems_assert_close(out, ref_out, dtype, reduce_dim=K)


@pytest.mark.addmm_out_vector_bias
@pytest.mark.addmm_out
@_addmm_layout_bias_only
@pytest.mark.parametrize("M, N, K", VECTOR_BIAS_MNK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("b_column_major", [True, False])
def test_addmm_out_vector_bias_shapes(M, N, K, dtype, b_column_major):
    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    if b_column_major:
        mat2 = torch.randn((N, K), dtype=dtype, device=flag_gems.device).t()
    else:
        mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((N,), dtype=dtype, device=flag_gems.device)
    out = torch.empty((M, N), dtype=dtype, device=flag_gems.device)

    ref_out = torch.addmm(
        utils.to_reference(bias, True),
        utils.to_reference(mat1, True),
        utils.to_reference(mat2, True),
    )
    flag_gems.addmm_out(bias, mat1, mat2, out=out)

    utils.gems_assert_close(out, ref_out, dtype, reduce_dim=K)


@pytest.mark.addmm_out
@_addmm_layout_bias_only
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("b_column_major", [True, False])
def test_addmm_out_noncontiguous(dtype, b_column_major):
    M, N, K = 15, 17, 32
    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    if b_column_major:
        mat2 = torch.randn((N, K), dtype=dtype, device=flag_gems.device).t()
    else:
        mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((N,), dtype=dtype, device=flag_gems.device)
    out = torch.empty((N, M), dtype=dtype, device=flag_gems.device).t()

    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)
    ref_bias = utils.to_reference(bias, True)
    ref_out = utils.to_reference(out, True)

    torch.addmm(ref_bias, ref_mat1, ref_mat2, out=ref_out)
    flag_gems.addmm_out(bias, mat1, mat2, out=out)

    utils.gems_assert_close(out, ref_out, dtype, reduce_dim=K)


@pytest.mark.addmm
@_addmm_layout_bias_only
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("bias_shape", [(), (1, 128), (128, 1)])
def test_addmm_broadcast_bias(dtype, bias_shape):
    M, N, K = 128, 128, 128
    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((N, K), dtype=dtype, device=flag_gems.device).t()
    bias = torch.randn(bias_shape, dtype=dtype, device=flag_gems.device)

    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)
    ref_bias = utils.to_reference(bias, True)
    ref_out = torch.addmm(ref_bias, ref_mat1, ref_mat2)
    out = flag_gems.addmm(bias, mat1, mat2)

    utils.gems_assert_close(out, ref_out, dtype, reduce_dim=K)


@pytest.mark.addmm_dtype
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse("2.8"),
    reason="The operator addmm.dtype was added starting from 2.8.0",
)
def test_addmm_dtype_fp32_accum(M, N, K):
    dtype = torch.float16
    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((M, N), dtype=torch.float32, device=flag_gems.device)

    # CPU eager may not implement ``aten.addmm.dtype``; match fp32 accumulation explicitly.
    ref_out = torch.addmm(
        bias.detach().cpu().float(),
        mat1.detach().cpu().float(),
        mat2.detach().cpu().float(),
        beta=1.0,
        alpha=1.0,
    )

    res_out = flag_gems.addmm_dtype(
        bias, mat1, mat2, torch.float32, beta=1.0, alpha=1.0
    )

    if utils.TO_CPU:
        res_out = res_out.to("cpu")
    else:
        ref_out = ref_out.to(flag_gems.device)
    utils.gems_assert_close(res_out, ref_out, torch.float32, reduce_dim=K)


@pytest.mark.addmm_dtype
@_addmm_beta_zero_only
@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse("2.8"),
    reason="The operator addmm.dtype was added starting from 2.8.0",
)
def test_addmm_dtype_beta_zero_ignores_bias():
    M, N, K = 17, 19, 23
    mat1 = torch.randn((M, K), dtype=torch.float16, device=flag_gems.device)
    mat2 = torch.randn((K, N), dtype=torch.float16, device=flag_gems.device)
    bias = torch.full((N,), float("nan"), dtype=torch.float16, device=flag_gems.device)
    ref_out = mat1.detach().cpu().float() @ mat2.detach().cpu().float()

    result = flag_gems.addmm_dtype(bias, mat1, mat2, torch.float32, beta=0.0, alpha=1.0)

    if utils.TO_CPU:
        result = result.to("cpu")
    else:
        ref_out = ref_out.to(flag_gems.device)
    assert torch.isfinite(result).all()
    utils.gems_assert_close(result, ref_out, torch.float32, reduce_dim=K)


@pytest.mark.addmm_dtype_out
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse("2.8"),
    reason="The operator addmm.dtype_out was added starting from 2.8.0",
)
def test_addmm_dtype_out_fp32_accum(M, N, K):
    dtype = torch.float16
    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((M, N), dtype=torch.float32, device=flag_gems.device)
    out = torch.empty((M, N), dtype=torch.float32, device=flag_gems.device)

    ref_out = torch.addmm(
        bias.detach().cpu().float(),
        mat1.detach().cpu().float(),
        mat2.detach().cpu().float(),
        beta=1.0,
        alpha=1.0,
    )

    flag_gems.addmm_dtype_out(
        bias, mat1, mat2, torch.float32, beta=1.0, alpha=1.0, out=out
    )

    if utils.TO_CPU:
        out = out.to("cpu")
    else:
        ref_out = ref_out.to(flag_gems.device)
    utils.gems_assert_close(out, ref_out, torch.float32, reduce_dim=K)


@pytest.mark.addmm_dtype_out
@_addmm_beta_zero_only
@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse("2.8"),
    reason="The operator addmm.dtype_out was added starting from 2.8.0",
)
def test_addmm_dtype_out_beta_zero_ignores_bias():
    M, N, K = 17, 19, 23
    mat1 = torch.randn((M, K), dtype=torch.float16, device=flag_gems.device)
    mat2 = torch.randn((K, N), dtype=torch.float16, device=flag_gems.device)
    bias = torch.full((N,), float("nan"), dtype=torch.float16, device=flag_gems.device)
    out = torch.empty((M, N), dtype=torch.float32, device=flag_gems.device)
    ref_out = mat1.detach().cpu().float() @ mat2.detach().cpu().float()

    flag_gems.addmm_dtype_out(
        bias,
        mat1,
        mat2,
        torch.float32,
        beta=0.0,
        alpha=1.0,
        out=out,
    )

    if utils.TO_CPU:
        out = out.to("cpu")
    else:
        ref_out = ref_out.to(flag_gems.device)
    assert torch.isfinite(out).all()
    utils.gems_assert_close(out, ref_out, torch.float32, reduce_dim=K)


# Ascend large-tensor and coreDim regressions.
@pytest.mark.addmm
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="This test is specific to Ascend NPU coreDim limit",
)
@pytest.mark.parametrize(
    "M,N,K,description",
    [
        # Small baseline
        (512, 512, 512, "Baseline"),
        # Just below threshold: sqrt(65535) ≈ 256, so 256*256 = 65536 (exceeds)
        # Use 255*128 = 32640 (safe), 250*128 = 32000 (safe)
        (200 * 128, 200 * 128, 512, "Below threshold (200*200=40k tiles) ~1.5GB"),
        # Just above threshold: 256*256 = 65536
        (256 * 128, 256 * 128, 512, "Above threshold (256*256=65k tiles) ~4GB"),
        # Larger - use smaller K to control memory
        (300 * 128, 300 * 128, 512, "Large square (300*300=90k tiles) ~6.5GB"),
        # Non-square large matrices - keep total memory low
        (512 * 128, 200 * 128, 512, "Non-square (512*200=102k tiles) ~6.5GB"),
        (200 * 128, 512 * 128, 512, "Non-square (200*512=102k tiles) ~6.5GB"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_addmm_large_matrix(M, N, K, description, dtype):
    """Test addmm with large matrices that trigger coreDim > 65535.

    Memory estimate (float32):
    - M=N=256*128=32768, K=512: ~4.2 GB
    - M=N=300*128=38400, K=512: ~6.5 GB
    - M=512*128=65536, N=200*128=25600, K=512: ~6.5 GB
    """
    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((N,), dtype=dtype, device=flag_gems.device)

    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)
    ref_bias = utils.to_reference(bias, True)

    ref_out = torch.addmm(ref_bias, ref_mat1, ref_mat2)
    res_out = flag_gems.addmm(bias, mat1, mat2)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.addmm
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="This test is specific to Ascend NPU coreDim limit",
)
@pytest.mark.parametrize(
    "M,N,K",
    [
        (256 * 128, 256 * 128, 512),  # ~4 GB
        (300 * 128, 200 * 128, 512),  # ~4.5 GB, non-square
    ],
)
@pytest.mark.parametrize("alpha,beta", [(1.0, 1.0), (2.0, 0.5)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_addmm_large_matrix_alpha_beta(M, N, K, alpha, beta, dtype):
    """Test addmm with large matrices and different alpha/beta values.

    Memory estimate (float32):
    - M=N=32768, K=512: ~4.2 GB
    - M=38400, N=25600, K=512: ~4.5 GB
    """
    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((M, N), dtype=dtype, device=flag_gems.device)

    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)
    ref_bias = utils.to_reference(bias, True)

    ref_out = torch.addmm(ref_bias, ref_mat1, ref_mat2, alpha=alpha, beta=beta)
    res_out = flag_gems.addmm(bias, mat1, mat2, alpha=alpha, beta=beta)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.addmm
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="This test is specific to Ascend NPU coreDim limit",
)
@pytest.mark.parametrize(
    "N",
    [
        128,  # Small - uses _addmm_small_* path
        256,  # Medium
        512,  # Large
        1024,  # Very large
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_addmm_small_square_large(N, dtype):
    """Test _addmm_small_* path with large square matrices.

    This tests the special fast path for M==N==K with specific sizes.
    When N > 512, _addmm_small_add and _addmm_small_finish can have
    coreDim issues with M*N / 8192 exceeding 65535.
    """
    if N not in [128, 256, 512, 1024] or N < 128:
        pytest.skip("_addmm_small path only supports specific sizes")

    M = K = N
    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((M, N), dtype=dtype, device=flag_gems.device)

    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)
    ref_bias = utils.to_reference(bias, True)

    # Test with alpha=beta=1
    ref_out = torch.addmm(ref_bias, ref_mat1, ref_mat2, alpha=1.0, beta=1.0)
    res_out = flag_gems.addmm(bias, mat1, mat2, alpha=1.0, beta=1.0)
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)

    # Test with different alpha/beta
    ref_out2 = torch.addmm(ref_bias, ref_mat1, ref_mat2, alpha=2.0, beta=0.5)
    res_out2 = flag_gems.addmm(bias, mat1, mat2, alpha=2.0, beta=0.5)
    utils.gems_assert_close(res_out2, ref_out2, dtype, reduce_dim=K)


@pytest.mark.addmm_out
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="This test is specific to Ascend NPU coreDim limit",
)
@pytest.mark.parametrize(
    "M,N,K",
    [
        (256 * 128, 256 * 128, 512),  # ~4 GB
        (300 * 128, 200 * 128, 512),  # ~4.5 GB
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_addmm_out_large_matrix(M, N, K, dtype):
    """Test addmm_out with large matrices."""
    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((N,), dtype=dtype, device=flag_gems.device)
    out = torch.empty((M, N), dtype=dtype, device=flag_gems.device)

    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)
    ref_bias = utils.to_reference(bias, True)
    ref_out = utils.to_reference(out, True)

    torch.addmm(ref_bias, ref_mat1, ref_mat2, out=ref_out)
    flag_gems.addmm_out(bias, mat1, mat2, out=out)

    utils.gems_assert_close(out, ref_out, dtype, reduce_dim=K)


@pytest.mark.addmm
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="Ascend addmm fallback tuning candidates",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("b_column_major", [False, True])
@pytest.mark.parametrize("vector_bias", [False, True])
def test_addmm_fallback_autotuning(monkeypatch, dtype, b_column_major, vector_bias):
    # A cached best config or timing can hide an invalid autotuning candidate.
    # Exercise fresh tuning, including the NT layout that triggered an aicore
    # exception with the shared MM tuning space. The normal tuner still handles
    # candidates that the compiler rejects before launching them.
    M, N, K = 15, 160, 1024
    torch.manual_seed(0)
    a = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    b = torch.randn(
        (N, K) if b_column_major else (K, N), dtype=dtype, device=flag_gems.device
    )
    if b_column_major:
        b = b.t()
    bias = torch.randn(
        (N,) if vector_bias else (M, N), dtype=dtype, device=flag_gems.device
    )
    expected = torch.addmm(
        utils.to_reference(bias, True),
        utils.to_reference(a, True),
        utils.to_reference(b, True),
        alpha=0.5,
        beta=2.0,
    )
    op_globals = flag_gems.addmm.__globals__
    tuner = op_globals["addmm_kernel"]
    while not isinstance(tuner, triton.runtime.Autotuner):
        tuner = tuner.fn
    # Bypass LibEntry's dispatch cache and isolate both tuning caches without
    # clearing or writing the user's persistent tuning results.
    libentry_module = importlib.import_module("flag_gems.utils.libentry")
    monkeypatch.setitem(op_globals, "addmm_kernel", tuner)
    monkeypatch.setattr(tuner, "cache", {})
    monkeypatch.setattr(libentry_module.libcache, "get_benchmark", lambda *args: {})
    actual = flag_gems.addmm(bias, a, b, alpha=0.5, beta=2.0)
    utils.gems_assert_close(actual, expected, dtype, reduce_dim=K)
    assert tuner.benchmark_success_count > 0
    assert tuner.benchmark_cache_hit_count == 0
