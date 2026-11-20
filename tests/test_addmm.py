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
from .gemm_utils import GEMM_DTYPES, accuracy_dtype_cases, accuracy_shapes

IS_KUNLUNXIN = flag_gems.vendor_name == "kunlunxin"

MNK_SHAPES = accuracy_shapes("addmm", QUICK_MODE)
VECTOR_BIAS_MNK_SHAPES = (
    [(64, 256, 184)]
    if QUICK_MODE
    else [
        (64, 256, 184),
        (128, 512, 512),
        (256, 256, 1024),
        (512, 1024, 1024),
        (1024, 512, 1536),
    ]
)


_ADDMM_LAYOUT_BIAS_VENDORS = (
    "ascend",
    "nvidia",
    "hygon",
    "thead",
    "mthreads",
    "metax",
)
_ADDMM_BETA_ZERO_VENDORS = ("ascend", "nvidia", "hygon", "thead", "metax")

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
@pytest.mark.parametrize(
    "dtype, scalar, M, N, K, b_column_major",
    accuracy_dtype_cases("addmm", QUICK_MODE, scalars=utils.SCALARS),
)
def test_addmm(M, N, K, scalar, dtype, b_column_major):
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
@pytest.mark.parametrize("dtype", GEMM_DTYPES)
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
@pytest.mark.parametrize("dtype", GEMM_DTYPES)
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
@pytest.mark.parametrize("dtype", GEMM_DTYPES)
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
@pytest.mark.parametrize("dtype", GEMM_DTYPES)
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
@pytest.mark.parametrize("dtype", GEMM_DTYPES)
def test_addmm_out(M, N, K, scalar, dtype):
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
@pytest.mark.parametrize("dtype", GEMM_DTYPES)
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
@pytest.mark.parametrize("dtype", GEMM_DTYPES)
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
@pytest.mark.parametrize("dtype", GEMM_DTYPES)
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
@pytest.mark.parametrize("dtype", GEMM_DTYPES)
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
@pytest.fixture
def release_addmm_tuning_tensors():
    # LibTuner retains its last inputs/output for offline benchmark replay.
    # These tests do not replay them: release the captured tensors between
    # cases so the previous multi-GiB output does not overlap the next case.
    tuner = flag_gems.addmm.__globals__["addmm_kernel"]
    while not isinstance(tuner, triton.runtime.Autotuner):
        tuner = tuner.fn

    def release():
        tuner._last_benchmark_args = None
        tuner._last_benchmark_meta = None

    release()
    try:
        yield
    finally:
        release()


def _assert_large_addmm_close(out, bias, mat1, mat2, *, alpha=1, beta=1):
    # Launch Gems on the full matrix, but compute and compare the reference in
    # row chunks. A second full output (and assert_close's temporaries) can
    # exhaust a shared NPU even when the operator itself fits comfortably.
    # Visit every output element and retain the usual reference dtype/tolerance.
    M, N = out.shape
    rows_per_chunk = max(1, min(256, (1 << 22) // N))
    ref_mat2 = utils.to_reference(mat2, True)
    bias = bias.broadcast_to(out.shape)
    for start in range(0, M, rows_per_chunk):
        end = min(start + rows_per_chunk, M)
        ref_mat1 = utils.to_reference(mat1[start:end], True)
        ref_bias = utils.to_reference(bias[start:end], True)
        ref_out = torch.addmm(ref_bias, ref_mat1, ref_mat2, alpha=alpha, beta=beta)
        utils.gems_assert_close(
            out[start:end], ref_out, out.dtype, reduce_dim=mat1.shape[1]
        )
        del ref_mat1, ref_bias, ref_out


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
        # Tile counts below assume BLOCK_M = BLOCK_N = 128.
        (200 * 128, 200 * 128, 512, "Below threshold (200*200=40k tiles)"),
        # Just above threshold: 256*256 = 65536
        (256 * 128, 256 * 128, 512, "Above threshold (256*256=65k tiles)"),
        # Larger - use smaller K to control memory
        (300 * 128, 300 * 128, 512, "Large square (300*300=90k tiles)"),
        # Non-square large matrices - keep total memory low
        (512 * 128, 200 * 128, 512, "Non-square (512*200=102k tiles)"),
        (200 * 128, 512 * 128, 512, "Non-square (200*512=102k tiles)"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.usefixtures("release_addmm_tuning_tensors")
def test_addmm_large_matrix(M, N, K, description, dtype):
    """Test addmm with large matrices that trigger coreDim > 65535.

    Output alone (float32): 4 GiB for M=N=32768, 5.49 GiB for M=N=38400,
    and 6.25 GiB for M=65536, N=25600. Reference chunks avoid doubling that.
    """
    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((N,), dtype=dtype, device=flag_gems.device)

    res_out = flag_gems.addmm(bias, mat1, mat2)
    _assert_large_addmm_close(res_out, bias, mat1, mat2)


@pytest.mark.addmm
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="This test is specific to Ascend NPU coreDim limit",
)
@pytest.mark.parametrize(
    "M,N,K",
    [
        (256 * 128, 256 * 128, 512),
        (300 * 128, 200 * 128, 512),
    ],
)
@pytest.mark.parametrize("alpha,beta", [(1.0, 1.0), (2.0, 0.5)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.usefixtures("release_addmm_tuning_tensors")
def test_addmm_large_matrix_alpha_beta(M, N, K, alpha, beta, dtype):
    """Test addmm with large matrices and different alpha/beta values.

    The float32 matrix bias and output require up to 8 GiB together.
    Compute the reference in chunks to leave room for kernel workspace.
    """
    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((M, N), dtype=dtype, device=flag_gems.device)

    res_out = flag_gems.addmm(bias, mat1, mat2, alpha=alpha, beta=beta)
    _assert_large_addmm_close(res_out, bias, mat1, mat2, alpha=alpha, beta=beta)


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
        (256 * 128, 256 * 128, 512),
        (300 * 128, 200 * 128, 512),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.usefixtures("release_addmm_tuning_tensors")
def test_addmm_out_large_matrix(M, N, K, dtype):
    """Test addmm_out with large matrices."""
    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((N,), dtype=dtype, device=flag_gems.device)
    out = torch.empty((M, N), dtype=dtype, device=flag_gems.device)

    result = flag_gems.addmm_out(bias, mat1, mat2, out=out)
    assert result is out
    _assert_large_addmm_close(out, bias, mat1, mat2)


@pytest.mark.addmm
@pytest.mark.addmm_out
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="Ascend addmm workspace and grid-stride regression",
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("use_out", [False, True])
def test_addmm_bounded_workspace(monkeypatch, dtype, use_out):
    # A small limit exercises repeated grid-stride iterations and masked tails
    # without requiring a multi-GiB allocation just to test the launch policy.
    M, N, K = 513, 769, 67
    op_globals = flag_gems.addmm.__globals__
    kernel = op_globals["addmm_kernel"]
    program_limit = 7
    grids = []

    class CheckedKernel:
        def __getitem__(self, grid):
            def checked_grid(meta):
                launch_grid = grid(meta)
                assert 0 < launch_grid[0] * launch_grid[1] <= program_limit
                tiles = triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(
                    N, meta["BLOCK_N"]
                )
                assert tiles > launch_grid[0]
                grids.append(launch_grid)
                return launch_grid

            return kernel[checked_grid]

    monkeypatch.setitem(op_globals, "_MAX_ADDMM_PROGRAMS", program_limit)
    monkeypatch.setitem(op_globals, "addmm_kernel", CheckedKernel())
    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    bias = torch.randn(
        (M, N) if use_out else (N,), dtype=dtype, device=flag_gems.device
    )
    if use_out:
        out = torch.empty((M, N), dtype=dtype, device=flag_gems.device)
        result = flag_gems.addmm_out(bias, mat1, mat2, alpha=2, beta=0.5, out=out)
        assert result is out
    else:
        out = flag_gems.addmm(bias, mat1, mat2, alpha=2, beta=0.5)
    assert grids
    _assert_large_addmm_close(out, bias, mat1, mat2, alpha=2, beta=0.5)


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


@pytest.mark.addmm
@pytest.mark.addmm_out
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend addmm")
@pytest.mark.parametrize("N", [128, 512])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_addmm_ascend_beta_cache_transitions(N, dtype):
    # Cover both the small-square path and the larger fused tiles that used to
    # overflow UB. Reusing shapes must not reuse the wrong bias/beta variant.
    mat1 = torch.randn((N, N), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((N, N), dtype=dtype, device=flag_gems.device)
    out = torch.empty_like(mat1)
    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)
    for bias_shape in [(N, N), (N,), ()]:
        finite_bias = torch.randn(bias_shape, dtype=dtype, device=flag_gems.device)
        for beta in [1.0, 0.0, -0.5, 0.0, 1.0]:
            bias = (
                torch.full_like(finite_bias, float("nan")) if beta == 0 else finite_bias
            )
            expected = torch.addmm(
                utils.to_reference(bias, True), ref_mat1, ref_mat2, alpha=0.5, beta=beta
            )
            actual = flag_gems.addmm(bias, mat1, mat2, alpha=0.5, beta=beta)
            utils.gems_assert_close(actual, expected, dtype, reduce_dim=N)
            result = flag_gems.addmm_out(
                bias, mat1, mat2, alpha=0.5, beta=beta, out=out
            )
            assert result is out
            utils.gems_assert_close(out, expected, dtype, reduce_dim=N)


@pytest.mark.addmm
@pytest.mark.addmm_out
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend addmm")
@pytest.mark.parametrize("M,N", [(0, 32), (16, 0), (0, 0)])
@pytest.mark.parametrize("use_out", [False, True])
def test_addmm_ascend_review_empty(monkeypatch, M, N, use_out):
    ops = flag_gems.addmm.__globals__

    class NoLaunch:
        def __getitem__(self, grid):
            pytest.fail(f"empty addmm launched grid {grid}")

    monkeypatch.setitem(ops, "addmm_kernel", NoLaunch())
    a = torch.empty((M, 128), device=flag_gems.device)
    b = torch.empty((128, N), device=a.device)
    bias = torch.empty((N,), device=a.device)
    if use_out:
        out = torch.empty(0, device=a.device)
        result = flag_gems.addmm_out(bias, a, b, out=out)
        assert result is out
    else:
        result = flag_gems.addmm(bias, a, b)
    torch.npu.synchronize()
    assert result.shape == (M, N) and result.numel() == 0


@pytest.mark.addmm_out
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend addmm")
@pytest.mark.parametrize(
    "layout", ["empty", "wrong_shape", "strided", "bias_alias", "mat_alias"]
)
def test_addmm_ascend_review_out_resize(layout):
    a = torch.randn((33, 128), device=flag_gems.device)
    b = torch.randn((128, 17), device=a.device)
    bias = torch.randn((17,), device=a.device)
    expected = torch.addmm(bias.cpu(), a.cpu(), b.cpu(), alpha=0.5, beta=2)
    if layout == "strided":
        out = torch.empty((17, 33), device=a.device).t()
    elif layout == "bias_alias":
        out = bias
    elif layout == "mat_alias":
        out = a
    else:
        out = torch.empty((0,) if layout == "empty" else (1, 3), device=a.device)
    old_stride = out.stride()
    assert flag_gems.addmm_out(bias, a, b, out=out, alpha=0.5, beta=2) is out
    torch.npu.synchronize()
    if layout == "strided":
        assert out.stride() == old_stride
    utils.gems_assert_close(out, expected, out.dtype, reduce_dim=128)


@pytest.mark.addmm
@pytest.mark.addmm_out
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend addmm")
@pytest.mark.parametrize("shape", [(3, 5, 16), (128, 128, 128)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("use_out", [False, True])
def test_addmm_ascend_review_beta_zero(shape, dtype, use_out):
    M, N, K = shape
    a = torch.ones((M, K), dtype=dtype, device=flag_gems.device)
    b = torch.ones((K, N), dtype=dtype, device=a.device)
    bias = torch.full((M, N), float("nan"), dtype=dtype, device=a.device)
    bias[:, ::2] = float("inf")
    if use_out:
        result = torch.empty(0, dtype=dtype, device=a.device)
        assert flag_gems.addmm_out(bias, a, b, beta=0, alpha=0.5, out=result) is result
    else:
        result = flag_gems.addmm(bias, a, b, beta=0, alpha=0.5)
    torch.npu.synchronize()
    expected = torch.addmm(
        bias.cpu().float(), a.cpu().float(), b.cpu().float(), beta=0, alpha=0.5
    ).to(dtype)
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)


@pytest.mark.addmm_out
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend addmm")
@pytest.mark.parametrize("invalid", ["dtype", "device"])
def test_addmm_ascend_review_out_validation(invalid):
    a = torch.ones((3, 128), device=flag_gems.device)
    b = torch.ones((128, 5), device=a.device)
    bias = torch.ones(5, device=a.device)
    out = torch.empty(
        0,
        device="cpu" if invalid == "device" else a.device,
        dtype=torch.float16 if invalid == "dtype" else a.dtype,
    )
    with pytest.raises(RuntimeError, match=invalid):
        flag_gems.addmm_out(bias, a, b, out=out)
    assert out.shape == (0,)


# These compact aligned, rectangular and tail cases target all three fused regions.
ADDMM_BACKWARD_SHAPES = [(17, 29, 31), (64, 64, 64), (33, 65, 129)]


def _addmm_backward_reference(tensor):
    # The fused kernel computes IEEE FP32 even when cuBLAS enables TF32.
    # Use the GEMM suite's high-precision reference for float32 inputs.
    return utils.to_reference(tensor, upcast=tensor.dtype == torch.float32)


@pytest.mark.addmm_backward
@pytest.mark.parametrize("M, N, K", ADDMM_BACKWARD_SHAPES)
@pytest.mark.parametrize("alpha, beta", [(1.0, 1.0), (0.75, -0.5)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_addmm_backward(M, N, K, alpha, beta, dtype):
    bias = torch.randn((M, N), dtype=dtype, device=flag_gems.device)
    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    grad = torch.randn((M, N), dtype=dtype, device=flag_gems.device)
    ref_inputs = [
        _addmm_backward_reference(value.detach()).requires_grad_()
        for value in (bias, mat1, mat2)
    ]
    res_inputs = [value.detach().requires_grad_() for value in (bias, mat1, mat2)]

    ref_grads = torch.autograd.grad(
        torch.addmm(*ref_inputs, alpha=alpha, beta=beta),
        ref_inputs,
        _addmm_backward_reference(grad),
    )
    res_grads = torch.autograd.grad(
        flag_gems.addmm(*res_inputs, alpha=alpha, beta=beta), res_inputs, grad
    )
    direct_grads = flag_gems.addmm_backward(grad, mat1, mat2, alpha=alpha, beta=beta)

    reduction_dims = (1, N, M)
    for result, reference, reduce_dim in zip(res_grads, ref_grads, reduction_dims):
        utils.gems_assert_close(
            result, reference, dtype, reduce_dim=reduce_dim, atol=3e-4
        )
    for result, reference, reduce_dim in zip(direct_grads, ref_grads, reduction_dims):
        utils.gems_assert_close(
            result, reference, dtype, reduce_dim=reduce_dim, atol=3e-4
        )


@pytest.mark.addmm_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_addmm_backward_registered_dispatch(dtype):
    M, N, K = 33, 47, 65
    inputs = [
        torch.randn((M, N), dtype=dtype, device=flag_gems.device, requires_grad=True),
        torch.randn((M, K), dtype=dtype, device=flag_gems.device, requires_grad=True),
        torch.randn((K, N), dtype=dtype, device=flag_gems.device, requires_grad=True),
    ]
    grad = torch.randn((M, N), dtype=dtype, device=flag_gems.device)
    ref_inputs = [
        _addmm_backward_reference(value.detach()).requires_grad_() for value in inputs
    ]
    ref_grads = torch.autograd.grad(
        torch.addmm(*ref_inputs, alpha=0.75, beta=-0.5),
        ref_inputs,
        _addmm_backward_reference(grad),
    )

    lib = torch.library.Library("aten", "IMPL")
    previous_registrar = getattr(flag_gems, "current_work_registrar", None)
    try:
        flag_gems.only_enable(lib=lib, include=["addmm"])
        res_out = torch.addmm(*inputs, alpha=0.75, beta=-0.5)
        assert "AddmmFunction" in type(res_out.grad_fn).__name__
        res_grads = torch.autograd.grad(res_out, inputs, grad)
    finally:
        lib._destroy()
        flag_gems.current_work_registrar = previous_registrar

    for result, reference, reduce_dim in zip(res_grads, ref_grads, (1, N, M)):
        utils.gems_assert_close(
            result, reference, dtype, reduce_dim=reduce_dim, atol=3e-4
        )


@pytest.mark.addmm_backward
@pytest.mark.parametrize("bias_kind", ["vector", "scalar"])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_addmm_backward_broadcast_bias(bias_kind, dtype):
    M, N, K = 17, 29, 31
    bias_shape = (N,) if bias_kind == "vector" else (1, 1)
    bias = torch.randn(bias_shape, dtype=dtype, device=flag_gems.device)
    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    grad = torch.randn((M, N), dtype=dtype, device=flag_gems.device)
    ref_inputs = [
        _addmm_backward_reference(value.detach()).requires_grad_()
        for value in (bias, mat1, mat2)
    ]
    res_inputs = [value.detach().requires_grad_() for value in (bias, mat1, mat2)]

    ref_grads = torch.autograd.grad(
        torch.addmm(*ref_inputs, alpha=0.75, beta=-0.5),
        ref_inputs,
        _addmm_backward_reference(grad),
    )
    res_grads = torch.autograd.grad(
        flag_gems.addmm(*res_inputs, alpha=0.75, beta=-0.5), res_inputs, grad
    )

    bias_reduce_dim = M if bias_kind == "vector" else M * N
    for result, reference, reduce_dim in zip(
        res_grads, ref_grads, (bias_reduce_dim, N, M)
    ):
        utils.gems_assert_close(
            result, reference, dtype, reduce_dim=reduce_dim, atol=3e-4
        )


@pytest.mark.addmm_backward
@pytest.mark.parametrize("required_input", [0, 1, 2])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_addmm_backward_single_input(required_input, dtype):
    M, N, K = 17, 19, 23
    inputs = [
        torch.randn((M, N), dtype=dtype, device=flag_gems.device),
        torch.randn((M, K), dtype=dtype, device=flag_gems.device),
        torch.randn((K, N), dtype=dtype, device=flag_gems.device),
    ]
    inputs[required_input].requires_grad_()
    grad = torch.randn((M, N), dtype=dtype, device=flag_gems.device)
    ref_inputs = [_addmm_backward_reference(value.detach()) for value in inputs]
    ref_inputs[required_input].requires_grad_()

    ref_grad = torch.autograd.grad(
        torch.addmm(*ref_inputs, alpha=0.75, beta=-0.5),
        ref_inputs[required_input],
        _addmm_backward_reference(grad),
    )[0]
    res_grad = torch.autograd.grad(
        flag_gems.addmm(*inputs, alpha=0.75, beta=-0.5),
        inputs[required_input],
        grad,
    )[0]

    reduce_dim = (1, N, M)[required_input]
    utils.gems_assert_close(res_grad, ref_grad, dtype, reduce_dim=reduce_dim, atol=3e-4)


@pytest.mark.addmm_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_addmm_backward_noncontiguous(dtype):
    M, N, K = 33, 47, 65
    bias = torch.randn((N, M), dtype=dtype, device=flag_gems.device).t()
    mat1 = torch.randn((K, M), dtype=dtype, device=flag_gems.device).t()
    mat2 = torch.randn((N, K), dtype=dtype, device=flag_gems.device).t()
    grad = torch.randn((N, M), dtype=dtype, device=flag_gems.device).t()
    inputs = [value.requires_grad_() for value in (bias, mat1, mat2)]
    ref_inputs = [
        _addmm_backward_reference(value.detach()).requires_grad_() for value in inputs
    ]

    ref_grads = torch.autograd.grad(
        torch.addmm(*ref_inputs, alpha=0.75, beta=-0.5),
        ref_inputs,
        _addmm_backward_reference(grad),
    )
    res_grads = torch.autograd.grad(
        flag_gems.addmm(*inputs, alpha=0.75, beta=-0.5), inputs, grad
    )

    for result, reference, reduce_dim in zip(res_grads, ref_grads, (1, N, M)):
        utils.gems_assert_close(
            result, reference, dtype, reduce_dim=reduce_dim, atol=3e-4
        )


@pytest.mark.addmm_backward
def test_addmm_backward_fused_kernel_count():
    """Bias and both matrix gradients are emitted by one Triton launch."""
    from torch.profiler import ProfilerActivity, profile

    M = N = K = 64
    dtype = torch.float16
    inputs = [
        torch.randn((M, N), dtype=dtype, device=flag_gems.device, requires_grad=True),
        torch.randn((M, K), dtype=dtype, device=flag_gems.device, requires_grad=True),
        torch.randn((K, N), dtype=dtype, device=flag_gems.device, requires_grad=True),
    ]
    grad = torch.randn((M, N), dtype=dtype, device=flag_gems.device)
    out = flag_gems.addmm(*inputs, alpha=0.75, beta=-0.5)

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        torch.autograd.grad(out, inputs, grad)
        flag_gems.runtime.torch_device_fn.synchronize()

    launches = [
        (event.key, event.count)
        for event in prof.key_averages()
        if event.device_time_total > 0
    ]
    backward = [(name, count) for name, count in launches if "addmm" in name]
    assert len(backward) == 1 and backward[0][1] == 1, launches
    assert sum(count for _, count in launches) == 1, launches
