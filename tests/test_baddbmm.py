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

import pytest
import torch

import flag_gems

from .accuracy_utils import SCALARS, gems_assert_close, to_reference
from .conftest import QUICK_MODE
from .gemm_utils import GEMM_DTYPES, accuracy_dtype_cases, accuracy_shapes

BMNK_SHAPES = accuracy_shapes("baddbmm", QUICK_MODE)
MNK_SHAPES = list(dict.fromkeys(shape[1:] for shape in BMNK_SHAPES))
BACKWARD_SHAPES = [(2, 16, 4096), (32, 64, 2048)]

GNK_SHAPES = [(16, 512, 2048), (16, 2560, 2048), (64, 2048, 128)]


FP8_MNK_SHAPES = [
    (128, 256, 512),
    (64, 128, 128),
    (256, 256, 256),
    (83, 7748, 3884),
    (84, 7168, 3884),
]


@pytest.mark.baddbmm
@pytest.mark.parametrize(
    "dtype, scalar, batch, M, N, K, b_column_major",
    accuracy_dtype_cases("baddbmm", QUICK_MODE, scalars=SCALARS),
)
def test_baddbmm(batch, M, N, K, scalar, dtype, b_column_major):
    mat1 = torch.randn((batch, M, K), dtype=dtype, device=flag_gems.device)
    mat2 = (
        torch.randn((batch, N, K), dtype=dtype, device=flag_gems.device).transpose(1, 2)
        if b_column_major
        else torch.randn((batch, K, N), dtype=dtype, device=flag_gems.device)
    )
    bias = torch.randn((N,), dtype=dtype, device=flag_gems.device)
    ref_mat1 = to_reference(mat1, True)
    ref_mat2 = to_reference(mat2, True)
    ref_bias = to_reference(bias, True)

    alpha = beta = scalar

    ref_out = torch.baddbmm(ref_bias, ref_mat1, ref_mat2, alpha=alpha, beta=beta)
    res_out = flag_gems.baddbmm(bias, mat1, mat2, alpha=alpha, beta=beta)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.baddbmm_out
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("dtype", GEMM_DTYPES)
def test_baddbmm_out(M, N, K, scalar, dtype):
    batch = 4
    mat1 = torch.randn((batch, M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((batch, K, N), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((N,), dtype=dtype, device=flag_gems.device)
    out = torch.empty((batch, M, N), dtype=dtype, device=flag_gems.device)
    ref_mat1 = to_reference(mat1, True)
    ref_mat2 = to_reference(mat2, True)
    ref_bias = to_reference(bias, True)
    ref_out = to_reference(out, True)

    alpha = beta = scalar

    torch.baddbmm(ref_bias, ref_mat1, ref_mat2, alpha=alpha, beta=beta, out=ref_out)
    flag_gems.baddbmm_out(bias, mat1, mat2, alpha=alpha, beta=beta, out=out)

    gems_assert_close(out, ref_out, dtype, reduce_dim=K)


@pytest.mark.baddbmm
@pytest.mark.baddbmm_out
@pytest.mark.skipif(
    flag_gems.vendor_name != "mthreads", reason="MThreads small-M baddbmm regression"
)
@pytest.mark.parametrize("scalar", [1.0, *SCALARS])
@pytest.mark.parametrize("dtype", GEMM_DTYPES)
def test_baddbmm_mthreads_small_m(scalar, dtype):
    batch, M, N, K = 4, 15, 160, 1024
    bias = torch.randn((N,), dtype=dtype, device=flag_gems.device)
    mat1 = torch.randn((batch, M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((batch, K, N), dtype=dtype, device=flag_gems.device)
    expected = torch.baddbmm(
        to_reference(bias, True),
        to_reference(mat1, True),
        to_reference(mat2, True),
        alpha=scalar,
        beta=scalar,
    )

    actual = flag_gems.baddbmm(bias, mat1, mat2, alpha=scalar, beta=scalar)
    gems_assert_close(actual, expected, dtype, reduce_dim=K)

    out = torch.empty_like(actual)
    result = flag_gems.baddbmm_out(bias, mat1, mat2, alpha=scalar, beta=scalar, out=out)
    assert result is out
    gems_assert_close(out, expected, dtype, reduce_dim=K)


@pytest.mark.baddbmm
@pytest.mark.baddbmm_out
@pytest.mark.skipif(
    flag_gems.vendor_name != "mthreads", reason="MThreads row-wise GEMV paths"
)
@pytest.mark.parametrize(
    "M,N,K,b_column_major",
    [
        (1, 64, 4096, False),
        (2, 16, 4096, False),
        (2, 16, 4096, True),
        (64, 4, 16384, False),
        (64, 4, 16384, True),
    ],
)
@pytest.mark.parametrize("dtype", GEMM_DTYPES)
def test_baddbmm_mthreads_full_bias_gemv(M, N, K, b_column_major, dtype):
    batch = 4
    a = torch.randn((batch, M, K), dtype=dtype, device=flag_gems.device)
    b = (
        torch.randn((batch, N, K), dtype=dtype, device=flag_gems.device).transpose(1, 2)
        if b_column_major
        else torch.randn((batch, K, N), dtype=dtype, device=flag_gems.device)
    )
    bias = torch.randn((batch, M, N), dtype=dtype, device=flag_gems.device)
    reference = torch.baddbmm(
        to_reference(bias, True), to_reference(a, True), to_reference(b, True)
    )
    result = flag_gems.baddbmm(bias, a, b)
    gems_assert_close(result, reference, dtype, reduce_dim=K)
    out = torch.empty_like(bias)
    assert flag_gems.baddbmm_out(bias, a, b, out=out) is out
    gems_assert_close(out, reference, dtype, reduce_dim=K)


@pytest.mark.baddbmm
@pytest.mark.parametrize("M, N, K", BACKWARD_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("dtype", GEMM_DTYPES)
def test_baddbmm_backward(M, N, K, scalar, dtype):
    batch = 2
    mat1 = torch.randn(
        (batch, M, K), dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    mat2 = torch.randn(
        (batch, K, N), dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    bias = torch.randn(
        (batch, M, N), dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    ref_mat1 = to_reference(mat1, True)
    ref_mat2 = to_reference(mat2, True)
    ref_bias = to_reference(bias, True)
    alpha = beta = scalar

    ref_out = torch.baddbmm(ref_bias, ref_mat1, ref_mat2, alpha=alpha, beta=beta)
    res_out = flag_gems.baddbmm(bias, mat1, mat2, alpha=alpha, beta=beta)

    _assert_baddbmm_gradients(
        res_out,
        ref_out,
        (bias, mat1, mat2),
        (ref_bias, ref_mat1, ref_mat2),
        bias_reduce_dim=K,
    )


@pytest.mark.baddbmm
@pytest.mark.parametrize("dtype", GEMM_DTYPES)
def test_baddbmm_backward_broadcast(dtype):
    batch, M, N, K = 2, 4, 16, 128
    alpha, beta = 0.25, 0.5
    inputs = tuple(
        torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
        for shape in ((N,), (batch, M, K), (batch, K, N))
    )
    ref_inputs = tuple(to_reference(x, True) for x in inputs)
    ref_out = torch.baddbmm(*ref_inputs, alpha=alpha, beta=beta)
    res_out = flag_gems.baddbmm(*inputs, alpha=alpha, beta=beta)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)
    _assert_baddbmm_gradients(
        res_out, ref_out, inputs, ref_inputs, bias_reduce_dim=batch * M
    )


def _assert_baddbmm_gradients(res_out, ref_out, inputs, ref_inputs, bias_reduce_dim):
    out_grad = torch.randn_like(res_out)
    ref_grad = to_reference(out_grad, True)
    _, M, N = res_out.shape
    ref_in_bias, ref_in_grad1, ref_in_grad2 = torch.autograd.grad(
        ref_out, ref_inputs, ref_grad
    )
    res_in_bias, res_in_grad1, res_in_grad2 = torch.autograd.grad(
        res_out, inputs, out_grad
    )

    dtype = res_out.dtype
    gems_assert_close(res_in_bias, ref_in_bias, dtype, reduce_dim=bias_reduce_dim)
    gems_assert_close(res_in_grad1, ref_in_grad1, dtype, reduce_dim=N)
    gems_assert_close(res_in_grad2, ref_in_grad2, dtype, reduce_dim=M)
