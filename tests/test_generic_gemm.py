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

from flag_gems.ops.generic_gemm import generic_gemm
from flag_gems.utils.device_info import get_device_capability

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

try:
    from transformer_engine.pytorch.cpp_extensions.gemm import (
        general_gemm as te_general_gemm,
    )

    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False

if QUICK_MODE:
    SHAPES = [
        (1, 64, 64),
    ]
    LAYOUTS = ["NN"]
    FLOAT_DTYPES = [torch.float32]
else:
    SHAPES = [
        (1, 64, 64),
        (4, 128, 256),
        (16, 512, 1024),
        (32, 1024, 2048),
        (128, 2048, 4096),
    ]
    LAYOUTS = ["NN", "TN", "NT"]
    FLOAT_DTYPES = utils.FLOAT_DTYPES

FP8_SUPPORTED = get_device_capability() >= (9, 0)
FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0

if QUICK_MODE:
    FP8_SHAPES = [
        (1, 64, 64),
    ]
else:
    FP8_SHAPES = SHAPES


def _torch_gelu(x):
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    return 0.5 * x * (1.0 + torch.tanh(inner))


def _torch_dgelu(x):
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    tanh_inner = torch.tanh(inner)
    sech2 = 1.0 - tanh_inner * tanh_inner
    inner_grad = 0.7978845608028654 * (1.0 + 0.134145 * x * x)
    return 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * inner_grad


@pytest.mark.generic_gemm
@pytest.mark.parametrize("M, N, K", SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_generic_gemm_matmul(M, N, K, layout, dtype):
    transa = layout[0] == "T"
    transb = layout[1] == "T"
    a_shape = (K, M) if transa else (M, K)
    b_shape = (N, K) if transb else (K, N)

    a = torch.randn(a_shape, dtype=dtype, device="cuda")
    b = torch.randn(b_shape, dtype=dtype, device="cuda")

    ref_a = utils.to_reference(a, upcast=True)
    ref_b = utils.to_reference(b, upcast=True)
    if transa:
        ref_a = ref_a.T
    if transb:
        ref_b = ref_b.T
    ref_out = ref_a @ ref_b

    res_out, bias_grad, pre_gelu, extra = generic_gemm(a, b, layout=layout)

    assert bias_grad is None
    assert pre_gelu is None
    assert extra is None

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


def _make_inputs(M, N, K, dtype):
    a = torch.randn((M, K), dtype=dtype, device="cuda")
    b = torch.randn((K, N), dtype=dtype, device="cuda")
    return a, b


def _ref_matmul(ref_a, ref_b, ref_bias=None):
    out = ref_a @ ref_b
    if ref_bias is not None:
        out = out + ref_bias
    return out


@pytest.mark.generic_gemm
@pytest.mark.parametrize("M, N, K", SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_generic_gemm_bias(M, N, K, dtype):
    a, b = _make_inputs(M, N, K, dtype)
    bias = torch.randn((N,), dtype=dtype, device="cuda")

    ref_a = utils.to_reference(a, upcast=True)
    ref_b = utils.to_reference(b, upcast=True)
    ref_bias = utils.to_reference(bias, upcast=True)
    ref_out = _ref_matmul(ref_a, ref_b, ref_bias)

    res_out, bias_grad, pre_gelu, extra = generic_gemm(a, b, bias=bias)

    assert bias_grad is None
    assert pre_gelu is None
    assert extra is None

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.generic_gemm
@pytest.mark.parametrize("M, N, K", SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_generic_gemm_gelu(M, N, K, dtype):
    a, b = _make_inputs(M, N, K, dtype)

    ref_a = utils.to_reference(a, upcast=True)
    ref_b = utils.to_reference(b, upcast=True)
    ref_pre_gelu = _ref_matmul(ref_a, ref_b)
    ref_out = _torch_gelu(ref_pre_gelu)

    res_out, bias_grad, pre_gelu, extra = generic_gemm(a, b, gelu=True)

    assert bias_grad is None
    assert extra is None
    assert pre_gelu is not None

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)
    utils.gems_assert_close(pre_gelu, ref_pre_gelu, dtype, reduce_dim=K)


@pytest.mark.generic_gemm
@pytest.mark.parametrize("M, N, K", SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_generic_gemm_bias_gelu(M, N, K, dtype):
    a, b = _make_inputs(M, N, K, dtype)
    bias = torch.randn((N,), dtype=dtype, device="cuda")

    ref_a = utils.to_reference(a, upcast=True)
    ref_b = utils.to_reference(b, upcast=True)
    ref_bias = utils.to_reference(bias, upcast=True)
    ref_pre_gelu = _ref_matmul(ref_a, ref_b, ref_bias)
    ref_out = _torch_gelu(ref_pre_gelu)

    res_out, bias_grad, pre_gelu, extra = generic_gemm(a, b, bias=bias, gelu=True)

    assert bias_grad is None
    assert extra is None
    assert pre_gelu is not None

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)
    utils.gems_assert_close(pre_gelu, ref_pre_gelu, dtype, reduce_dim=K)


@pytest.mark.generic_gemm
@pytest.mark.parametrize("M, N, K", SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_generic_gemm_accumulate(M, N, K, dtype):
    a, b = _make_inputs(M, N, K, dtype)
    c_init = torch.randn((M, N), dtype=dtype, device="cuda")

    ref_a = utils.to_reference(a, upcast=True)
    ref_b = utils.to_reference(b, upcast=True)
    ref_c = utils.to_reference(c_init, upcast=True)
    ref_out = _ref_matmul(ref_a, ref_b) + ref_c

    res_out, bias_grad, pre_gelu, extra = generic_gemm(
        a,
        b,
        accumulate=True,
        out=c_init.clone(),
    )

    assert bias_grad is None
    assert pre_gelu is None
    assert extra is None

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.generic_gemm
@pytest.mark.parametrize("M, N, K", SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_generic_gemm_dgelu(M, N, K, dtype):
    a, b = _make_inputs(M, N, K, dtype)
    gelu_in = torch.randn((M, N), dtype=dtype, device="cuda")

    ref_a = utils.to_reference(a, upcast=True)
    ref_b = utils.to_reference(b, upcast=True)
    ref_gelu_in = utils.to_reference(gelu_in, upcast=True)
    ref_matmul = _ref_matmul(ref_a, ref_b)
    ref_out = ref_matmul * _torch_dgelu(ref_gelu_in)

    res_out, bias_grad, pre_gelu, extra = generic_gemm(
        a,
        b,
        gelu=True,
        gelu_in=gelu_in,
        grad=True,
    )

    assert pre_gelu is None
    assert extra is None

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.generic_gemm
@pytest.mark.parametrize("M, N, K", SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_generic_gemm_scalar(M, N, K, scalar, dtype):
    a, b = _make_inputs(M, N, K, dtype)

    ref_a = utils.to_reference(a, upcast=True)
    ref_b = utils.to_reference(b, upcast=True)
    ref_out = scalar * _ref_matmul(ref_a, ref_b)

    res_out, _, _, _ = generic_gemm(a, b, alpha=scalar)
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.generic_gemm
@pytest.mark.parametrize("M, N, K", SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_generic_gemm_alpha_beta(M, N, K, dtype):
    a, b = _make_inputs(M, N, K, dtype)
    c_init = torch.randn((M, N), dtype=dtype, device="cuda")
    scalar = 0.5

    ref_a = utils.to_reference(a, upcast=True)
    ref_b = utils.to_reference(b, upcast=True)
    ref_c = utils.to_reference(c_init, upcast=True)
    ref_out = scalar * _ref_matmul(ref_a, ref_b) + scalar * ref_c

    res_out, _, _, _ = generic_gemm(a, b, alpha=scalar, beta=scalar, out=c_init.clone())
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.generic_gemm
@pytest.mark.parametrize("M, N, K", SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_generic_gemm_out(M, N, K, dtype):
    a, b = _make_inputs(M, N, K, dtype)
    out = torch.empty((M, N), dtype=dtype, device="cuda")

    ref_a = utils.to_reference(a, upcast=True)
    ref_b = utils.to_reference(b, upcast=True)
    ref_out = _ref_matmul(ref_a, ref_b)

    res_out, bias_grad, pre_gelu, extra = generic_gemm(a, b, out=out)

    assert res_out is out
    assert bias_grad is None
    assert pre_gelu is None
    assert extra is None

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.generic_gemm
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_generic_gemm_out_dtype(dtype):
    a, b = _make_inputs(16, 64, 128, dtype)
    out_dtype = torch.float32

    ref_a = utils.to_reference(a, upcast=True)
    ref_b = utils.to_reference(b, upcast=True)
    ref_out = _ref_matmul(ref_a, ref_b)

    res_out, _, _, _ = generic_gemm(a, b, out_dtype=out_dtype)

    assert res_out.dtype == out_dtype
    utils.gems_assert_close(res_out, ref_out, out_dtype, reduce_dim=128)


@pytest.mark.generic_gemm
@pytest.mark.parametrize("M, N, K", SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_generic_gemm_dbias(M, N, K, dtype):
    a, b = _make_inputs(M, N, K, dtype)
    bias = torch.randn((N,), dtype=dtype, device="cuda")

    ref_out = _ref_matmul(
        utils.to_reference(a, upcast=True),
        utils.to_reference(b, upcast=True),
    )
    ref_bias_grad = ref_out.to(dtype).float().sum(dim=0)

    res_out, bias_grad, pre_gelu, extra = generic_gemm(a, b, bias=bias, grad=True)

    assert pre_gelu is None
    assert extra is None
    assert bias_grad is not None

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)
    bias_atol = 0.001 if dtype == torch.bfloat16 else 0.0005
    utils.gems_assert_close(
        bias_grad, ref_bias_grad, torch.float32, reduce_dim=max(N, 1), atol=bias_atol
    )


@pytest.mark.generic_gemm
def test_generic_gemm_validation():
    a = torch.randn((4, 8), device="cuda")
    b = torch.randn((8, 6), device="cuda")

    with pytest.raises(ValueError, match="alpha must be non-zero"):
        generic_gemm(a, b, alpha=0.0)

    c, _, _, _ = generic_gemm(a, b, beta=1.0, accumulate=False)
    assert c.shape == (4, 6)

    out = torch.empty((6, 4), device="cuda").t()
    with pytest.raises(ValueError, match="not contiguous"):
        generic_gemm(a, b, out=out)

    with pytest.raises(ValueError, match="2D inputs"):
        generic_gemm(a.unsqueeze(0), b)

    with pytest.raises(AssertionError, match="gelu_in is required"):
        generic_gemm(a, b, gelu=True, grad=True)


def _per_tensor_quantize(x: torch.Tensor):
    amax = x.abs().max()
    scale = (amax / FP8_MAX).to(torch.float32)
    x_fp8 = (x / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return x_fp8, scale


def _fp8_ref_matmul(ref_a, ref_b, scale_a, scale_b, ref_bias=None):
    a_q = (ref_a / scale_a).clamp(-FP8_MAX, FP8_MAX)
    b_q = (ref_b / scale_b).clamp(-FP8_MAX, FP8_MAX)
    out = a_q @ b_q
    out = out * (scale_a * scale_b)
    if ref_bias is not None:
        out = out + ref_bias
    return out


@pytest.mark.generic_gemm
@pytest.mark.skipif(not FP8_SUPPORTED, reason="FP8 requires SM>=90")
@pytest.mark.parametrize("M, N, K", FP8_SHAPES)
def test_generic_gemm_fp8_matmul(M, N, K):
    a = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((K, N), dtype=torch.bfloat16, device="cuda")

    a_fp8, scale_a = _per_tensor_quantize(a)
    b_fp8, scale_b = _per_tensor_quantize(b)

    ref_a = utils.to_reference(a, upcast=True)
    ref_b = utils.to_reference(b, upcast=True)
    ref_out = _fp8_ref_matmul(
        ref_a,
        ref_b,
        utils.to_reference(scale_a, upcast=True),
        utils.to_reference(scale_b, upcast=True),
    )

    res_out, bias_grad, pre_gelu, extra = generic_gemm(
        a_fp8,
        b_fp8,
        layout="NN",
        scale_a=scale_a,
        scale_b=scale_b,
    )

    assert bias_grad is None
    assert pre_gelu is None
    assert extra is None

    utils.gems_assert_close(res_out, ref_out, torch.bfloat16, reduce_dim=K, atol=0.1)


@pytest.mark.generic_gemm
@pytest.mark.skipif(not FP8_SUPPORTED, reason="FP8 requires SM>=90")
@pytest.mark.parametrize("M, N, K", FP8_SHAPES)
def test_generic_gemm_fp8_bias(M, N, K):
    a = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((K, N), dtype=torch.bfloat16, device="cuda")
    bias = torch.randn((N,), dtype=torch.bfloat16, device="cuda")

    a_fp8, scale_a = _per_tensor_quantize(a)
    b_fp8, scale_b = _per_tensor_quantize(b)

    ref_a = utils.to_reference(a, upcast=True)
    ref_b = utils.to_reference(b, upcast=True)
    ref_bias = utils.to_reference(bias, upcast=True)
    ref_out = _fp8_ref_matmul(
        ref_a,
        ref_b,
        utils.to_reference(scale_a, upcast=True),
        utils.to_reference(scale_b, upcast=True),
        ref_bias,
    )

    res_out, bias_grad, pre_gelu, extra = generic_gemm(
        a_fp8,
        b_fp8,
        layout="NN",
        scale_a=scale_a,
        scale_b=scale_b,
        bias=bias,
    )

    assert bias_grad is None
    assert pre_gelu is None
    assert extra is None

    utils.gems_assert_close(res_out, ref_out, torch.bfloat16, reduce_dim=K, atol=0.1)


@pytest.mark.generic_gemm
@pytest.mark.skipif(not FP8_SUPPORTED, reason="FP8 requires SM>=90")
@pytest.mark.parametrize("M, N, K", FP8_SHAPES)
def test_generic_gemm_fp8_gelu(M, N, K):
    a = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((K, N), dtype=torch.bfloat16, device="cuda")

    a_fp8, scale_a = _per_tensor_quantize(a)
    b_fp8, scale_b = _per_tensor_quantize(b)

    ref_a = utils.to_reference(a, upcast=True)
    ref_b = utils.to_reference(b, upcast=True)
    ref_pre_gelu = _fp8_ref_matmul(
        ref_a,
        ref_b,
        utils.to_reference(scale_a, upcast=True),
        utils.to_reference(scale_b, upcast=True),
    )
    ref_out = _torch_gelu(ref_pre_gelu)

    res_out, bias_grad, pre_gelu, extra = generic_gemm(
        a_fp8,
        b_fp8,
        layout="NN",
        scale_a=scale_a,
        scale_b=scale_b,
        gelu=True,
    )

    assert bias_grad is None
    assert extra is None
    assert pre_gelu is not None

    utils.gems_assert_close(res_out, ref_out, torch.bfloat16, reduce_dim=K, atol=0.1)
    utils.gems_assert_close(
        pre_gelu, ref_pre_gelu, torch.bfloat16, reduce_dim=K, atol=0.1
    )


@pytest.mark.generic_gemm
@pytest.mark.skipif(not FP8_SUPPORTED, reason="FP8 requires SM>=90")
@pytest.mark.parametrize("M, N, K", FP8_SHAPES)
def test_generic_gemm_fp8_output(M, N, K):
    a = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((K, N), dtype=torch.bfloat16, device="cuda")

    a_fp8, scale_a = _per_tensor_quantize(a)
    b_fp8, scale_b = _per_tensor_quantize(b)

    res_out, bias_grad, pre_gelu, extra = generic_gemm(
        a_fp8,
        b_fp8,
        layout="NN",
        scale_a=scale_a,
        scale_b=scale_b,
        fp8_output=True,
    )

    assert bias_grad is None
    assert pre_gelu is None
    assert extra is None
    assert res_out is not None
    assert res_out.dtype == FP8_DTYPE
    assert res_out.shape == (M, N)


@pytest.mark.generic_gemm
@pytest.mark.skipif(not TE_AVAILABLE, reason="TE not available")
@pytest.mark.parametrize("M, N, K", SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_generic_gemm_te_ref_matmul(M, N, K, dtype):
    inp = torch.randn((M, K), dtype=dtype, device="cuda")
    weight = torch.randn((N, K), dtype=dtype, device="cuda")

    te_out, te_bias_grad, te_pre_gelu, _ = te_general_gemm(weight, inp)

    gems_out, gems_bias_grad, gems_pre_gelu, gems_extra = generic_gemm(
        inp, weight, layout="NT"
    )

    assert te_bias_grad is None
    assert te_pre_gelu is None
    assert gems_bias_grad is None
    assert gems_pre_gelu is None
    assert gems_extra is None

    utils.gems_assert_close(gems_out, utils.to_reference(te_out), dtype, reduce_dim=K)


@pytest.mark.generic_gemm
@pytest.mark.skipif(not TE_AVAILABLE, reason="TE not available")
@pytest.mark.parametrize("M, N, K", SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_generic_gemm_te_ref_bias(M, N, K, dtype):
    inp = torch.randn((M, K), dtype=dtype, device="cuda")
    weight = torch.randn((N, K), dtype=dtype, device="cuda")
    bias = torch.randn((N,), dtype=dtype, device="cuda")

    te_out, te_bias_grad, te_pre_gelu, _ = te_general_gemm(weight, inp, bias=bias)

    gems_out, gems_bias_grad, gems_pre_gelu, gems_extra = generic_gemm(
        inp, weight, layout="NT", bias=bias
    )

    assert te_bias_grad is None
    assert te_pre_gelu is None
    assert gems_bias_grad is None
    assert gems_pre_gelu is None
    assert gems_extra is None

    utils.gems_assert_close(gems_out, utils.to_reference(te_out), dtype, reduce_dim=K)


@pytest.mark.generic_gemm
@pytest.mark.skipif(not TE_AVAILABLE, reason="TE not available")
@pytest.mark.parametrize("M, N, K", SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_generic_gemm_te_ref_gelu(M, N, K, dtype):
    inp = torch.randn((M, K), dtype=dtype, device="cuda")
    weight = torch.randn((N, K), dtype=dtype, device="cuda")

    te_out, te_bias_grad, te_pre_gelu, _ = te_general_gemm(weight, inp, gelu=True)

    gems_out, gems_bias_grad, gems_pre_gelu, gems_extra = generic_gemm(
        inp, weight, layout="NT", gelu=True
    )

    assert te_bias_grad is None
    assert te_pre_gelu is not None
    assert gems_bias_grad is None
    assert gems_pre_gelu is not None
    assert gems_extra is None

    utils.gems_assert_close(gems_out, utils.to_reference(te_out), dtype, reduce_dim=K)
    utils.gems_assert_close(
        gems_pre_gelu,
        utils.to_reference(te_pre_gelu),
        dtype,
        reduce_dim=K,
    )


@pytest.mark.generic_gemm
@pytest.mark.skipif(not TE_AVAILABLE, reason="TE not available")
@pytest.mark.parametrize("M, N, K", SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_generic_gemm_te_ref_bias_gelu(M, N, K, dtype):
    inp = torch.randn((M, K), dtype=dtype, device="cuda")
    weight = torch.randn((N, K), dtype=dtype, device="cuda")
    bias = torch.randn((N,), dtype=dtype, device="cuda")

    te_out, te_bias_grad, te_pre_gelu, _ = te_general_gemm(
        weight, inp, bias=bias, gelu=True
    )

    gems_out, gems_bias_grad, gems_pre_gelu, gems_extra = generic_gemm(
        inp, weight, layout="NT", bias=bias, gelu=True
    )

    assert te_bias_grad is None
    assert te_pre_gelu is not None
    assert gems_bias_grad is None
    assert gems_pre_gelu is not None
    assert gems_extra is None

    utils.gems_assert_close(gems_out, utils.to_reference(te_out), dtype, reduce_dim=K)
    utils.gems_assert_close(
        gems_pre_gelu,
        utils.to_reference(te_pre_gelu),
        dtype,
        reduce_dim=K,
    )


@pytest.mark.generic_gemm
@pytest.mark.skipif(not TE_AVAILABLE, reason="TE not available")
@pytest.mark.skipif(not FP8_SUPPORTED, reason="FP8 requires SM>=90")
@pytest.mark.parametrize("M, N, K", FP8_SHAPES)
def test_generic_gemm_te_ref_fp8(M, N, K):
    from transformer_engine.pytorch import fp8_autocast

    inp = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")

    a_fp8, scale_a = _per_tensor_quantize(inp)
    b_fp8, scale_b = _per_tensor_quantize(weight)

    with fp8_autocast(enabled=True):
        te_out, te_bias_grad, te_pre_gelu, _ = te_general_gemm(
            weight, inp, out_dtype=torch.bfloat16
        )

    gems_out, gems_bias_grad, gems_pre_gelu, gems_extra = generic_gemm(
        a_fp8, b_fp8, layout="NT", scale_a=scale_a, scale_b=scale_b
    )

    assert te_bias_grad is None
    assert te_pre_gelu is None
    assert gems_bias_grad is None
    assert gems_pre_gelu is None
    assert gems_extra is None

    utils.gems_assert_close(
        gems_out,
        utils.to_reference(te_out),
        torch.bfloat16,
        reduce_dim=K,
        atol=0.5,
    )
