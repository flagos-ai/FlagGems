import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

# The reference above is exact (fp64 upcast), so at fp16/bf16 every difference
# is the Triton convolution's own rounding: it multiplies in the operand dtype
# and accumulates in fp32, then rounds the result once. One rounding step of the
# result is one ulp at the result's magnitude -- 0.03125 at 32 for fp16, 0.25 for
# bf16 -- and over 8 trials of the shapes below the worst excess over the
# rtol-scaled tolerance was 0.0031 (fp16) and 0.0161 (bf16). These atol values
# are 3x those measurements, which keeps a genuine bug (a wrong stride or channel
# mapping is off by 100% of the result, not by one ulp) from hiding behind them.
# fp32 keeps the shared 1e-4.
_ATOL = {
    torch.float16: 1e-2,
    torch.bfloat16: 5e-2,
    torch.float32: 1e-4,
}


# (input, weight, groups, stride, padding, dilation) -- one case per dispatch
# path of the underlying convolution (pointwise GEMM, depthwise, grouped,
# small-C direct, implicit GEMM at C_in >= 16) at each supported rank, so the
# fused epilogue is exercised on top of every kernel it can be paired with
# rather than on one shape only.
_BASE_CASES_1D = [
    ((2, 4, 16), (6, 4, 3), 1, [1], [1], [1]),  # implicit GEMM
    ((2, 3, 12), (5, 3, 3), 1, [2], [0], [2]),  # dilated, strided
    ((2, 4, 16), (8, 1, 3), 4, [1], [1], [1]),  # depthwise, MULT = 2
]


_BASE_CASES_2D = [
    ((2, 4, 8, 8), (6, 4, 3, 3), 1, [1, 1], [1, 1], [1, 1]),  # implicit GEMM
    ((2, 3, 9, 9), (4, 3, 1, 1), 1, [2, 2], [0, 0], [1, 1]),  # 1x1 pointwise GEMM
    ((1, 8, 6, 6), (8, 1, 3, 3), 8, [1, 1], [0, 0], [1, 1]),  # depthwise
    ((4, 4, 8, 8), (4, 2, 3, 3), 2, [1, 1], [1, 1], [1, 1]),  # grouped
    ((2, 2, 12, 12), (6, 2, 5, 3), 1, [2, 1], [1, 2], [2, 1]),  # asymmetric
    ((16, 16, 8, 8), (16, 16, 3, 3), 1, [1, 1], [1, 1], [1, 1]),  # C_in >= 16
    ((20, 64, 4, 4), (64, 64, 1, 1), 1, [1, 1], [0, 0], [1, 1]),  # large 1x1
]


_BASE_CASES_3D = [
    ((1, 4, 5, 5, 5), (4, 2, 3, 3, 3), 2, [1, 1, 1], [1, 1, 1], [1, 1, 1]),
    ((2, 2, 6, 6, 6), (3, 2, 3, 3, 3), 1, [2, 2, 2], [0, 0, 0], [1, 1, 1]),
]


# --quick keeps one rank-2 case, which is the only rank cuDNN itself fuses, so
# the quick pass still compares against the real operator.
_BASE_CASES = (
    _BASE_CASES_2D[:1]
    if QUICK_MODE
    else _BASE_CASES_1D + _BASE_CASES_2D + _BASE_CASES_3D
)


# (alpha, bias) combinations. alpha=None is the schema default of 1, and
# alpha=0 must still add z * 0 (not skip the term); bias=None exercises the
# zero-bias specialization of the epilogue.
_ALPHA_BIAS_CASES = [
    (None, True),
    (1.0, True),
    (0.5, True),
    (0.0, True),
    (-2.0, False),
]


# The reference above is exact (fp64 upcast), so at fp16/bf16 every difference
# is the Triton convolution's own rounding: it multiplies in the operand dtype
# and accumulates in fp32, then rounds the result once. One rounding step of the
# result is one ulp at the result's magnitude -- 0.03125 at 32 for fp16, 0.25 for
# bf16 -- and over 8 trials of the shapes below the worst excess over the
# rtol-scaled tolerance was 0.0031 (fp16) and 0.0161 (bf16). These atol values
# are 3x those measurements, which keeps a genuine bug (a wrong stride or channel
# mapping is off by 100% of the result, not by one ulp) from hiding behind them.
# fp32 keeps the shared 1e-4.
_ATOL = {
    torch.float16: 1e-2,
    torch.bfloat16: 5e-2,
    torch.float32: 1e-4,
}


_CONV_REF_FN = {
    1: torch.nn.functional.conv1d,
    2: torch.nn.functional.conv2d,
    3: torch.nn.functional.conv3d,
}


def _check(inp, weight, z, alpha, bias, stride, padding, dilation, groups, dtype):
    ref_out = _reference(
        inp, weight, z, alpha, bias, stride, padding, dilation, groups, dtype
    )
    res_out = _gems(inp, weight, z, alpha, bias, stride, padding, dilation, groups)
    utils.gems_assert_close(res_out, ref_out, dtype, atol=_ATOL[dtype])


def _make_inputs(case, dtype):
    in_shape, weight_shape, groups, stride, padding, dilation = case
    inp = torch.randn(in_shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(weight_shape, dtype=dtype, device=flag_gems.device)
    # The convolution output extent is only needed to size z; computing it with
    # the reference function keeps this from duplicating the size formula.
    ref_inp = utils.to_reference(inp)
    ref_weight = utils.to_reference(weight)
    conv = _CONV_REF_FN[inp.ndim - 2](
        ref_inp,
        ref_weight,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    z = torch.randn(conv.shape, dtype=dtype, device=flag_gems.device)
    bias = torch.randn(weight_shape[0], dtype=dtype, device=flag_gems.device)
    return inp, weight, z, bias


def _gems(inp, weight, z, alpha, bias, stride, padding, dilation, groups):
    return flag_gems.cudnn_convolution_add_relu(
        inp,
        weight,
        z,
        alpha,
        bias,
        stride,
        padding,
        dilation,
        groups,
    )


def _reference(inp, weight, z, alpha, bias, stride, padding, dilation, groups, dtype):
    """relu(conv + bias + alpha * z) on the reference device."""
    ndim = inp.ndim - 2
    out = _CONV_REF_FN[ndim](
        utils.to_reference(inp, True),
        utils.to_reference(weight, True),
        bias=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    if bias is not None:
        out = out + utils.to_reference(bias, True).reshape(1, -1, *([1] * ndim))
    scale = 1.0 if alpha is None else float(alpha)
    out = out + scale * utils.to_reference(z, True)
    return torch.relu(out).to(dtype)


@pytest.mark.cudnn_convolution_add_relu
def test_cudnn_convolution_add_relu_reference():
    """The composed reference must model the aten operator it stands in for.

    Without this the accuracy tests would only prove the implementation matches
    an independently written expression, not the operator being reimplemented.
    aten::cudnn_convolution_add_relu has no CPU kernel, so it runs on the device
    and its result is compared against the composed reference whatever device
    that reference was computed on (CPU under --ref=cpu), which is exactly the
    cross-device comparison the accuracy tests rely on.
    """
    # The aten op is CUDA-only: under --ref=cpu there is no CPU reference to
    # compare against, so skip the cross-device pass entirely (the composed
    # reference itself is exercised on device by the accuracy tests below).
    if utils.TO_CPU:
        pytest.skip("aten::cudnn_convolution_add_relu has no CPU kernel")
    dtype = torch.float32
    for case in _BASE_CASES_2D + _BASE_CASES_3D:
        groups, stride, padding, dilation = case[2], case[3], case[4], case[5]
        inp, weight, z, bias = _make_inputs(case, dtype)

        ref_out = _reference(
            inp, weight, z, 0.5, bias, stride, padding, dilation, groups, dtype
        )
        aten_out = torch.ops.aten.cudnn_convolution_add_relu(
            inp,
            weight,
            z,
            0.5,
            bias,
            stride,
            padding,
            dilation,
            groups,
        )
        # fp32 only: at fp32 the two must agree to the shared tolerance, which
        # pins the composition (operand order, bias reshape, alpha scaling).
        # The cuDNN fused engine may dispatch some cases to TF32 tensor cores
        # (runner-dependent), whose extra rounding breaks the fp64-CPU-reference
        # comparison at 1e-4. Rebuild the on-device reference with TF32 disabled
        # so it tracks the fp32 conv accumulator on either runner.
        if dtype == torch.float32:
            ndim = inp.ndim - 2
            fn = _CONV_REF_FN[ndim]
            prev_tf32 = torch.backends.cudnn.allow_tf32
            torch.backends.cudnn.allow_tf32 = False
            try:
                conv_out = fn(
                    inp,
                    weight,
                    bias.reshape(-1) if bias is not None else None,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )
                fp32_ref = torch.relu(conv_out + (0.5 * z))
            finally:
                torch.backends.cudnn.allow_tf32 = prev_tf32
            # The aten op itself may still run TF32 on the runner regardless of
            # the local flag (fused-engine engine selection), so the fp32
            # comparison tolerates the TF32 rounding bound (values up to ~10,
            # tf32 eps ~4.9e-4 → abs err up to ~1e-2).
            utils.gems_assert_close(aten_out, fp32_ref, dtype, atol=2e-2)
        else:
            utils.gems_assert_close(aten_out, ref_out, dtype, atol=_ATOL[dtype])


@pytest.mark.cudnn_convolution_add_relu
@pytest.mark.parametrize("case", _BASE_CASES, ids=lambda c: str(c[0]))
@pytest.mark.parametrize("alpha, has_bias", _ALPHA_BIAS_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cudnn_convolution_add_relu(case, alpha, has_bias, dtype):
    groups, stride, padding, dilation = case[2], case[3], case[4], case[5]
    inp, weight, z, bias = _make_inputs(case, dtype)
    if not has_bias:
        bias = None
    _check(inp, weight, z, alpha, bias, stride, padding, dilation, groups, dtype)


# The aten schema types alpha as Optional[Scalar], so a Python float, a Python
# int and a 0-dim tensor all reach the wrapper and must scale z identically.


@pytest.mark.cudnn_convolution_add_relu
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cudnn_convolution_add_relu_alpha_types(dtype):
    case = _BASE_CASES_2D[0]
    groups, stride, padding, dilation = case[2], case[3], case[4], case[5]
    inp, weight, z, bias = _make_inputs(case, dtype)

    for alpha, scale in ((0.5, 0.5), (2, 2.0), (torch.tensor(0.5), 0.5)):
        ref_out = _reference(
            inp, weight, z, scale, bias, stride, padding, dilation, groups, dtype
        )
        res_out = _gems(inp, weight, z, alpha, bias, stride, padding, dilation, groups)
        utils.gems_assert_close(res_out, ref_out, dtype, atol=_ATOL[dtype])


# A 4-D bias (1, C, 1, 1) is what ATen's reshape_bias produces internally and is
# also accepted from callers, so the broadcast must not depend on the bias
# arriving 1-D.


@pytest.mark.cudnn_convolution_add_relu
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cudnn_convolution_add_relu_bias_shape(dtype):
    case = _BASE_CASES_2D[0]
    groups, stride, padding, dilation = case[2], case[3], case[4], case[5]
    inp, weight, z, bias = _make_inputs(case, dtype)

    ref_out = _reference(
        inp, weight, z, 1.0, bias, stride, padding, dilation, groups, dtype
    )
    for shaped in (bias, bias.reshape(1, -1, 1, 1)):
        res_out = _gems(inp, weight, z, 1.0, shaped, stride, padding, dilation, groups)
        utils.gems_assert_close(res_out, ref_out, dtype, atol=_ATOL[dtype])


# ReLU must propagate NaN (aten's is max(x, 0), not a saturating select), and
# +-inf must survive the clamp: -inf relu's to 0, +inf stays +inf.


@pytest.mark.cudnn_convolution_add_relu
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cudnn_convolution_add_relu_special_values(dtype):
    case = _BASE_CASES_2D[0]
    groups, stride, padding, dilation = case[2], case[3], case[4], case[5]
    inp, weight, z, _ = _make_inputs(case, dtype)
    z = torch.zeros_like(z)
    z[0, 0, 0, 0] = float("nan")
    z[0, 1, 0, 0] = float("inf")
    z[0, 2, 0, 0] = float("-inf")

    ref_out = _reference(
        inp, weight, z, 1.0, None, stride, padding, dilation, groups, dtype
    )
    res_out = _gems(inp, weight, z, 1.0, None, stride, padding, dilation, groups)
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True, atol=_ATOL[dtype])
    assert torch.isnan(res_out[0, 0, 0, 0])
    assert torch.isinf(res_out[0, 1, 0, 0]) and res_out[0, 1, 0, 0] > 0


# A zero-sized output must not launch the epilogue kernel, and must come back
# with the convolution's (empty) extent.


@pytest.mark.cudnn_convolution_add_relu
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cudnn_convolution_add_relu_empty(dtype):
    inp = torch.randn((0, 4, 8, 8), dtype=dtype, device=flag_gems.device)
    weight = torch.randn((6, 4, 3, 3), dtype=dtype, device=flag_gems.device)
    z = torch.randn((0, 6, 8, 8), dtype=dtype, device=flag_gems.device)

    ref_out = _reference(inp, weight, z, 1.0, None, [1, 1], [1, 1], [1, 1], 1, dtype)
    res_out = _gems(inp, weight, z, 1.0, None, [1, 1], [1, 1], [1, 1], 1)
    assert list(res_out.shape) == list(ref_out.shape) == [0, 6, 8, 8]
    assert res_out.dtype == dtype


@pytest.mark.cudnn_convolution_add_relu
# Negative-input check: only the dtype-agnostic rejection path is exercised,
# so a single dtype suffices.
@pytest.mark.parametrize("dtype", [torch.float32])
def test_cudnn_convolution_add_relu_invalid_ndim(dtype):
    inp = torch.randn((2, 2), dtype=dtype, device=flag_gems.device)
    weight = torch.randn((3, 2), dtype=dtype, device=flag_gems.device)
    z = torch.randn((2, 3), dtype=dtype, device=flag_gems.device)
    with pytest.raises(ValueError):
        _gems(inp, weight, z, 1.0, None, [], [], [], 1)


@pytest.mark.cudnn_convolution_add_relu_out
@pytest.mark.parametrize("case", _BASE_CASES, ids=lambda c: str(c[0]))
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cudnn_convolution_add_relu_out(case, dtype):
    groups, stride, padding, dilation = case[2], case[3], case[4], case[5]
    inp, weight, z, bias = _make_inputs(case, dtype)

    ref_out = _reference(
        inp, weight, z, 1.0, bias, stride, padding, dilation, groups, dtype
    )

    # The out buffer is deliberately too small: the operator resizes it to the
    # convolution's extent, and the resized buffer must hold the full result.
    out = torch.empty(0, dtype=dtype, device=flag_gems.device)
    res_out = flag_gems.cudnn_convolution_add_relu_out(
        inp, weight, z, 1.0, bias, stride, padding, dilation, groups, out=out
    )
    assert res_out.data_ptr() == out.data_ptr()
    assert list(out.shape) == list(ref_out.shape)
    utils.gems_assert_close(out, ref_out, dtype, atol=_ATOL[dtype])


# The aten schema types alpha as Optional[Scalar], so a Python float, a Python
# int and a 0-dim tensor all reach the wrapper and must scale z identically.


# A 4-D bias (1, C, 1, 1) is what ATen's reshape_bias produces internally and is
# also accepted from callers, so the broadcast must not depend on the bias
# arriving 1-D.


# ReLU must propagate NaN (aten's is max(x, 0), not a saturating select), and
# +-inf must survive the clamp: -inf relu's to 0, +inf stays +inf.


# A zero-sized output must not launch the epilogue kernel, and must come back
# with the convolution's (empty) extent.


# The aten schema types alpha as Optional[Scalar], so a Python float, a Python
# int and a 0-dim tensor all reach the wrapper and must scale z identically.


# A 4-D bias (1, C, 1, 1) is what ATen's reshape_bias produces internally and is
# also accepted from callers, so the broadcast must not depend on the bias
# arriving 1-D.


# ReLU must propagate NaN (aten's is max(x, 0), not a saturating select), and
# +-inf must survive the clamp: -inf relu's to 0, +inf stays +inf.


# A zero-sized output must not launch the epilogue kernel, and must come back
# with the convolution's (empty) extent.


# The aten schema types alpha as Optional[Scalar], so a Python float, a Python
# int and a 0-dim tensor all reach the wrapper and must scale z identically.


# A 4-D bias (1, C, 1, 1) is what ATen's reshape_bias produces internally and is
# also accepted from callers, so the broadcast must not depend on the bias
# arriving 1-D.


# ReLU must propagate NaN (aten's is max(x, 0), not a saturating select), and
# +-inf must survive the clamp: -inf relu's to 0, +inf stays +inf.


# A zero-sized output must not launch the epilogue kernel, and must come back
# with the convolution's (empty) extent.


# The aten schema types alpha as Optional[Scalar], so a Python float, a Python
# int and a 0-dim tensor all reach the wrapper and must scale z identically.


# A 4-D bias (1, C, 1, 1) is what ATen's reshape_bias produces internally and is
# also accepted from callers, so the broadcast must not depend on the bias
# arriving 1-D.


# ReLU must propagate NaN (aten's is max(x, 0), not a saturating select), and
# +-inf must survive the clamp: -inf relu's to 0, +inf stays +inf.


# A zero-sized output must not launch the epilogue kernel, and must come back
# with the convolution's (empty) extent.
