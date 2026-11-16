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

import math

import pytest
import torch
import torch.nn.functional as F

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    LAYER_NORM_SHAPES = [(1, 40999)]
    LAYER_NORM_MEDIUM_SHAPES = []
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    LAYER_NORM_SHAPES = [(200, 36), (4096, 100), (1, 40999), (100, 40499), (4096, 256)]
    LAYER_NORM_MEDIUM_SHAPES = [
        (256, 512),
        (4096, 256),
        (4096, 512),
        (2048, 1024),
        (1024, 2048),
    ]

LAYER_NORM_FORWARD_SHAPES = LAYER_NORM_SHAPES + LAYER_NORM_MEDIUM_SHAPES


@pytest.mark.native_layer_norm
# Cover one-, two-, and three-dimensional normalized suffixes of equal size.
@pytest.mark.parametrize(
    "shape, normalized_shape",
    [
        ((4, 64), (64,)),
        ((4, 8, 8), (8, 8)),
        ((4, 4, 4, 4), (4, 4, 4)),
        pytest.param(
            (4, 40999),
            (40999,),
            marks=pytest.mark.skipif(
                flag_gems.vendor_name != "thead",
                reason="Check FP32 statistics on the PPU wide-reduction path",
            ),
        ),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("affine", [True, False])
def test_native_layer_norm(shape, normalized_shape, dtype, affine, caplog):
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = (
        torch.randn(normalized_shape, dtype=dtype, device=flag_gems.device)
        if affine
        else None
    )
    bias = torch.randn_like(weight) if affine else None
    eps = 1e-5

    ref_result = torch.ops.aten.native_layer_norm.default(
        utils.to_reference(inp, True),
        normalized_shape,
        utils.to_reference(weight, True),
        utils.to_reference(bias, True),
        eps,
    )

    with caplog.at_level(
        "DEBUG", logger=utils.gems_log_logger(flag_gems.native_layer_norm)
    ):
        result = flag_gems.native_layer_norm(inp, normalized_shape, weight, bias, eps)

    assert (
        f"{utils.gems_log_prefix(flag_gems.native_layer_norm)} NATIVE_LAYER_NORM"
        in caplog.text
    )
    assert len(result) == len(ref_result) == 3
    stats_dtype = (
        torch.float32
        if flag_gems.vendor_name in ("hygon", "nvidia", "thead")
        else dtype
    )
    for index, (actual, expected, expected_dtype) in enumerate(
        zip(result, ref_result, (dtype, stats_dtype, stats_dtype))
    ):
        if flag_gems.vendor_name in ("nvidia", "thead") and index > 0:
            assert actual.dtype == torch.float32
        utils.gems_assert_close(actual, expected, expected_dtype)


@pytest.mark.layer_norm
@pytest.mark.parametrize("shape", LAYER_NORM_FORWARD_SHAPES)
@pytest.mark.parametrize("wb_none", [False, True])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_layer_norm(shape, dtype, wb_none):
    torch.manual_seed(0)
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if wb_none:
        res_weight = None
        res_bias = None
    else:
        res_weight = torch.randn(shape[1:], dtype=dtype, device=flag_gems.device)
        res_bias = torch.randn(shape[1:], dtype=dtype, device=flag_gems.device)
    eps = 1e-5

    ref_inp = utils.to_reference(res_inp, True)
    ref_weight = utils.to_reference(res_weight, True)
    ref_bias = utils.to_reference(res_bias, True)

    ref_out = torch.layer_norm(
        ref_inp,
        shape[1:],
        weight=ref_weight,
        bias=ref_bias,
        eps=eps,
    )
    res_out, _, _ = flag_gems.layer_norm(
        res_inp,
        shape[1:],
        weight=res_weight,
        bias=res_bias,
        eps=eps,
    )

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.layer_norm
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", [(17, 37), (4, 8, 512)])
def test_layer_norm_fp32_affine(shape, dtype):
    torch.manual_seed(0)
    # Slicing also covers the layout conversion before the fused reduction.
    inp = torch.randn(
        (*shape[:-1], shape[-1] * 2), device=flag_gems.device, dtype=dtype
    )
    inp = inp[..., ::2]
    weight = torch.randn(shape[-1], device=flag_gems.device, dtype=torch.float32)
    bias = torch.randn_like(weight)
    ref = torch.nn.functional.layer_norm(
        utils.to_reference(inp, True),
        (shape[-1],),
        utils.to_reference(weight, True),
        utils.to_reference(bias, True),
        1e-6,
    )
    result, _, _ = flag_gems.layer_norm(inp, (shape[-1],), weight, bias, 1e-6)
    utils.gems_assert_close(result, ref, dtype)


@pytest.mark.layer_norm
@pytest.mark.parametrize("shape", LAYER_NORM_MEDIUM_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_native_layer_norm_statistics(shape, dtype):
    torch.manual_seed(0)
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    normalized_shape = shape[1:]
    res_weight = torch.randn(normalized_shape, dtype=dtype, device=flag_gems.device)
    res_bias = torch.randn(normalized_shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(res_inp, True)
    ref_weight = utils.to_reference(res_weight, True)
    ref_bias = utils.to_reference(res_bias, True)
    ref_out, ref_mean, ref_rstd = torch.ops.aten.native_layer_norm(
        ref_inp,
        normalized_shape,
        ref_weight,
        ref_bias,
        1e-5,
    )
    res_out, res_mean, res_rstd = flag_gems.native_layer_norm(
        res_inp,
        normalized_shape,
        res_weight,
        res_bias,
        1e-5,
    )

    reduce_dim = math.prod(normalized_shape)
    stats_dtype = (
        torch.float32
        if flag_gems.vendor_name in ("hygon", "nvidia", "thead")
        else dtype
    )
    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(res_mean, ref_mean, stats_dtype, reduce_dim=reduce_dim)
    utils.gems_assert_close(res_rstd, ref_rstd, stats_dtype, reduce_dim=reduce_dim)


@pytest.mark.layer_norm_backward
@pytest.mark.parametrize("shape", LAYER_NORM_SHAPES)
@pytest.mark.parametrize("wb_none", [False, True])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_layer_norm_backward(monkeypatch, shape, dtype, wb_none):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    if flag_gems.vendor_name == "mthreads":
        # Compatible with older versions of LLVM
        monkeypatch.setenv("DISABLE_LLVM_OPT", "1")

    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_grad = torch.randn_like(res_inp)
    res_mean = torch.randn(shape[0], dtype=dtype, device=flag_gems.device)
    res_rstd = torch.randn(shape[0], dtype=dtype, device=flag_gems.device)
    if wb_none:
        res_weight = None
        res_bias = None
        output_mask = [True, False, False]
    else:
        res_weight = torch.randn(shape[1:], dtype=dtype, device=flag_gems.device)
        res_bias = torch.randn(shape[1:], dtype=dtype, device=flag_gems.device)
        output_mask = [True, True, True]

    normalized_shape = shape[1:]

    ref_inp = utils.to_reference(res_inp, True)
    ref_grad = utils.to_reference(res_grad, True)
    ref_mean = utils.to_reference(res_mean, True)
    ref_rstd = utils.to_reference(res_rstd, True)
    ref_weight = utils.to_reference(res_weight, True)
    ref_bias = utils.to_reference(res_bias, True)

    (
        ref_in_grad,
        ref_weight_grad,
        ref_bias_grad,
    ) = torch.ops.aten.native_layer_norm_backward(
        ref_grad,
        ref_inp,
        normalized_shape,
        ref_mean,
        ref_rstd,
        ref_weight,
        ref_bias,
        output_mask,
    )
    (
        res_in_grad,
        res_weight_grad,
        res_bias_grad,
    ) = flag_gems.layer_norm_backward(
        res_grad,
        res_inp,
        normalized_shape,
        res_mean,
        res_rstd,
        res_weight,
        res_bias,
        output_mask,
    )

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)
    if not wb_none:
        utils.gems_assert_close(
            res_weight_grad, ref_weight_grad, dtype, reduce_dim=shape[0]
        )
        utils.gems_assert_close(
            res_bias_grad, ref_bias_grad, dtype, reduce_dim=shape[0]
        )


# Keep accuracy coverage independent of benchmark configuration.
LAYERNORM_CASES = [
    ((200, 36), (36,), 1e-05),
    ((4096, 100), (100,), 1e-05),
    ((1, 40999), (40999,), 1e-05),
    ((100, 40499), (40499,), 1e-05),
    ((4096, 256), (256,), 1e-05),
    ((256, 512), (512,), 1e-05),
    ((4096, 512), (512,), 1e-05),
    ((2048, 1024), (1024,), 1e-05),
    ((1024, 2048), (2048,), 1e-05),
    ((4, 64), (64,), 1e-05),
    ((4, 8, 8), (8, 8), 1e-05),
    ((4, 4, 4, 4), (4, 4, 4), 1e-05),
    ((16, 16, 64), (16, 64), 1e-05),
    ((16, 16, 1024), (16, 1024), 1e-05),
    ((1, 8, 4, 4), (8, 4, 4), 1e-05),
    ((1, 2), (2,), 1e-05),
    ((200, 40999), (40999,), 1e-05),
    ((2, 32), (32,), 1e-05),
    ((65536, 1, 1152), (1152,), 1e-06),
]
# Exercise the existing streaming path for normalized sizes above one tile.
LARGE_LAYERNORM_CASES = (
    ((16, 16, 4098), (16, 4098)),
    ((16, 8, 128, 128), (8, 128, 128)),
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "shape,norm",
    [
        ((0, 1152), (1152,)),
        ((1, 1), (1,)),
        ((7, 113), (113,)),
        ((3, 5, 7), (5, 7)),
        ((33, 1152), (1152,)),
        ((2, 16384), (16384,)),
        ((1, 65536), (65536,)),
    ],
)
@pytest.mark.parametrize("affine", ["none", "weight", "bias", "both"])
@pytest.mark.layer_norm
@pytest.mark.skipif(flag_gems.vendor_name != "metax", reason="MetaX C550")
def test_layernorm_contract(dtype, shape, norm, affine):
    torch.manual_seed(42)
    tensor = torch.randn(shape, dtype=dtype, device="cuda")
    weight = (
        torch.randn(norm, dtype=dtype, device="cuda")
        if affine in ("weight", "both")
        else None
    )
    bias = (
        torch.randn(norm, dtype=dtype, device="cuda")
        if affine in ("bias", "both")
        else None
    )
    expected = F.layer_norm(tensor, norm, weight, bias, 1e-6)
    actual, mean, rstd = flag_gems.layer_norm(tensor, norm, weight, bias, 1e-6)
    tolerance = (
        1e-6 if dtype == torch.float32 else (1e-2 if dtype == torch.bfloat16 else 1e-3)
    )
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    assert actual.shape == tensor.shape and actual.dtype == tensor.dtype
    _, expected_mean, expected_rstd = torch.ops.aten.native_layer_norm(
        tensor, norm, weight, bias, 1e-6
    )
    assert mean.shape == rstd.shape == (tensor.numel() // math.prod(norm),)
    assert mean.dtype == rstd.dtype == tensor.dtype
    for stats, expected_stats in ((mean, expected_mean), (rstd, expected_rstd)):
        torch.testing.assert_close(
            stats,
            expected_stats.reshape(-1).to(tensor.dtype),
            atol=tolerance,
            rtol=tolerance,
        )


@pytest.mark.parametrize("layout", ["transpose", "slice", "expand"])
@pytest.mark.layer_norm
@pytest.mark.skipif(flag_gems.vendor_name != "metax", reason="MetaX C550")
def test_layernorm_strides(layout):
    torch.manual_seed(13)
    if layout == "transpose":
        tensor = torch.randn(5, 11, 7, device="cuda").transpose(0, 1)
    elif layout == "slice":
        tensor = torch.randn(11, 5, 14, device="cuda")[:, :, ::2]
    else:
        tensor = torch.randn(1, 5, 7, device="cuda").expand(11, 5, 7)
    weight = torch.randn(7, 5, device="cuda").t()
    bias = torch.randn(5, 14, device="cuda")[:, ::2]
    torch.testing.assert_close(
        flag_gems.layer_norm(tensor, (5, 7), weight, bias)[0],
        F.layer_norm(tensor, (5, 7), weight, bias),
        atol=2e-6,
        rtol=2e-6,
    )


@pytest.mark.parametrize(
    "value", [0.0, -0.0, float("nan"), float("inf"), -float("inf"), 10000.0]
)
@pytest.mark.layer_norm
@pytest.mark.skipif(flag_gems.vendor_name != "metax", reason="MetaX C550")
def test_layernorm_special_values(value):
    tensor = torch.full((3, 1152), value, device="cuda")
    tensor[1, 0] = 1.0
    torch.testing.assert_close(
        flag_gems.layer_norm(tensor, (1152,))[0],
        F.layer_norm(tensor, (1152,)),
        atol=2e-3,
        rtol=2e-3,
        equal_nan=True,
    )


@pytest.mark.layer_norm
@pytest.mark.skipif(flag_gems.vendor_name != "metax", reason="MetaX C550")
def test_layernorm_real_shape():
    torch.manual_seed(77)
    tensor = torch.randn(65536, 1, 1152, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(1152, device="cuda", dtype=tensor.dtype)
    bias = torch.randn_like(weight)
    torch.testing.assert_close(
        flag_gems.layer_norm(tensor, (1152,), weight, bias, 1e-6)[0],
        F.layer_norm(tensor, (1152,), weight, bias, 1e-6),
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.layer_norm
@pytest.mark.skipif(flag_gems.vendor_name != "metax", reason="MetaX C550")
def test_layernorm_errors():
    torch.manual_seed(42)
    tensor = torch.randn(3, 1152, device="cuda")
    with pytest.raises(ValueError):
        flag_gems.layer_norm(tensor, (1024,))[0]
    with pytest.raises(ValueError):
        flag_gems.layer_norm(
            tensor, (1152,), torch.empty(1152, device="cuda", dtype=torch.bfloat16)
        )[0]
    with pytest.raises(NotImplementedError):
        flag_gems.layer_norm(tensor.double(), (1152,))[0]
    tensor.requires_grad_()
    torch.testing.assert_close(
        flag_gems.layer_norm(tensor, (1152,))[0],
        F.layer_norm(tensor, (1152,)),
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "width", [320, 384, 576, 768, 1152, 1280, 1536, 2304, 2560, 3072]
)
@pytest.mark.parametrize("affine", ["none", "weight", "bias", "both"])
@pytest.mark.layer_norm
@pytest.mark.skipif(flag_gems.vendor_name != "metax", reason="MetaX C550")
def test_layernorm_exact_segments(dtype, width, affine):
    torch.manual_seed(20261103)
    tensor = torch.randn((257, 1, width), dtype=dtype, device="cuda")
    weight = (
        torch.randn(width, dtype=dtype, device="cuda")
        if affine in ("weight", "both")
        else None
    )
    bias = (
        torch.randn(width, dtype=dtype, device="cuda")
        if affine in ("bias", "both")
        else None
    )
    actual = flag_gems.layer_norm(tensor, (width,), weight, bias, 1e-6)[0]
    expected = F.layer_norm(tensor, (width,), weight, bias, 1e-6)
    tolerance = (
        1e-6 if dtype == torch.float32 else (1e-2 if dtype == torch.bfloat16 else 1e-3)
    )
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("shape,normalized_shape,eps", LAYERNORM_CASES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("affine", [False, True])
@pytest.mark.layer_norm
@pytest.mark.skipif(flag_gems.vendor_name != "metax", reason="MetaX C550")
def test_layernorm_core_shapes(shape, normalized_shape, eps, dtype, affine):
    torch.manual_seed(20261104)
    tensor = torch.randn(shape, dtype=dtype, device="cuda")
    weight = (
        torch.randn(normalized_shape, dtype=dtype, device="cuda") if affine else None
    )
    bias = torch.randn_like(weight) if affine else None
    actual = flag_gems.layer_norm(tensor, normalized_shape, weight, bias, eps)[0]
    expected = F.layer_norm(tensor, normalized_shape, weight, bias, eps)
    tolerance = (
        1e-6 if dtype == torch.float32 else (1e-2 if dtype == torch.bfloat16 else 1e-3)
    )
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("shape,normalized_shape", LARGE_LAYERNORM_CASES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.layer_norm
@pytest.mark.skipif(flag_gems.vendor_name != "metax", reason="MetaX C550")
def test_layernorm_large_normalized_extents(shape, normalized_shape, dtype):
    torch.manual_seed(20261104)
    tensor = torch.randn(shape, dtype=dtype, device="cuda")
    actual, mean, rstd = flag_gems.layer_norm(tensor, normalized_shape)
    expected, expected_mean, expected_rstd = torch.ops.aten.native_layer_norm(
        tensor, normalized_shape, None, None, 1e-5
    )
    tolerance = (
        1e-6 if dtype == torch.float32 else (1e-2 if dtype == torch.bfloat16 else 1e-3)
    )
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    for stats, expected_stats in ((mean, expected_mean), (rstd, expected_rstd)):
        torch.testing.assert_close(
            stats, expected_stats.reshape(-1).to(dtype), atol=tolerance, rtol=tolerance
        )


@pytest.mark.layer_norm_backward
@pytest.mark.skipif(flag_gems.vendor_name != "metax", reason="MetaX C550")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "shape,norm", [((7, 113), (113,)), ((3, 5, 7), (7,)), ((4, 8, 8), (8, 8))]
)
def test_layer_norm_saved_statistics_backward(shape, norm, dtype):
    torch.manual_seed(20261105)
    tensor = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)
    weight = torch.randn(norm, dtype=dtype, device="cuda", requires_grad=True)
    bias = torch.randn_like(weight, requires_grad=True)
    output, mean, rstd = flag_gems.layer_norm(tensor, norm, weight, bias)
    grad = torch.randn_like(output)
    actual = flag_gems.layer_norm_backward(
        grad, tensor, norm, mean, rstd, weight, bias, (True, True, True)
    )
    expected = torch.ops.aten.native_layer_norm_backward(
        utils.to_reference(grad, True),
        utils.to_reference(tensor, True),
        norm,
        utils.to_reference(mean, True),
        utils.to_reference(rstd, True),
        utils.to_reference(weight, True),
        utils.to_reference(bias, True),
        (True, True, True),
    )
    for actual_grad, expected_grad in zip(actual, expected):
        utils.gems_assert_close(
            actual_grad, expected_grad, dtype, reduce_dim=math.prod(norm)
        )


nvidia_only = pytest.mark.skipif(
    flag_gems.vendor_name != "nvidia",
    reason="the FP32 statistics contract and row scheduling are NVIDIA-scoped",
)

# The NVIDIA entry switches schedules at a normalized width of 16384: the
# single-pass persistent kernel with tiled rows at or below it, the two-pass
# loop kernel above. Straddle that, and the generic host's own 128 and 4096
# boundaries, so every path is covered.
NVIDIA_PATH_SHAPES = [
    (4096, 128),
    (4096, 129),
    (65536, 256),
    (4096, 4096),
    (16, 16384),
    (16, 16385),
    (7, 40999),
]


@pytest.mark.layer_norm
@pytest.mark.native_layer_norm
@nvidia_only
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(37, 1152), (4, 40999)])
def test_native_layer_norm_fp32_statistics(shape, dtype):
    """Half-precision inputs still return FP32 statistics, as ATen does."""
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    normalized_shape = shape[1:]
    weight = torch.randn(normalized_shape, dtype=dtype, device=flag_gems.device)
    bias = torch.randn_like(weight)

    _, ref_mean, ref_rstd = torch.ops.aten.native_layer_norm(
        utils.to_reference(inp, True),
        normalized_shape,
        utils.to_reference(weight, True),
        utils.to_reference(bias, True),
        1e-5,
    )
    _, res_mean, res_rstd = flag_gems.native_layer_norm(
        inp, normalized_shape, weight, bias, 1e-5
    )
    assert res_mean.dtype == torch.float32
    assert res_rstd.dtype == torch.float32
    reduce_dim = math.prod(normalized_shape)
    utils.gems_assert_close(res_mean, ref_mean, torch.float32, reduce_dim=reduce_dim)
    utils.gems_assert_close(res_rstd, ref_rstd, torch.float32, reduce_dim=reduce_dim)


@pytest.mark.layer_norm
@nvidia_only
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", NVIDIA_PATH_SHAPES)
def test_layer_norm_nvidia_paths(shape, dtype):
    """Every scheduling path, including both sides of the width limit."""
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    normalized_shape = shape[1:]
    weight = torch.randn(normalized_shape, dtype=dtype, device=flag_gems.device)
    bias = torch.randn_like(weight)

    ref_out = torch.layer_norm(
        utils.to_reference(inp, True),
        normalized_shape,
        weight=utils.to_reference(weight, True),
        bias=utils.to_reference(bias, True),
        eps=1e-5,
    )
    res_out, mean, rstd = flag_gems.layer_norm(
        inp, normalized_shape, weight=weight, bias=bias, eps=1e-5
    )
    assert mean.shape == rstd.shape == (shape[0],)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.layer_norm
@nvidia_only
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(17, 37), (4, 8, 512)])
def test_layer_norm_fp32_affine_noncontiguous(shape, dtype):
    """FP32 affine parameters with a half-precision, strided input."""
    torch.manual_seed(0)
    # Slicing also exercises the layout conversion before the reduction.
    inp = torch.randn(
        (*shape[:-1], shape[-1] * 2), device=flag_gems.device, dtype=dtype
    )[..., ::2]
    assert not inp.is_contiguous()
    weight = torch.randn(shape[-1], device=flag_gems.device, dtype=torch.float32)
    bias = torch.randn_like(weight)

    ref_out = torch.nn.functional.layer_norm(
        utils.to_reference(inp, True),
        (shape[-1],),
        utils.to_reference(weight, True),
        utils.to_reference(bias, True),
        1e-6,
    )
    res_out, _, _ = flag_gems.layer_norm(inp, (shape[-1],), weight, bias, 1e-6)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.layer_norm
@nvidia_only
def test_layer_norm_empty_rows():
    """A zero-sized outer dimension returns empty tensors, launching nothing."""
    inp = torch.randn((0, 64), device=flag_gems.device)
    weight = torch.randn(64, device=flag_gems.device)
    out, mean, rstd = flag_gems.layer_norm(inp, (64,), weight, weight, 1e-5)
    assert out.shape == (0, 64)
    assert mean.numel() == rstd.numel() == 0
    assert mean.dtype == rstd.dtype == torch.float32
