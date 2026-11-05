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

LAYER_NORM_FORWARD_CASES = [
    (shape, shape[1:], 1e-5)
    for shape in dict.fromkeys(LAYER_NORM_SHAPES + LAYER_NORM_MEDIUM_SHAPES)
]
if not cfg.QUICK_MODE:
    # Match the public benchmark suffixes and retain the real workload contract.
    LAYER_NORM_FORWARD_CASES += [
        ((4, 16, 64, 4), (16, 64, 4), 1e-05),
        ((16, 16, 8, 48), (16, 8, 48), 1e-05),
        ((16, 16, 8, 88), (16, 8, 88), 1e-05),
        ((16, 16, 128), (16, 128), 1e-05),
        ((20, 6, 65536), (6, 65536), 1e-05),
        ((16, 16, 64), (16, 64), 1e-05),
        ((16, 16, 1024), (16, 1024), 1e-05),
        ((16, 16, 4098), (16, 4098), 1e-05),
        ((1, 8, 4, 4), (8, 4, 4), 1e-05),
        ((16, 8, 128, 128), (8, 128, 128), 1e-05),
        ((4, 64), (64,), 1e-05),
        ((4, 8, 8), (8, 8), 1e-05),
        ((4, 4, 4, 4), (4, 4, 4), 1e-05),
        ((65536, 1, 1152), (1152,), 1e-06),
    ]


@pytest.mark.native_layer_norm
# Cover one-, two-, and three-dimensional normalized suffixes of equal size.
@pytest.mark.parametrize(
    "shape, normalized_shape",
    [((4, 64), (64,)), ((4, 8, 8), (8, 8)), ((4, 4, 4, 4), (4, 4, 4))],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("affine", [True, False])
def test_native_layer_norm(shape, normalized_shape, dtype, affine, caplog):
    utils.init_seed(0)
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
    for index, (actual, expected) in enumerate(zip(result, ref_result)):
        result_dtype = (
            torch.float32 if index and flag_gems.vendor_name == "mthreads" else dtype
        )
        utils.gems_assert_close(actual, expected, result_dtype)


@pytest.mark.layer_norm
@pytest.mark.parametrize("shape, normalized_shape, eps", LAYER_NORM_FORWARD_CASES)
@pytest.mark.parametrize("wb_none", [False, True])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_layer_norm(shape, normalized_shape, eps, dtype, wb_none):
    utils.init_seed(0)

    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if wb_none:
        res_weight = None
        res_bias = None
    else:
        res_weight = torch.randn(normalized_shape, dtype=dtype, device=flag_gems.device)
        res_bias = torch.randn(normalized_shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(res_inp, True)
    ref_weight = utils.to_reference(res_weight, True)
    ref_bias = utils.to_reference(res_bias, True)

    ref_out = torch.layer_norm(
        ref_inp,
        normalized_shape,
        weight=ref_weight,
        bias=ref_bias,
        eps=eps,
    )
    res_out, _, _ = flag_gems.layer_norm(
        res_inp,
        normalized_shape,
        weight=res_weight,
        bias=res_bias,
        eps=eps,
    )

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.layer_norm
@pytest.mark.parametrize("shape, normalized_shape, eps", LAYER_NORM_FORWARD_CASES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_native_layer_norm_statistics(shape, normalized_shape, eps, dtype):
    utils.init_seed(0)
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
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
        eps,
    )
    res_out, res_mean, res_rstd = flag_gems.native_layer_norm(
        res_inp,
        normalized_shape,
        res_weight,
        res_bias,
        eps,
    )

    reduce_dim = math.prod(normalized_shape)
    utils.gems_assert_close(res_out, ref_out, dtype)
    stats_dtype = torch.float32 if flag_gems.vendor_name == "mthreads" else dtype
    utils.gems_assert_close(res_mean, ref_mean, stats_dtype, reduce_dim=reduce_dim)
    utils.gems_assert_close(res_rstd, ref_rstd, stats_dtype, reduce_dim=reduce_dim)


@pytest.mark.layer_norm_backward
@pytest.mark.parametrize("shape", LAYER_NORM_SHAPES)
@pytest.mark.parametrize("wb_none", [False, True])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_layer_norm_backward(monkeypatch, shape, dtype, wb_none):
    utils.init_seed(0)

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


def check_layer_norm_reference(
    input_tensor: torch.Tensor,
    normalized_shape: tuple[int, ...],
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    epsilon: float,
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    normalized_size = math.prod(normalized_shape)
    row_count = input_tensor.numel() // normalized_size
    statistics_shape = input_tensor.shape[: -len(normalized_shape)] + (1,) * len(
        normalized_shape
    )
    assert outputs[0].shape == input_tensor.shape
    assert outputs[0].dtype == input_tensor.dtype
    assert outputs[1].shape == outputs[2].shape == statistics_shape
    statistics_dtype = (
        torch.float32 if flag_gems.vendor_name == "mthreads" else input_tensor.dtype
    )
    assert outputs[1].dtype == outputs[2].dtype == statistics_dtype
    values = input_tensor.cpu().reshape(row_count, normalized_size)
    actual = outputs[0].cpu().reshape(row_count, normalized_size)
    actual_mean, actual_rstd = (
        tensor.cpu().reshape(row_count) for tensor in outputs[1:]
    )
    scale = 1.0 if weight is None else weight.cpu().double().reshape(normalized_size)
    shift = 0.0 if bias is None else bias.cpu().double().reshape(normalized_size)
    # Chunking keeps the independent FP64 oracle bounded for the 144 MiB workload.
    for start in range(0, row_count, 512):
        chunk = values[start : start + 512].double()
        mean = chunk.mean(-1, keepdim=True)
        centered = chunk - mean
        reciprocal_std = torch.rsqrt(
            (centered * centered).mean(-1, keepdim=True) + epsilon
        )
        expected = (centered * reciprocal_std * scale + shift).to(input_tensor.dtype)
        utils.gems_assert_close(
            actual[start : start + 512], expected, input_tensor.dtype
        )
        utils.gems_assert_close(
            actual_mean[start : start + 512],
            mean.flatten().to(statistics_dtype),
            statistics_dtype,
            atol=2e-5,
        )
        utils.gems_assert_close(
            actual_rstd[start : start + 512],
            reciprocal_std.flatten().to(statistics_dtype),
            statistics_dtype,
            atol=2e-5,
        )


@pytest.mark.native_layer_norm
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize(
    "normalized_size",
    [
        1,
        3,
        127,
        128,
        255,
        256,
        768,
        1024,
        1025,
        4096,
        8193,
        16384,
        65535,
        65536,
        65537,
        393216,
        524288,
        524289,
    ],
)
@pytest.mark.parametrize(
    "pattern", ["random", "constant", "small_variance", "large_offset", "alternating"]
)
def test_layer_norm_centered_variance(
    dtype: torch.dtype, normalized_size: int, pattern: str
) -> None:
    utils.init_seed(20261102)
    host_input = torch.randn(7, normalized_size)
    if pattern == "constant":
        host_input.fill_(3)
    elif pattern == "small_variance":
        host_input = 1 + host_input * 0.001
    elif pattern == "large_offset":
        host_input = 1000 + host_input * 0.125
    elif pattern == "alternating":
        host_input = torch.where(
            torch.arange(normalized_size) % 2 == 0, 100.0, -100.0
        ).expand(7, normalized_size)
    else:
        pass
    input_tensor = host_input.to(device=flag_gems.device, dtype=dtype)
    weight = torch.randn(normalized_size).to(device=input_tensor.device, dtype=dtype)
    bias = torch.randn_like(weight)
    outputs = flag_gems.native_layer_norm(
        input_tensor, (normalized_size,), weight, bias, 1e-5
    )
    check_layer_norm_reference(
        input_tensor, (normalized_size,), weight, bias, 1e-5, outputs
    )


@pytest.mark.native_layer_norm
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("affine", ["both", "weight", "bias", "none"])
@pytest.mark.parametrize(
    "shape, normalized_shape",
    [
        ((7, 255), (255,)),
        ((2, 3, 8, 16), (8, 16)),
        ((128,), (128,)),
        ((2, 3, 4), (2, 3, 4)),
        ((0, 2, 128), (128,)),
        ((2, 3, 256, 256), (256, 256)),
    ],
)
def test_layer_norm_strided_affine(
    dtype: torch.dtype,
    affine: str,
    shape: tuple[int, ...],
    normalized_shape: tuple[int, ...],
) -> None:
    utils.init_seed(20261107)
    input_tensor = torch.randn(shape + (2,), device=flag_gems.device, dtype=dtype)[
        ..., 0
    ]
    weight = (
        torch.randn(normalized_shape + (2,), device=input_tensor.device, dtype=dtype)[
            ..., 0
        ]
        if affine in ("both", "weight")
        else None
    )
    bias = (
        torch.randn(normalized_shape + (2,), device=input_tensor.device, dtype=dtype)[
            ..., 0
        ]
        if affine in ("both", "bias")
        else None
    )
    outputs = flag_gems.native_layer_norm(
        input_tensor, normalized_shape, weight, bias, 1e-5
    )
    check_layer_norm_reference(
        input_tensor, normalized_shape, weight, bias, 1e-5, outputs
    )


@pytest.mark.layer_norm
@pytest.mark.parametrize(
    "seed, affine, epsilon, pattern",
    [
        (20261103, "both", 1e-6, "random"),
        (20261104, "both", 1e-6, "random"),
        (20261129, "none", 1e-6, "random"),
        (20261129, "none", 1e-5, "random"),
        (20261130, "float32", 1e-6, "random"),
        (20261130, "float32", 1e-5, "random"),
        (20261201, "both", 1e-6, "constant"),
        (20261201, "both", 1e-6, "large_offset"),
    ],
)
def test_layer_norm_real_startup_contract(
    seed: int, affine: str, epsilon: float, pattern: str
) -> None:
    utils.init_seed(seed)
    input_tensor = torch.randn((65536, 1, 1152)).to(torch.bfloat16)
    if pattern == "constant":
        input_tensor.fill_(3)
    elif pattern == "large_offset":
        input_tensor = (input_tensor.float() * 32 + 4096).to(torch.bfloat16)
    else:
        pass
    input_tensor = input_tensor.to(flag_gems.device)
    affine_dtype = torch.float32 if affine == "float32" else torch.bfloat16
    weight = (
        torch.randn(1152).to(device=input_tensor.device, dtype=affine_dtype)
        if affine != "none"
        else None
    )
    bias = torch.randn_like(weight) if weight is not None else None
    normalized_shape = (1152,)
    outputs = flag_gems.native_layer_norm(
        input_tensor, normalized_shape, weight, bias, epsilon
    )
    check_layer_norm_reference(
        input_tensor, normalized_shape, weight, bias, epsilon, outputs
    )
    direct_output, _, _ = flag_gems.layer_norm(
        input_tensor, normalized_shape, weight, bias, epsilon
    )
    utils.gems_assert_equal(direct_output, utils.to_reference(outputs[0]))
