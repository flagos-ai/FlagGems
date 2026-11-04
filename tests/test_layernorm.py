# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

import flag_gems
from flag_gems.testing.layernorm_shapes import (
    LEGACY_LAYERNORM_CASES,
    UNSUPPORTED_LEGACY_LAYERNORM_CASES,
)

pytestmark = [
    pytest.mark.layernorm,
    pytest.mark.skipif(
        flag_gems.vendor_name != "metax", reason="C550-specific implementation"
    ),
]


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
    actual = flag_gems.layernorm(tensor, norm, weight, bias, 1e-6)
    tolerance = (
        1e-6 if dtype == torch.float32 else (1e-2 if dtype == torch.bfloat16 else 1e-3)
    )
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    assert actual.shape == tensor.shape and actual.dtype == tensor.dtype


@pytest.mark.parametrize("layout", ["transpose", "slice", "expand"])
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
        flag_gems.layernorm(tensor, (5, 7), weight, bias),
        F.layer_norm(tensor, (5, 7), weight, bias),
        atol=2e-6,
        rtol=2e-6,
    )


@pytest.mark.parametrize(
    "value", [0.0, -0.0, float("nan"), float("inf"), -float("inf"), 10000.0]
)
def test_layernorm_special_values(value):
    tensor = torch.full((3, 1152), value, device="cuda")
    tensor[1, 0] = 1.0
    torch.testing.assert_close(
        flag_gems.layernorm(tensor, (1152,)),
        F.layer_norm(tensor, (1152,)),
        atol=2e-3,
        rtol=2e-3,
        equal_nan=True,
    )


@pytest.mark.layernorm
def test_layernorm_real_shape():
    torch.manual_seed(77)
    tensor = torch.randn(65536, 1, 1152, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(1152, device="cuda", dtype=tensor.dtype)
    bias = torch.randn_like(weight)
    torch.testing.assert_close(
        flag_gems.layernorm(tensor, (1152,), weight, bias, 1e-6),
        F.layer_norm(tensor, (1152,), weight, bias, 1e-6),
        atol=1e-2,
        rtol=1e-2,
    )


def test_layernorm_errors():
    tensor = torch.empty(3, 1152, device="cuda")
    with pytest.raises(ValueError):
        flag_gems.layernorm(tensor, (1024,))
    with pytest.raises(ValueError):
        flag_gems.layernorm(
            tensor, (1152,), torch.empty(1152, device="cuda", dtype=torch.bfloat16)
        )
    with pytest.raises(NotImplementedError):
        flag_gems.layernorm(tensor.double(), (1152,))
    with pytest.raises(NotImplementedError):
        flag_gems.layernorm(tensor.requires_grad_(), (1152,))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "width", [320, 384, 576, 768, 1152, 1280, 1536, 2304, 2560, 3072]
)
@pytest.mark.parametrize("affine", ["none", "weight", "bias", "both"])
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
    actual = flag_gems.layernorm(tensor, (width,), weight, bias, 1e-6)
    expected = F.layer_norm(tensor, (width,), weight, bias, 1e-6)
    tolerance = (
        1e-6 if dtype == torch.float32 else (1e-2 if dtype == torch.bfloat16 else 1e-3)
    )
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.layernorm
@pytest.mark.parametrize("shape,normalized_shape", LEGACY_LAYERNORM_CASES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("affine", [False, True])
def test_layernorm_legacy_shapes(shape, normalized_shape, dtype, affine):
    torch.manual_seed(20261104)
    tensor = torch.randn(shape, dtype=dtype, device="cuda")
    weight = (
        torch.randn(normalized_shape, dtype=dtype, device="cuda") if affine else None
    )
    bias = torch.randn_like(weight) if affine else None
    actual = flag_gems.layernorm(tensor, normalized_shape, weight, bias, 1e-5)
    expected = F.layer_norm(tensor, normalized_shape, weight, bias, 1e-5)
    tolerance = (
        1e-6 if dtype == torch.float32 else (1e-2 if dtype == torch.bfloat16 else 1e-3)
    )
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.layernorm
@pytest.mark.parametrize("shape,normalized_shape", UNSUPPORTED_LEGACY_LAYERNORM_CASES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_layernorm_legacy_unsupported_extents(shape, normalized_shape, dtype):
    tensor = torch.empty(shape, dtype=dtype, device="cuda")
    with pytest.raises(NotImplementedError, match="above 65536"):
        flag_gems.layernorm(tensor, normalized_shape)
