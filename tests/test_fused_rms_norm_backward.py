# Copyright 2026 FlagOS Contributors

import pytest
import torch

import flag_gems


@pytest.mark.parametrize(
    "shape, normalized_shape", [((3, 8), (8,)), ((2, 3, 4), (3, 4))]
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "output_mask", [(True, True), (True, False), (False, True), (False, False)]
)
def test_fused_rms_norm_backward(shape, normalized_shape, dtype, output_mask):
    inp = torch.randn(shape, device="cuda", dtype=dtype)
    weight = torch.randn(normalized_shape, device="cuda", dtype=dtype)
    grad = torch.randn_like(inp)
    reduce_dims = tuple(range(inp.ndim - len(normalized_shape), inp.ndim))
    rstd = torch.rsqrt(inp.float().pow(2).mean(dim=reduce_dims) + 1e-5)
    expected = torch.ops.aten._fused_rms_norm_backward(
        grad, inp, normalized_shape, rstd, weight, output_mask
    )

    with flag_gems.use_gems(include=["_fused_rms_norm_backward"]):
        actual = torch.ops.aten._fused_rms_norm_backward(
            grad, inp, normalized_shape, rstd, weight, output_mask
        )

    for result, reference in zip(actual, expected):
        if reference is None:
            assert result is None
        else:
            torch.testing.assert_close(result, reference, rtol=3e-2, atol=3e-2)


def test_fused_rms_norm_backward_without_weight():
    inp = torch.randn((3, 8), device="cuda")
    grad = torch.randn_like(inp)
    rstd = torch.rsqrt(inp.pow(2).mean(dim=-1) + 1e-5)
    expected = torch.ops.aten._fused_rms_norm_backward(
        grad, inp, [8], rstd, None, [True, False]
    )
    with flag_gems.use_gems(include=["_fused_rms_norm_backward"]):
        actual = torch.ops.aten._fused_rms_norm_backward(
            grad, inp, [8], rstd, None, [True, False]
        )
    torch.testing.assert_close(actual[0], expected[0])
    assert actual[1] is None
