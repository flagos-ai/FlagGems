# Copyright 2026 FlagOS Contributors

import pytest
import torch

import flag_gems


@pytest.mark.parametrize(
    "shape, output_size",
    [((2, 3, 7, 9), (3, 4)), ((1, 2, 8, 8), (1, 1)), ((3, 5, 6), (2, 4))],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_adaptive_max_pool2d(shape, output_size, dtype):
    inp = torch.randn(shape, device="cuda", dtype=dtype)
    expected, expected_indices = torch.ops.aten.adaptive_max_pool2d(inp, output_size)

    with flag_gems.use_gems(include=["adaptive_max_pool2d"]):
        actual, actual_indices = torch.ops.aten.adaptive_max_pool2d(inp, output_size)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_indices, expected_indices)


def test_adaptive_max_pool2d_out():
    inp = torch.randn((2, 3, 7, 9), device="cuda")
    expected, expected_indices = torch.ops.aten.adaptive_max_pool2d(inp, (3, 4))
    actual = torch.empty(0, device="cuda")
    actual_indices = torch.empty(0, device="cuda", dtype=torch.int64)

    with flag_gems.use_gems(include=["adaptive_max_pool2d_out"]):
        result, result_indices = torch.ops.aten.adaptive_max_pool2d.out(
            inp, (3, 4), out=actual, indices=actual_indices
        )

    assert result is actual
    assert result_indices is actual_indices
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_indices, expected_indices)
