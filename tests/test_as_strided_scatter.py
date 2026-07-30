# Copyright 2026 FlagOS Contributors

import pytest
import torch

import flag_gems


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "size, stride, storage_offset, self_len",
    [((2, 3), (3, 1), 0, 6), ((4, 4), (10, 2), 3, 40), ((100,), (2,), 1, 200)],
)
def test_as_strided_scatter(dtype, size, stride, storage_offset, self_len):
    inp = torch.randn(self_len, device="cuda", dtype=dtype)
    src = torch.randn(size, device="cuda", dtype=dtype)
    expected = torch.ops.aten.as_strided_scatter(inp, src, size, stride, storage_offset)

    with flag_gems.use_gems(include=["as_strided_scatter"]):
        actual = torch.ops.aten.as_strided_scatter(
            inp, src, size, stride, storage_offset
        )

    torch.testing.assert_close(actual, expected)


def test_as_strided_scatter_noncontiguous_fallback_and_out():
    inp = torch.randn((5, 7), device="cuda").mT
    src = torch.randn((2, 2), device="cuda")
    expected = torch.ops.aten.as_strided_scatter(inp, src, (2, 2), (1, 5), None)
    out = torch.empty(0, device="cuda")

    with flag_gems.use_gems(include=["as_strided_scatter_out"]):
        result = torch.ops.aten.as_strided_scatter.out(
            inp, src, (2, 2), (1, 5), None, out=out
        )

    assert result is out
    torch.testing.assert_close(out, expected)
