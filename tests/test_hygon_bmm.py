"""Numerical and layout regressions for the Hygon small-kernel paths."""

import pytest
import torch

import flag_gems

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name != "hygon", reason="Hygon kernels"
)


@pytest.mark.bmm
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize(
    "shape",
    [
        (7, 1, 3, 10),
        (33, 3, 10, 2),
        (129, 1, 16, 48),
        (3, 5, 3, 128),
        (2, 1, 48, 7),
        (4, 8, 128, 1),
        (3, 1, 0, 4),
        (0, 1, 3, 7),
    ],
)
def test_hygon_small_bmm(dtype, shape):
    batch, m, k, n = shape
    torch.manual_seed(2301)
    a = torch.randn(batch, k, m, dtype=dtype).transpose(1, 2)
    b = torch.randn(batch, n, k, dtype=dtype).transpose(1, 2)
    reference = (
        torch.bmm(a.float(), b.float()).to(dtype)
        if dtype in (torch.float16, torch.bfloat16)
        else torch.bmm(a, b)
    )
    result = flag_gems.bmm(a.to(flag_gems.device), b.to(flag_gems.device)).cpu()
    torch.testing.assert_close(result, reference)


@pytest.mark.bmm
@pytest.mark.parametrize(
    "shape",
    [
        (3, 33, 16, 41),
        (3, 101, 16, 101),
        (3, 101, 101, 16),
        (2, 65, 3, 67),
        (3, 300, 3, 300),
        (2, 129, 8, 257),
    ],
)
@pytest.mark.parametrize("strided_output", [False, True])
def test_hygon_short_matrix_bmm_out(shape, strided_output):
    batch, m, k, n = shape
    a = torch.randn(batch, k, m).transpose(1, 2)
    b = torch.randn(batch, n, k).transpose(1, 2)
    expected = torch.bmm(a, b)
    if strided_output:
        out = torch.empty(batch, n, m, device=flag_gems.device).transpose(1, 2)
    else:
        out = torch.empty(batch, m, n, device=flag_gems.device)
    result = flag_gems.bmm_out(a.to(flag_gems.device), b.to(flag_gems.device), out)
    assert result.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out.cpu(), expected)


@pytest.mark.bmm
@pytest.mark.parametrize(
    "shape", [(7, 32, 32), (115, 48, 48), (17, 63, 57), (3, 64, 64)]
)
@pytest.mark.parametrize("layout", ["plain", "strided_input", "strided_output"])
def test_hygon_single_row_bmm(shape, layout):
    batch, k, n = shape
    generator = torch.Generator().manual_seed(6701)
    a_base = torch.randn(batch, 1, 2 * k + 1, generator=generator) * 0.25
    b_base = torch.randn(batch, k, n + 3, generator=generator) * 0.25
    if layout == "strided_input":
        a, b = a_base[:, :, 1::2], b_base[:, :, 1 : n + 1]
        ag_base, bg_base = a_base.to(flag_gems.device), b_base.to(flag_gems.device)
        ag, bg = ag_base[:, :, 1::2], bg_base[:, :, 1 : n + 1]
    else:
        a, b = a_base[:, :, :k].contiguous(), b_base[:, :, :n].contiguous()
        ag, bg = a.to(flag_gems.device), b.to(flag_gems.device)
    expected = torch.bmm(a, b)
    if layout == "strided_output":
        out = torch.empty(batch, 1, 2 * n, device=flag_gems.device)[:, :, ::2]
        actual = flag_gems.bmm_out(ag, bg, out)
        assert actual.data_ptr() == out.data_ptr()
    else:
        actual = flag_gems.bmm(ag, bg)
    torch.testing.assert_close(actual.cpu(), expected)


"""Regression coverage for expanded Hygon layouts and precise reductions."""


pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name != "hygon", reason="Hygon kernels"
)


@pytest.mark.bmm
def test_hygon_bmm_uses_specialized_export():
    from flag_gems.runtime.backend._hygon.ops.bmm_specialized import bmm

    assert flag_gems.bmm is bmm


@pytest.mark.bmm
def test_hygon_bmm_out_overlap_uses_upstream_copy():
    a = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4) / 16
    b = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4) / 16
    expected = torch.bmm(a, b)
    ad, bd = a.to(flag_gems.device), b.to(flag_gems.device)
    result = flag_gems.bmm_out(ad, bd, out=ad)
    assert result is ad
    torch.testing.assert_close(result.cpu(), expected)
