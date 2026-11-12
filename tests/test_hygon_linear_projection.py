import pytest
import torch

import flag_gems


@pytest.mark.addmm
@pytest.mark.skipif(
    flag_gems.vendor_name != "hygon",
    reason="The Hygon public addmm entry owns the K1 specialization",
)
@pytest.mark.parametrize("N", [1, 17, 257, 1025])
@pytest.mark.parametrize("cancellation", [False, True])
def test_addmm_single_row_k1_strided_cpu_reference(N, cancellation):
    from flag_gems.runtime.backend._hygon.ops.addmm import addmm

    assert flag_gems.addmm is addmm

    generator = torch.Generator().manual_seed(20261109 + N)
    backing = [
        torch.rand(8, generator=generator) * 2 - 1,
        torch.rand(N * 3 + 7, generator=generator) * 2 - 1,
        torch.rand(N * 2 + 9, generator=generator) * 2 - 1,
    ]
    mat1 = backing[0].as_strided((1, 1), (3, 2), 2)
    mat2 = backing[1].as_strided((1, N), (1, 3), 3)
    bias = backing[2].as_strided((N,), (2,), 5)
    if cancellation:
        bias.copy_(-(mat1[0, 0] * mat2[0]))
    expected = torch.addmm(bias, mat1, mat2)

    device_backing = [x.to(flag_gems.device) for x in backing]
    device_mat1 = device_backing[0].as_strided((1, 1), (3, 2), 2)
    device_mat2 = device_backing[1].as_strided((1, N), (1, 3), 3)
    device_bias = device_backing[2].as_strided((N,), (2,), 5)
    actual = flag_gems.addmm(device_bias, device_mat1, device_mat2)

    torch.testing.assert_close(actual.cpu(), expected)
    assert actual.shape == (1, N)
    for original, device in zip(backing, device_backing):
        torch.testing.assert_close(device.cpu(), original, rtol=0, atol=0)
        assert not torch._C._is_alias_of(actual, device)


@pytest.mark.mm
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_hygon_projection_keeps_general_mm_route(dtype):
    from flag_gems.runtime.backend._hygon.ops.linear_projection import (
        try_linear_projection,
    )

    a = torch.ones((8, 8), device=flag_gems.device, dtype=dtype)
    b = a.t()
    assert try_linear_projection(a, b) is None
    actual = flag_gems.mm(a, b)
    torch.testing.assert_close(actual.cpu(), torch.full((8, 8), 8, dtype=dtype))
