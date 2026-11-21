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

import flag_gems

pytestmark = [
    pytest.mark.sum_dim,
    pytest.mark.skipif(
        flag_gems.vendor_name != "hygon", reason="Hygon short reduction"
    ),
]


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((126636, 4, 32), torch.float32),
        ((2806, 4, 32), torch.float32),
        ((61, 1896, 3), torch.float64),
        ((96380, 10), torch.float64),
        ((854, 854, 3), torch.float32),
        ((8193, 3), torch.float32),
        ((8193, 7), torch.float32),
        ((8193, 7), torch.float64),
        ((8193, 10), torch.float64),
        ((8193, 32), torch.float64),
        ((2002, 2, 2), torch.float32),
        ((35, 1088, 2), torch.float64),
        ((12709, 2, 3), torch.float32),
        ((25132, 3), torch.float64),
        ((265, 265, 2), torch.float32),
        ((842, 842, 3), torch.float32),
        ((142170, 15), torch.float64),
        ((50836, 2, 13), torch.float32),
        ((111, 3454, 5), torch.float64),
    ],
)
@pytest.mark.parametrize("keepdim", [False, True])
def test_sum_hygon_short(shape, dtype, keepdim):
    generator = torch.Generator().manual_seed(20261105)
    cpu = torch.randn(shape, dtype=dtype, generator=generator)
    inp = cpu.to(flag_gems.device)
    result = flag_gems.sum_dim(inp, [-1], keepdim=keepdim)
    expected = cpu.to(torch.float64).sum(-1, keepdim=keepdim)
    bound = cpu.abs().to(torch.float64).sum(-1, keepdim=keepdim)
    error = (result.cpu().to(torch.float64) - expected).abs()
    assert result.shape == expected.shape
    assert result.dtype == dtype
    assert torch.all(error <= 8 * torch.finfo(dtype).eps * bound.clamp_min(1))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize(
    "shape,stride,offset",
    [
        ((8193, 3), (6, 2), 3),
        ((8193, 7), (16, 2), 5),
        ((8193, 3), (1, 8193), 0),
        ((8193, 3), (0, 1), 2),
        ((8193, 3), (1, 0), 1),
        ((8193, 2), (8, 2), 3),
        ((8193, 5), (12, 2), 3),
        ((8193, 13), (28, 2), 3),
        ((8193, 15), (32, 2), 3),
    ],
)
def test_sum_hygon_short_strided(shape, stride, offset, dtype, keepdim):
    size = offset + sum((n - 1) * s for n, s in zip(shape, stride)) + 1
    storage = torch.randn(
        (size,), dtype=dtype, generator=torch.Generator().manual_seed(20261109)
    )
    cpu = storage.as_strided(shape, stride, offset)
    inp = storage.to(flag_gems.device).as_strided(shape, stride, offset)
    assert inp.stride() == stride
    assert inp.storage_offset() == offset
    result = flag_gems.sum_dim(inp, [-1], keepdim=keepdim)
    expected = cpu.to(torch.float64).sum(-1, keepdim=keepdim)
    bound = cpu.abs().to(torch.float64).sum(-1, keepdim=keepdim).clamp_min(1)
    assert result.shape == expected.shape
    assert result.dtype == dtype
    assert torch.all(
        (result.cpu().to(torch.float64) - expected).abs()
        <= 8 * torch.finfo(dtype).eps * bound
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sum_hygon_short_nonfinite(dtype):
    cpu = torch.ones((8193, 3), dtype=dtype)
    cpu[0, 0] = float("nan")
    cpu[1, 1] = float("inf")
    cpu[2, 2] = -float("inf")
    cpu[3, 0] = float("inf")
    cpu[3, 1] = -float("inf")
    actual = flag_gems.sum_dim(cpu.to(flag_gems.device), [-1]).cpu()
    torch.testing.assert_close(actual, cpu.sum(-1), equal_nan=True)


@pytest.mark.parametrize("m", [1023, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sum_hygon_width_two_boundary(m, dtype):
    cpu = torch.randn((m, 2), dtype=dtype)
    actual = flag_gems.sum_dim(cpu.to(flag_gems.device), [-1]).cpu()
    torch.testing.assert_close(actual, cpu.sum(-1))


@pytest.mark.parametrize("kind", ["small", "strided", "dtype", "dim"])
def test_sum_hygon_short_fallback(kind):
    cpu = torch.randn((8193, 3), generator=torch.Generator().manual_seed(42))
    dim, dtype = [-1], None
    if kind == "small":
        cpu = cpu[:31]
    elif kind == "strided":
        cpu = cpu.t()
    elif kind == "dtype":
        dtype = torch.float64
    elif kind == "dim":
        dim = [0]
    actual = flag_gems.sum_dim(cpu.to(flag_gems.device), dim, dtype=dtype).cpu()
    expected = cpu.sum(dim, dtype=dtype)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("n", [16384, 262144, 262145, 2097152])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("keepdim", [False, True])
def test_sum_hygon_split_row(n, dtype, keepdim):
    # Preserve the workbook's noncanonical singleton stride and a nonzero offset.
    storage = torch.randn(
        n + 3, dtype=dtype, generator=torch.Generator().manual_seed(20261110)
    )
    cpu = storage.as_strided((1, n), (3, 1), 3)
    inp = storage.to(flag_gems.device).as_strided((1, n), (3, 1), 3)
    result = flag_gems.sum_dim(inp, [-1], keepdim=keepdim)
    expected = cpu.to(torch.float64).sum(-1, keepdim=keepdim)
    bound = cpu.abs().to(torch.float64).sum(-1, keepdim=keepdim).clamp_min(1)
    assert result.shape == expected.shape
    assert result.dtype == dtype
    assert torch.all(
        (result.cpu().to(torch.float64) - expected).abs()
        <= 8 * torch.finfo(dtype).eps * bound
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sum_hygon_split_row_nonfinite(dtype):
    cpu = torch.ones((4, 262144), dtype=dtype)
    cpu[0, 0] = float("nan")
    cpu[1, 0] = float("inf")
    cpu[2, 0] = -float("inf")
    cpu[3, 0], cpu[3, -1] = float("inf"), -float("inf")
    for row in cpu:
        actual = flag_gems.sum_dim(row.to(flag_gems.device), [-1]).cpu()
        torch.testing.assert_close(actual, row.sum(-1), equal_nan=True)


@pytest.mark.parametrize("shape", [(3, 120, 99, 99), (3, 105, 87, 87)])
@pytest.mark.parametrize("dims", [[-1, -2], [2, 3], [3, -2]])
@pytest.mark.parametrize("keepdim", [False, True])
def test_sum_hygon_trailing_pair(shape, dims, keepdim):
    cpu = torch.randn(shape, generator=torch.Generator().manual_seed(20261110))
    expected = cpu.double().sum(dims, keepdim=keepdim).float()
    actual = flag_gems.sum_dim(cpu.to(flag_gems.device), dims, keepdim=keepdim).cpu()
    torch.testing.assert_close(actual, expected)
