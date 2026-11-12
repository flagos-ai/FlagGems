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
import triton

import flag_gems

from .accuracy_utils import gems_assert_close, gems_assert_equal

pytestmark = pytest.mark.skipif(
    "grouped_gemm_w8a8_fp8" not in vars(flag_gems),
    reason="requires a registered grouped FP8 contiguous GEMM backend",
)


def make_grouped_fp8_case(
    lengths: list[int], n: int, k: int, device: str, column_major_scales: bool = False
) -> tuple[
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]:
    torch.manual_seed(42)
    padded = [triton.cdiv(length, 256) * 256 for length in lengths]
    m, experts = sum(padded), len(lengths)
    a = torch.randn((m, k), device=device).to(torch.float8_e4m3fn)
    b = torch.randn((experts, n, k), device=device).to(torch.float8_e4m3fn)
    if column_major_scales:
        sa = (0.01 + 0.02 * torch.rand((k // 128, m), device=device)).T
    else:
        sa = 0.01 + 0.02 * torch.rand((m, k // 128), device=device)
    sb = 0.01 + 0.02 * torch.rand(
        (experts, triton.cdiv(n, 128), k // 128), device=device
    )
    groups = []
    for expert, (length, size) in enumerate(zip(lengths, padded)):
        groups.extend([expert] * length + [-1] * (size - length))
    indices = torch.tensor(groups, dtype=torch.int32, device=device)
    out = torch.full((m, n), 7.0, dtype=torch.bfloat16, device=device)
    return (a, sa), (b, sb), out, indices


def deepgemm_grouped_fp8(
    lhs: tuple[torch.Tensor, torch.Tensor],
    rhs: tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    m_indices: torch.Tensor,
    *,
    alignment_m: int = 256,
) -> None:
    import deep_gemm

    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        lhs, rhs, out, m_indices, alignment_m=alignment_m
    )


def grouped_fp8_cpu_reference(
    lhs: tuple[torch.Tensor, torch.Tensor],
    rhs: tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    m_indices: torch.Tensor,
) -> torch.Tensor:
    a, sa = (t.cpu().double() for t in lhs)
    b, sb = (t.cpu().double() for t in rhs)
    indices = m_indices.cpu()
    reference = out.cpu().clone()
    n, k = b.shape[1:]
    for expert in range(b.shape[0]):
        rows = indices == expert
        acc = torch.zeros((int(rows.sum()), n), dtype=torch.float64)
        for block in range(k // 128):
            section = slice(block * 128, (block + 1) * 128)
            dot = a[rows, section] @ b[expert, :, section].T
            bs = sb[expert, :, block].repeat_interleave(128)[:n]
            acc += dot * sa[rows, block, None] * bs[None, :]
        reference[rows] = acc.bfloat16()
    return reference


@pytest.mark.grouped_gemm_w8a8_fp8
@pytest.mark.parametrize("column_major", [False, True])
@pytest.mark.parametrize(
    "lengths,n,k",
    [
        ([1, 0, 7], 64, 128),
        ([63, 65, 0, 129], 192, 256),
        ([0, 0], 128, 128),
        ([7, 3], 256, 6144),
        ([1, 3], 6144, 128),
        ([1, 129], 512, 1024),
    ],
)
def test_grouped_gemm_w8a8_fp8(lengths, n, k, column_major):
    args = make_grouped_fp8_case(lengths, n, k, flag_gems.device, column_major)
    expected = grouped_fp8_cpu_reference(*args)
    flag_gems.grouped_gemm_w8a8_fp8(*args, alignment_m=256)
    gems_assert_close(args[2].cpu(), expected, torch.bfloat16)


@pytest.mark.grouped_gemm_w8a8_fp8
@pytest.mark.parametrize("n,k", [(256, 6144), (6144, 128), (192, 256)])
def test_grouped_gemm_w8a8_fp8_native(n, k):
    pytest.importorskip("deep_gemm")
    args = make_grouped_fp8_case([1, 0, 17, 129], n, k, flag_gems.device)
    expected = args[2].clone()
    deepgemm_grouped_fp8(*args[:2], expected, args[3])
    flag_gems.grouped_gemm_w8a8_fp8(*args, alignment_m=256)
    valid = args[3] >= 0
    gems_assert_close(args[2][valid], expected[valid], torch.bfloat16)


@pytest.mark.grouped_gemm_w8a8_fp8
def test_grouped_gemm_w8a8_fp8_changing_routes():
    args = make_grouped_fp8_case([17, 31], 128, 512, flag_gems.device)
    original = args[3].clone()
    routes = (
        original,
        torch.full_like(original, -1),
        torch.full_like(original, 1),
        original,
    )
    for routing in routes:
        args[3].copy_(routing)
        expected = grouped_fp8_cpu_reference(*args)
        flag_gems.grouped_gemm_w8a8_fp8(*args, alignment_m=256)
        gems_assert_close(args[2].cpu(), expected, torch.bfloat16)


@pytest.mark.grouped_gemm_w8a8_fp8
def test_grouped_gemm_w8a8_fp8_padding():
    args = make_grouped_fp8_case([1, 65], 128, 256, flag_gems.device)
    flag_gems.grouped_gemm_w8a8_fp8(*args, alignment_m=256)
    padding = args[3] < 0
    gems_assert_equal(args[2][padding], torch.full_like(args[2][padding], 7.0))
