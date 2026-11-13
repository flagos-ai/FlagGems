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
    reason="requires a registered unpadded grouped FP8 GEMM backend",
)


def make_grouped_fp8_case(
    lengths: list[int], n: int, k: int, device: str, column_major_scales: bool = False
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(42)
    m, experts = sum(lengths), len(lengths)
    a = torch.randn((m, k), device=device).to(torch.float8_e4m3fn)
    b = torch.randn((experts, n, k), device=device).to(torch.float8_e4m3fn)
    if column_major_scales:
        sa = (0.01 + 0.02 * torch.rand((k // 128, m), device=device)).T
    else:
        sa = 0.01 + 0.02 * torch.rand((m, k // 128), device=device)
    sb = 0.01 + 0.02 * torch.rand(
        (experts, triton.cdiv(n, 128), k // 128), device=device
    )
    rows = torch.tensor(lengths, dtype=torch.int32, device=device)
    indices = torch.repeat_interleave(
        torch.arange(experts, dtype=torch.int32, device=device), rows, output_size=m
    )
    out = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device=device)
    return a, sa, b, sb, out, indices, rows


def mate_grouped_fp8(a, sa, b, sb, out, m_indices, m_rows):
    from mate.gemm import ragged_m_moe_gemm_8bit

    ragged_m_moe_gemm_8bit(
        (a, sa), (b, sb), m_rows, out, gemm_mode="per_expert", backend="mubin"
    )


def grouped_fp8_cpu_reference(a, sa, b, sb, out, lengths):
    a, sa, b, sb = (t.cpu().double() for t in (a, sa, b, sb))
    reference = torch.empty_like(out, device="cpu")
    n, k = b.shape[1:]
    start = 0
    for expert, length in enumerate(lengths):
        rows = slice(start, start + length)
        acc = torch.zeros((length, n), dtype=torch.float64)
        for block in range(k // 128):
            section = slice(block * 128, (block + 1) * 128)
            dot = a[rows, section] @ b[expert, :, section].T
            bs = sb[expert, :, block].repeat_interleave(128)[:n]
            acc += dot * sa[rows, block, None] * bs[None, :]
        reference[rows] = acc.bfloat16()
        start += length
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
        ([1] * 8 + [0] * 24, 256, 6144),
        ([1, 273, 512], 256, 6144),
        ([1, 273, 512], 512, 128),
        ([1, 0, 7], 192, 384),
        ([1, 63, 65], 192, 128),
    ],
)
def test_grouped_gemm_w8a8_fp8(lengths, n, k, column_major):
    args = make_grouped_fp8_case(lengths, n, k, flag_gems.device, column_major)
    expected = grouped_fp8_cpu_reference(*args[:5], lengths)
    # Leave real rows directly adjacent to BF16 sentinels on both sides.
    backing = torch.full(
        (sum(lengths) + 2, n), 7.0, device=flag_gems.device, dtype=torch.bfloat16
    )
    out = backing[1:-1]
    out.fill_(float("nan"))
    assert flag_gems.grouped_gemm_w8a8_fp8(*args[:4], out, *args[5:]) is None
    gems_assert_close(out.cpu(), expected, torch.bfloat16)
    guards = backing.cpu()[[0, -1]]
    gems_assert_equal(guards, torch.full_like(guards, 7.0))


@pytest.mark.grouped_gemm_w8a8_fp8
@pytest.mark.parametrize("n,k", [(256, 6144), (6144, 128)])
def test_grouped_gemm_w8a8_fp8_native(n, k):
    pytest.importorskip("mate.gemm")
    args = make_grouped_fp8_case([1, 0, 17, 129], n, k, flag_gems.device)
    expected = args[4].clone()
    mate_grouped_fp8(*args[:4], expected, *args[5:])
    flag_gems.grouped_gemm_w8a8_fp8(*args)
    gems_assert_close(args[4], expected, torch.bfloat16)


@pytest.mark.grouped_gemm_w8a8_fp8
@pytest.mark.parametrize("n,k", [(256, 512), (512, 128)])
def test_grouped_gemm_w8a8_fp8_changing_routes(n, k):
    args = make_grouped_fp8_case([257, 0, 513], n, k, flag_gems.device)
    for lengths in ([257, 0, 513], [0, 769, 1], [770, 0, 0], [257, 0, 513]):
        args[6].copy_(torch.tensor(lengths, device=args[6].device, dtype=torch.int32))
        args[5].copy_(
            torch.repeat_interleave(
                torch.arange(3, dtype=torch.int32, device=args[5].device),
                args[6],
                output_size=770,
            )
        )
        args[4].fill_(float("nan"))
        expected = grouped_fp8_cpu_reference(*args[:5], lengths)
        flag_gems.grouped_gemm_w8a8_fp8(*args)
        gems_assert_close(args[4].cpu(), expected, torch.bfloat16)
