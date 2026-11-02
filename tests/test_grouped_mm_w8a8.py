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

from .accuracy_utils import gems_assert_equal

pytestmark = pytest.mark.skipif(
    "grouped_mm_w8a8" not in vars(flag_gems),
    reason="requires a registered grouped_mm_w8a8 implementation",
)


def make_grouped_gemm_case(lengths, n, k, device, seed=20261019):
    torch.manual_seed(seed)
    m, e = sum(lengths), len(lengths)
    A = torch.randint(-128, 128, (m, k), dtype=torch.int8, device=device)
    B = torch.randint(-128, 128, (e, n, k), dtype=torch.int8, device=device)
    As = 0.005 + 0.01 * torch.rand((m, 1), device=device)
    Bs = 0.005 + 0.01 * torch.rand((e, n, 1), device=device)
    C = torch.empty((m, n), dtype=torch.bfloat16, device=device)
    rows = torch.tensor(lengths, dtype=torch.int32, device=device)
    indices = torch.repeat_interleave(
        torch.arange(e, dtype=torch.int32, device=device), rows, output_size=m
    )
    return A, As, B, Bs, C, indices, rows


def grouped_gemm_cpu_reference(A, As, B, Bs, lengths):
    A, As, B, Bs = (t.cpu() for t in (A, As, B, Bs))
    output = torch.empty((A.shape[0], B.shape[1]), dtype=torch.bfloat16)
    start = 0
    for expert, count in enumerate(lengths):
        end = start + count
        dot = A[start:end].double() @ B[expert].double().T
        output[start:end] = (dot.float() * As[start:end] * Bs[expert, :, 0]).bfloat16()
        start = end
    return output


@pytest.mark.parametrize(
    "lengths,n,k",
    [
        ([0, 1, 3, 0, 7], 65, 129),
        ([1, 32, 0, 65], 128, 256),
        ([0, 257, 3], 64, 128),
        ([1] + [0] * 255 + [13], 64, 6144),
        ([0, 129, 0, 3968], 257, 2051),
        ([0, 129] + [0] * 254 + [4000], 257, 2051),
        ([0, 1, 0], 17, 1),
        ([0, 0, 0], 64, 128),
    ],
)
@pytest.mark.grouped_mm_w8a8
def test_grouped_mm_w8a8(lengths, n, k):
    args = make_grouped_gemm_case(lengths, n, k, flag_gems.device)
    expected = grouped_gemm_cpu_reference(*args[:4], lengths)
    pointer = args[4].data_ptr()
    assert flag_gems.grouped_mm_w8a8(*args) is None
    assert args[4].data_ptr() == pointer
    gems_assert_equal(args[4].cpu(), expected)


@pytest.mark.grouped_mm_w8a8
def test_grouped_mm_w8a8_int8_extremes():
    args = make_grouped_gemm_case([1, 0, 1], 65, 6144, flag_gems.device)
    args[0].fill_(-128)
    args[2][0].fill_(-128)
    args[2][2].fill_(127)
    expected = grouped_gemm_cpu_reference(*args[:4], [1, 0, 1])
    pointer = args[4].data_ptr()
    assert flag_gems.grouped_mm_w8a8(*args) is None
    assert args[4].data_ptr() == pointer
    gems_assert_equal(args[4].cpu(), expected)


@pytest.mark.grouped_mm_w8a8
def test_grouped_mm_w8a8_changing_expert_rows():
    for lengths, n, k in (
        ([0, 257, 0, 511], 64, 128),
        ([256, 0, 512, 0], 128, 256),
        ([256, 0, 512, 0], 64, 128),
        ([0, 257, 0, 511], 128, 256),
    ):
        args = make_grouped_gemm_case(lengths, n, k, flag_gems.device)
        expected = grouped_gemm_cpu_reference(*args[:4], lengths)
        flag_gems.grouped_mm_w8a8(*args)
        gems_assert_equal(args[4].cpu(), expected)


@pytest.mark.grouped_mm_w8a8
@pytest.mark.parametrize(
    "dtype", [torch.int8, torch.float8_e4m3fn, torch.float8_e4m3fnuz]
)
@pytest.mark.parametrize(
    "lengths,n,k",
    [
        ([0, 3, 1], 17, 0),
        ([0, 3, 1], 0, 16),
        ([0, 0], 16, 16),
        ([0, 17, 65, 0, 131, 1], 65, 129),
        ([0, 257, 3], 128, 256),
        ([0, 513, 0], 256, 256),
    ],
)
def test_grouped_mm_w8a8_dtypes_tails(lengths, n, k, dtype):
    args = list(make_grouped_gemm_case(lengths, n, k, flag_gems.device))
    if dtype != torch.int8:
        for index in (0, 2):
            args[index] = (
                (args[index].cpu().float() / 64).to(dtype).to(flag_gems.device)
            )
    expected = grouped_gemm_cpu_reference(*args[:4], lengths)
    args[4].fill_(float("nan"))
    assert flag_gems.grouped_mm_w8a8(*args) is None
    if dtype == torch.int8:
        gems_assert_equal(args[4].cpu(), expected)
    else:
        # FP32 dot reduction order may change BF16 rounding near cancellation.
        # Permit one BF16 relative rounding step, with no absolute tolerance.
        torch.testing.assert_close(
            args[4].cpu(), expected, rtol=torch.finfo(torch.bfloat16).eps, atol=0
        )


@pytest.mark.grouped_mm_w8a8
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e4m3fnuz])
def test_grouped_mm_w8a8_fp8_encodings(dtype):
    bits = torch.arange(256, dtype=torch.int16).to(torch.uint8)
    if dtype == torch.float8_e4m3fnuz:
        bits[bits == 128] = 0
    else:
        bits[(bits & 127) == 127] = 0
    args = list(make_grouped_gemm_case([0, 16, 0], 16, 16, flag_gems.device))
    args[0] = bits.view(dtype).reshape(16, 16).to(flag_gems.device)
    args[2] = torch.eye(16).repeat(3, 1, 1).to(dtype).to(flag_gems.device)
    args[1].fill_(1)
    args[3].fill_(1)
    expected = args[0].cpu().float().bfloat16()
    flag_gems.grouped_mm_w8a8(*args)
    gems_assert_equal(args[4].cpu(), expected)


@pytest.mark.grouped_mm_w8a8
def test_grouped_mm_w8a8_int8_cancellation():
    args = make_grouped_gemm_case([0, 17, 65, 0, 131, 1], 64, 6144, flag_gems.device)
    args[0].fill_(127)
    args[2].fill_(127)
    args[0][:, 3072:] = -127
    args[0][:, 2048] = 126
    expected = grouped_gemm_cpu_reference(*args[:4], [0, 17, 65, 0, 131, 1])
    flag_gems.grouped_mm_w8a8(*args)
    gems_assert_equal(args[4].cpu(), expected)


@pytest.mark.grouped_mm_w8a8
@pytest.mark.parametrize("n,k", [(65, 2051), (1025, 129), (1025, 513)])
def test_grouped_mm_w8a8_sparse_changing_groups(n, k):
    # Compact/noncompact dispatch boundaries, output/K tails, and a group
    # spanning several M tiles. Reuse metadata storage across different routes.
    lengths = [0] * 257
    lengths[0], lengths[17], lengths[256] = 1, 3, 33
    args = make_grouped_gemm_case(lengths, n, k, flag_gems.device)
    for counts in ((1, 3, 33), (33, 1, 3)):
        lengths[0], lengths[17], lengths[256] = counts
        args[6].copy_(torch.tensor(lengths, dtype=torch.int32, device=flag_gems.device))
        args[5].copy_(
            torch.repeat_interleave(
                torch.arange(257, dtype=torch.int32, device=flag_gems.device),
                args[6],
                output_size=sum(lengths),
            )
        )
        expected = grouped_gemm_cpu_reference(*args[:4], lengths)
        args[4].fill_(float("nan"))
        assert flag_gems.grouped_mm_w8a8(*args) is None
        gems_assert_equal(args[4].cpu(), expected)
