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

from .accuracy_utils import gems_assert_close

pytestmark = [
    pytest.mark.grouped_gemm_w8a8_fp8,
    pytest.mark.skipif(
        not hasattr(flag_gems, "grouped_gemm_w8a8_fp8"),
        reason="requires a registered grouped FP8 GEMM implementation",
    ),
]


def make_case(lengths, n, k, strided=False):
    torch.manual_seed(42)
    m, e = sum(lengths), len(lengths)
    device = flag_gems.device
    a = torch.randn((m, k), device=device).to(torch.float8_e4m3fn)
    b = torch.randn((e, n, k), device=device).to(torch.float8_e4m3fn)
    sa = 0.01 + 0.02 * torch.rand((m, k // 128), device=device)
    sb = 0.01 + 0.02 * torch.rand((e, (n + 127) // 128, k // 128), device=device)
    if strided:
        sa = sa.T.contiguous().T
        sb = sb.transpose(1, 2).contiguous().transpose(1, 2)
    counts = torch.tensor(lengths, dtype=torch.int32, device=device)
    indices = torch.repeat_interleave(
        torch.arange(e, dtype=torch.int32, device=device), counts, output_size=m
    )
    # Contiguous output view with guards on both sides.
    storage = torch.full((m + 2, n), 17, device=device, dtype=torch.bfloat16)
    out = storage[1:-1]
    out.fill_(float("nan"))
    return (a, sa, b, sb, out, indices, counts), storage


def cpu_reference(args, lengths):
    a, sa, b, sb = [x.cpu().double() for x in args[:4]]
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty((m, n), dtype=torch.bfloat16)
    start = 0
    for expert, count in enumerate(lengths):
        end = start + count
        acc = torch.zeros((count, n), dtype=torch.float64)
        for block in range(k // 128):
            section = slice(block * 128, (block + 1) * 128)
            dot = a[start:end, section] @ b[expert, :, section].T
            bs = sb[expert, :, block].repeat_interleave(128)[:n]
            acc += dot * sa[start:end, block, None] * bs[None, :]
        out[start:end] = acc.bfloat16()
        start = end
    return out


@pytest.mark.grouped_gemm_w8a8_fp8
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize(
    "lengths,n,k",
    [
        ([0, 0, 0], 64, 128),
        ([0, 1, 3, 0, 7], 192, 384),
        ([1] + [0] * 255 + [13], 64, 6144),
        ([1, 32, 0, 65], 128, 256),
        ([0, 257, 3], 64, 128),
        ([0, 257, 0, 511], 192, 384),
        ([0, 566] + [0] * 61 + [18], 64, 6144),
    ],
)
@pytest.mark.parametrize("scale_factor", [1.0, 100.0])
def test_grouped_gemm_w8a8_fp8(lengths, n, k, strided, scale_factor):
    args, storage = make_case(lengths, n, k, strided)
    # Also exercise order-one scales, where absolute tolerance cannot hide errors.
    args[1].mul_(scale_factor)
    args[3].mul_(scale_factor)
    expected = cpu_reference(args, lengths)
    assert flag_gems.grouped_gemm_w8a8_fp8(*args) is None
    # Keep the original strict check for captured small scales. Order-one
    # scales use the repository's GEMM reduction tolerance (also used by bmm
    # and mm_w8a8_fp8), plus a scale-independent relative-L2 error bound.
    actual = args[4].cpu()
    gems_assert_close(
        actual, expected, torch.bfloat16, reduce_dim=k if scale_factor > 1 else 1
    )
    if expected.numel():
        relative_error = (
            actual.float() - expected.float()
        ).norm() / expected.float().norm()
        assert relative_error < 1e-3
    assert torch.all(storage[0] == 17)
    assert torch.all(storage[-1] == 17)


@pytest.mark.grouped_gemm_w8a8_fp8
@pytest.mark.parametrize("m", [16, 512, 2048])
@pytest.mark.parametrize("k", [128, 384])
def test_grouped_gemm_w8a8_fp8_graph_metadata(m, k):
    """The same graph must consume changed metadata, including empty experts."""
    lengths = [m // 4] * 4
    args, storage = make_case(lengths, 192, k, True)
    op = flag_gems.grouped_gemm_w8a8_fp8
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op(*args)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(*args)
    for lengths in ([0, m - 1, 0, 1], [m // 2, 0, m // 2, 0], [0, 0, 0, m]):
        args[6].copy_(torch.tensor(lengths, device=args[6].device, dtype=torch.int32))
        args[5].copy_(
            torch.repeat_interleave(
                torch.arange(4, device=args[5].device, dtype=torch.int32),
                args[6],
                output_size=m,
            )
        )
        args[4].fill_(float("nan"))
        expected = cpu_reference(args, lengths)
        graph.replay()
        gems_assert_close(args[4].cpu(), expected, torch.bfloat16)
        assert torch.all(storage[0] == 17)
        assert torch.all(storage[-1] == 17)


@pytest.mark.grouped_gemm_w8a8_fp8
@pytest.mark.parametrize("lengths", [[1, 33, 0, 29], [0, 257, 0, 511]])
@pytest.mark.parametrize("unaligned", ["a", "b", "out", "all"])
def test_grouped_gemm_w8a8_fp8_unaligned(lengths, unaligned):
    """Contiguous tensors can have non-TMA-aligned storage offsets."""
    args, _ = make_case(lengths, 192, 384, True)
    args = list(args)
    for name, index in (("a", 0), ("b", 2)):
        if unaligned in (name, "all"):
            original = args[index]
            storage = torch.empty(
                original.numel() + 1, dtype=original.dtype, device=original.device
            )
            args[index] = storage[1:].view(original.shape)
            args[index].copy_(original)
    output_guard = None
    if unaligned in ("out", "all"):
        original = args[4]
        output_guard = torch.full(
            (original.numel() + 2,), 17, dtype=original.dtype, device=original.device
        )
        args[4] = output_guard[1:-1].view(original.shape)
    expected = cpu_reference(args, lengths)
    flag_gems.grouped_gemm_w8a8_fp8(*args)
    gems_assert_close(args[4].cpu(), expected, torch.bfloat16)
    if output_guard is not None:
        assert output_guard[0] == 17 and output_guard[-1] == 17


@pytest.mark.grouped_gemm_w8a8_fp8
@pytest.mark.parametrize(
    "lengths", [[257, 1, 767, 0], [0, 1025, 0, 0], [511, 1, 1537, 0]]
)
@pytest.mark.parametrize("n,k", [(192, 128), (192, 384)])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("unaligned", [False, True])
def test_grouped_gemm_w8a8_fp8_large_groups(lengths, n, k, strided, unaligned):
    """Exercise large-group scheduling with tails, empty experts and order-one scales."""
    args, storage = make_case(lengths, n, k, strided)
    args = list(args)
    if unaligned:
        for index in (0, 2):
            original = args[index]
            backing = torch.empty(
                original.numel() + 1, dtype=original.dtype, device=original.device
            )
            args[index] = backing[1:].view(original.shape)
            args[index].copy_(original)
    args[1].mul_(100)
    args[3].mul_(100)
    expected = cpu_reference(args, lengths)
    flag_gems.grouped_gemm_w8a8_fp8(*args)
    actual = args[4].cpu()
    # FP8 Tensor Core cancellation error scales with the operands. A fixed
    # absolute near-zero tolerance rejects even native DeepGEMM for these
    # order-one scales. Check normalized elements and relative L2 against FP64.
    rms = expected.float().square().mean().sqrt()
    torch.testing.assert_close(
        actual.float() / rms, expected.float() / rms, rtol=0.016, atol=1e-3
    )
    relative_error = (
        actual.float() - expected.float()
    ).norm() / expected.float().norm()
    assert relative_error < 1e-3
    assert torch.all(storage[0] == 17) and torch.all(storage[-1] == 17)


@pytest.mark.grouped_gemm_w8a8_fp8
def test_grouped_gemm_w8a8_fp8_persistent_unaligned_output():
    """Aligned TMA inputs must support a contiguous, unaligned output view."""
    lengths = [257, 1, 767, 0]
    args, _ = make_case(lengths, 192, 128, True)
    args = list(args)
    out = args[4]
    storage = torch.full((out.numel() + 2,), 17, dtype=out.dtype, device=out.device)
    args[4] = storage[1:-1].view(out.shape)
    expected = cpu_reference(args, lengths)
    flag_gems.grouped_gemm_w8a8_fp8(*args)
    gems_assert_close(args[4].cpu(), expected, torch.bfloat16)
    assert storage[0] == 17 and storage[-1] == 17
