# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import flag_gems

pytestmark = [
    pytest.mark.row_tiled_mm,
    pytest.mark.skipif(flag_gems.vendor_name != "metax", reason="MetaX row-tiled MM"),
]


def reference(a, b):
    m, n = a.shape[0], b.shape[1]
    out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    for start in range(0, m, 64):
        flag_gems.mm_out(a[start : start + 64], b, out=out[start : start + 64])
    return out


@pytest.mark.parametrize(
    "m,n,k,pad",
    [
        (64, 448, 7168, 0),
        (128, 448, 7168, 0),
        (257, 896, 3584, 0),
        (256, 448, 7168, 32),
        (129, 256, 3072, 0),
    ],
)
def test_graph_matches_reference(m, n, k, pad):
    torch.manual_seed(7134)
    x = torch.empty((m, k + pad), device="cuda", dtype=torch.bfloat16)[:, :k]
    b = torch.randn(n, k, device="cuda", dtype=x.dtype).t()
    x.normal_()
    ref = reference(x, b)
    a = flag_gems.row_tiled_mm(x, b)
    assert torch.equal(a.view(torch.int16), ref.view(torch.int16))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = flag_gems.row_tiled_mm(x, b)
    for _ in range(3):
        x.normal_()
        graph.replay()
        assert torch.equal(out.view(torch.int16), reference(x, b).view(torch.int16))


def test_empty_and_rejection():
    for m, k, n in [(0, 128, 64), (64, 0, 128), (64, 128, 0)]:
        x = torch.empty((m, k), device="cuda", dtype=torch.bfloat16)
        b = torch.empty((n, k), device="cuda", dtype=x.dtype).t()
        out = flag_gems.row_tiled_mm(x, b)
        assert (
            out.shape == (m, n)
            and out.dtype == x.dtype
            and not torch.count_nonzero(out)
        )
    x = torch.zeros((128, 64), dtype=torch.bfloat16, device="cuda")
    b = torch.zeros((32, 64), dtype=x.dtype, device=x.device).t()
    for bad in [0, -64, 63, 129]:
        with pytest.raises(ValueError, match="rows_per_launch"):
            flag_gems.row_tiled_mm(x, b, rows_per_launch=bad)
    with pytest.raises(ValueError, match="BF16"):
        flag_gems.row_tiled_mm(x.float(), b.float())
    with pytest.raises(ValueError, match="transposed"):
        flag_gems.row_tiled_mm(x, b.contiguous())


def test_stream_isolation():
    b = torch.randn((448, 7168), dtype=torch.bfloat16, device="cuda").t()
    xs = [torch.randn((256, 7168), device=b.device, dtype=b.dtype) for _ in range(2)]
    refs = [reference(x, b) for x in xs]
    streams = [torch.cuda.Stream() for _ in xs]
    outputs = []
    for s, x in zip(streams, xs):
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            outputs.append(flag_gems.row_tiled_mm(x, b))
    torch.cuda.synchronize()
    assert all(
        torch.equal(x.view(torch.int16), y.view(torch.int16))
        for x, y in zip(refs, outputs)
    )


@pytest.mark.parametrize("rows_per_launch", [64, 128, 256, 512])
def test_dyadic_cpu_reference(rows_per_launch):
    generator = torch.Generator().manual_seed(321)
    x = torch.randint(-4, 5, (257, 128), generator=generator).bfloat16() / 8
    w = torch.randint(-4, 5, (64, 128), generator=generator).bfloat16() / 8
    a, b = x.cuda(), w.cuda().t()
    actual = flag_gems.row_tiled_mm(a, b, rows_per_launch=rows_per_launch)
    native = reference(a, b)
    assert torch.equal(actual.view(torch.int16), native.view(torch.int16))

    # Bound FP32 reduction error independently, including cancellation at zero.
    # BF16 products are exactly representable in FP32. Apply final BF16
    # rounding to the endpoints of the standard dot-product error interval.
    exact = x.double() @ w.double().t()
    ku = x.size(1) * (2.0**-24)
    error = (ku / (1.0 - ku)) * (x.double().abs() @ w.double().abs().t())
    lower, upper = (exact - error).bfloat16(), (exact + error).bfloat16()
    cpu = actual.cpu()
    assert torch.all((lower <= cpu) & (cpu <= upper))


def test_shared_input_storage_and_one_column():
    x = torch.randn(192, 64, device="cuda", dtype=torch.bfloat16)
    for rhs in (x[64:128].t(), torch.randn(1, 64, device="cuda", dtype=x.dtype).t()):
        assert torch.equal(
            flag_gems.row_tiled_mm(x, rhs).view(torch.int16),
            reference(x, rhs).view(torch.int16),
        )


def test_column_selection_matches_cpu_exactly():
    x = (
        torch.arange(257 * 64, dtype=torch.int64)
        .remainder(19)
        .reshape(257, 64)
        .bfloat16()
    )
    columns = torch.tensor([1, 8, 19, 27])
    w = torch.zeros(4, 64, dtype=torch.bfloat16)
    w[torch.arange(4), columns] = 1
    actual = flag_gems.row_tiled_mm(x.cuda(), w.cuda().t())
    assert torch.equal(actual.cpu(), x[:, columns])
