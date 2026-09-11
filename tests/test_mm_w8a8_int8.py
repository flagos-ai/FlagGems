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

import sys

import pytest
import torch

import flag_gems

if flag_gems.vendor_name == "thead":
    _backend = sys.modules[flag_gems.mm_w8a8_int8.__module__]
    _mm_w8a8_int8_prequantized = _backend._mm_w8a8_int8_prequantized
    _mm_w8a8_int8_prequantized_out = _backend._mm_w8a8_int8_prequantized_out


pytestmark = [
    pytest.mark.mm_w8a8_int8,
    pytest.mark.skipif(flag_gems.vendor_name != "thead", reason="THead INT8 backend"),
]

SHAPES = [
    (1, 16, 16),
    (16, 1, 128),
    (256, 1, 2048),
    (2, 32, 32),
    (8, 64, 64),
    (16, 128, 64),
    (32, 128, 128),
    (64, 256, 128),
    (128, 256, 256),
    (192, 512, 512),
    (256, 768, 1024),
    (512, 1024, 1024),
    (16, 1, 2048),
    (16, 64, 2048),
    (16, 256, 2048),
    (16, 1024, 2048),
    (16, 2048, 512),
    (16, 2048, 4096),
    (16, 9216, 2048),
    (16, 12288, 2048),
    (1, 248320, 2048),
    (3, 17, 33),
    (17, 65, 129),
]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.mm_w8a8_int8
def test_mm_w8a8_int8(shape, dtype):
    m, n, k = shape
    a = torch.randint(-128, 128, (m, k), device=flag_gems.device, dtype=torch.int8)
    b = torch.randint(-128, 128, (n, k), device=flag_gems.device, dtype=torch.int8).t()
    sa = torch.rand(m, device=a.device) * 0.01
    sb = torch.rand(n, device=a.device) * 0.01
    ref = ((a.float() @ b.float()) * sa[:, None] * sb[None, :]).to(dtype)
    y = _mm_w8a8_int8_prequantized(a, b, sa, sb, out_dtype=dtype)
    torch.testing.assert_close(
        y, ref, rtol=1.0e-5 if dtype == torch.float32 else 1.0e-2, atol=1.0e-5
    )
    out = torch.empty_like(y)
    assert _mm_w8a8_int8_prequantized_out(a, b, sa, sb, out=out) is out
    torch.testing.assert_close(out, y, rtol=0, atol=0)


@pytest.mark.parametrize("layout", ["row_major", "sliced", "broadcast"])
@pytest.mark.mm_w8a8_int8
def test_mm_w8a8_int8_strides(layout):
    m, n, k = 17, 35, 67
    a = torch.randint(-10, 11, (m, k), device=flag_gems.device, dtype=torch.int8)
    b = torch.randint(-10, 11, (k, n), device=a.device, dtype=torch.int8)
    if layout == "sliced":
        a = torch.randint(-10, 11, (m * 2, k * 2), device=a.device, dtype=torch.int8)[
            1::2, 1::2
        ]
        b = torch.randint(-10, 11, (k * 2, n * 2), device=a.device, dtype=torch.int8)[
            1::2, 1::2
        ]
    if layout == "broadcast":
        a = a[:1].expand(m, k)
        b = b[:, :1].expand(k, n)
    sa = torch.ones(m, device=a.device)
    sb = torch.ones(n, device=a.device)
    y = _mm_w8a8_int8_prequantized(a, b, sa, sb, out_dtype=torch.float32)
    torch.testing.assert_close(y, a.float() @ b.float(), rtol=0, atol=0)


@pytest.mark.parametrize("use_graph", [False, True])
@pytest.mark.parametrize("use_out", [False, True])
@pytest.mark.mm_w8a8_int8
@pytest.mark.parametrize("shape", [(16, 32, 128), (1, 64, 2048), (128, 1, 2048)])
def test_mm_w8a8_int8_updates(use_graph, use_out, shape):
    m, n, k = shape
    a = torch.ones((m, k), device=flag_gems.device, dtype=torch.int8)
    b = torch.ones((n, k), device=a.device, dtype=torch.int8).t()
    sa = torch.ones((m, 1), device=a.device)
    sb = torch.ones((1, n), device=a.device)
    out = torch.empty((m, n), device=a.device)

    def call():
        if use_out:
            return _mm_w8a8_int8_prequantized_out(a, b, sa, sb, out=out)
        return _mm_w8a8_int8_prequantized(a, b, sa, sb, out_dtype=torch.float32)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    if use_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y = call()
    for av, bv, sav, sbv in [
        (1, 1, 1, 1),
        (2, 1, 1, 1),
        (2, 3, 1, 1),
        (2, 3, 2, 1),
        (2, 3, 2, 0.5),
        (0, 3, 1, 1),
    ]:
        a.fill_(av)
        b.fill_(bv)
        sa.fill_(sav)
        sb.fill_(sbv)
        if use_graph:
            graph.replay()
        else:
            y = call()
        torch.testing.assert_close(y, torch.full_like(y, k * av * bv * sav * sbv))


@pytest.mark.parametrize("shape", [(2, 3, 0), (0, 3, 8), (2, 0, 8), (0, 0, 0)])
@pytest.mark.mm_w8a8_int8
def test_mm_w8a8_int8_empty(shape):
    m, n, k = shape
    a = torch.empty((m, k), device=flag_gems.device, dtype=torch.int8)
    b = torch.empty((k, n), device=a.device, dtype=torch.int8)
    sa = torch.ones(m, device=a.device)
    sb = torch.ones(n, device=a.device)
    y = _mm_w8a8_int8_prequantized(a, b, sa, sb)
    torch.testing.assert_close(y, torch.zeros_like(y))


@pytest.mark.mm_w8a8_int8
def test_mm_w8a8_int8_reject_float():
    a = torch.ones((2, 3), device=flag_gems.device)
    b = torch.ones((3, 4), device=a.device)
    with pytest.raises(TypeError, match="prequantized"):
        _mm_w8a8_int8_prequantized(
            a, b, torch.ones(2, device=a.device), torch.ones(4, device=a.device)
        )


def _activation_int8_reference(a, b, sb, dtype):
    # CPU INT64 accumulation is independent of PPU quantization and GEMM.
    a = a.detach().float().cpu()
    b, sb = b.detach().cpu().to(torch.int64), sb.detach().float().cpu().reshape(-1)
    if not a.shape[0] or not b.shape[1] or not a.shape[1]:
        return torch.zeros((a.shape[0], b.shape[1]), dtype=dtype)
    peak_a = a.abs().amax(1).clamp_min(1e-10)
    sa = peak_a * (1.0 / 127)
    aq = torch.round((a / peak_a[:, None]) * 127).clamp(-127, 127).to(torch.int64)
    return ((aq @ b).float() * sa[:, None] * sb[None, :]).to(dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("column_major", [False, True])
@pytest.mark.parametrize("shape", [(1, 17, 32), (17, 35, 67), (64, 128, 128)])
def test_mm_w8a8_int8_activation(dtype, out_dtype, column_major, shape):
    m, n, k = shape
    torch.manual_seed(42)
    a = torch.randn((m, k), device=flag_gems.device, dtype=dtype)
    b = torch.randint(-128, 128, (k, n), device=a.device, dtype=torch.int8)
    if column_major:
        b = b.t().contiguous().t()
    sb = torch.rand(n, device=a.device) * 0.03
    if m > 1:
        a[-1] = 0
    else:
        a[0, 0] = 0
    b[:, 0] = 0
    ref = _activation_int8_reference(a, b, sb, out_dtype)
    y = flag_gems.mm_w8a8_int8(a, b, sb, out_dtype=out_dtype)
    torch.testing.assert_close(y.cpu(), ref, rtol=1e-5, atol=1e-4)
    out = torch.empty_like(y)
    assert flag_gems.mm_w8a8_int8_out(a, b, sb, out=out) is out
    torch.testing.assert_close(out, y, rtol=0, atol=0)


@pytest.mark.parametrize("use_graph", [False, True])
@pytest.mark.parametrize("use_out", [False, True])
@pytest.mark.parametrize("column_major", [False, True])
@pytest.mark.parametrize("shape", [(16, 17, 32), (1, 64, 128), (128, 1, 128)])
def test_mm_w8a8_int8_activation_updates(use_graph, use_out, column_major, shape):
    m, n, k = shape
    a = torch.ones((m, k), device=flag_gems.device)
    b = torch.ones((k, n), device=a.device, dtype=torch.int8)
    if column_major:
        b = b.t().contiguous().t()
    sb = torch.ones((1, n), device=a.device)
    out = torch.empty((m, n), device=a.device)

    def call():
        if use_out:
            return flag_gems.mm_w8a8_int8_out(a, b, sb, out=out)
        return flag_gems.mm_w8a8_int8(a, b, sb, out_dtype=torch.float32)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    if use_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y = call()
    for av, bv, sbv in [
        (1, 1, 1),
        (2, 1, 1),
        (2, -128, 1),
        (2, -128, 0.03),
        (0, 2, 1),
        (0.5, 4, 0),
    ]:
        a.fill_(av)
        b.fill_(bv)
        sb.fill_(sbv)
        if use_graph:
            graph.replay()
        else:
            y = call()
        torch.testing.assert_close(y, torch.full_like(y, k * av * bv * sbv))


@pytest.mark.parametrize("shape", [(2, 3, 0), (0, 3, 8), (2, 0, 8), (0, 0, 0)])
def test_mm_w8a8_int8_activation_empty(shape):
    m, n, k = shape
    a = torch.empty((m, k), device=flag_gems.device)
    b = torch.empty((k, n), device=a.device, dtype=torch.int8)
    sb = torch.ones(n, device=a.device)
    y = flag_gems.mm_w8a8_int8(a, b, sb)
    assert y.dtype == torch.bfloat16
    torch.testing.assert_close(y, torch.zeros_like(y))
    out = torch.full((m, n), float("nan"), device=a.device)
    assert flag_gems.mm_w8a8_int8_out(a, b, sb, out=out) is out
    torch.testing.assert_close(out, torch.zeros_like(out))


@pytest.mark.parametrize("layout", ["row_major", "column_major", "sliced", "broadcast"])
def test_mm_w8a8_int8_activation_strides(layout):
    a = torch.randn((34, 134), device=flag_gems.device)[::2, ::2]
    b = torch.randint(-128, 128, (67, 35), device=a.device, dtype=torch.int8)
    if layout == "column_major":
        b = b.t().contiguous().t()
    elif layout == "sliced":
        storage = torch.empty((134, 70), device=a.device, dtype=torch.int8)
        storage[::2, ::2] = b
        b = storage[::2, ::2]
    elif layout == "broadcast":
        a = a[:1].expand(17, 67)
        b = b[:, :1].expand(67, 35)
    sb = torch.rand(35, device=a.device)
    before_b, before_sb = b.clone(), sb.clone()
    aq, bq, _, returned_sb = _backend._prepare_mm_w8a8_int8_inputs(a, b, sb)
    assert aq.stride() == (67, 1)
    assert bq.stride() == (1, 67)
    assert returned_sb is sb
    torch.testing.assert_close(bq, before_b, rtol=0, atol=0)
    y = flag_gems.mm_w8a8_int8(a, b, sb, out_dtype=torch.float32)
    torch.testing.assert_close(
        y.cpu(),
        _activation_int8_reference(a, b, sb, torch.float32),
        rtol=1e-5,
        atol=1e-4,
    )
    torch.testing.assert_close(b, before_b, rtol=0, atol=0)
    torch.testing.assert_close(sb, before_sb, rtol=0, atol=0)


def test_mm_w8a8_int8_activation_rounding():
    # Peak=127 gives exact half-integer ties, including negative values.
    a = torch.tensor(
        [[127, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5, -127]], device=flag_gems.device
    )
    b = torch.eye(8, device=a.device, dtype=torch.int8)
    sb = torch.ones(8, device=a.device)
    y = flag_gems.mm_w8a8_int8(a, b, sb, out_dtype=torch.float32)
    torch.testing.assert_close(y, torch.round(a), rtol=0, atol=0)
    assert flag_gems.mm_w8a8_int8(a, b, sb).dtype == torch.bfloat16


@pytest.mark.parametrize("use_out", [False, True])
@pytest.mark.parametrize(
    "invalid",
    [
        "a_dtype",
        "b_dtype",
        "scale_dtype",
        "scale_shape",
        "scale_stride",
        "scale_device",
        "shape",
        "tensor",
        "output_dtype",
        "output_shape",
        "output_stride",
    ],
)
def test_mm_w8a8_int8_activation_validation(use_out, invalid):
    a = torch.ones((2, 3), device=flag_gems.device)
    b = torch.ones((3, 4), device=a.device, dtype=torch.int8)
    sb = torch.ones(4, device=a.device)
    out = torch.empty((2, 4), device=a.device)
    out_dtype = torch.float32
    error, match = ValueError, ""
    if invalid == "a_dtype":
        a = a.to(torch.int8)
        error, match = TypeError, "A must"
    elif invalid == "b_dtype":
        b = b.float()
        error, match = TypeError, "prequantized"
    elif invalid == "scale_dtype":
        sb = sb.half()
        match = "FP32"
    elif invalid == "scale_shape":
        sb = sb[:, None]
        match = "scale_b"
    elif invalid == "scale_stride":
        sb = torch.ones(8, device=a.device)[::2]
        match = "contiguous FP32"
    elif invalid == "scale_device":
        sb = sb.cpu()
        match = "same PPU"
    elif invalid == "shape":
        b = b[:2]
        match = "expected A"
    elif invalid == "tensor":
        sb = 1.0
        error, match = TypeError, "Tensor"
    elif invalid == "output_dtype":
        out, out_dtype = out.to(torch.int8), torch.int8
        error, match = TypeError, "BF16, FP16 or FP32"
    elif invalid == "output_shape":
        if not use_out:
            pytest.skip("out-only validation")
        out = out[:1]
        match = "out must be contiguous"
    elif invalid == "output_stride":
        if not use_out:
            pytest.skip("out-only validation")
        out = torch.empty((2, 8), device=a.device)[:, ::2]
        match = "out must be contiguous"
    with pytest.raises(error, match=match):
        if use_out:
            flag_gems.mm_w8a8_int8_out(a, b, sb, out=out)
        else:
            flag_gems.mm_w8a8_int8(a, b, sb, out_dtype=out_dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("n", [1, 17])
def test_mm_w8a8_int8_long_reduction(dtype, n):
    m, k = 3, 33001
    a = torch.randn(m * 2, k * 2, device=flag_gems.device, dtype=dtype)[::2, ::2]
    b = torch.randint(-128, 128, (k * 2, n * 2), device=a.device, dtype=torch.int8)[
        ::2, ::2
    ]
    sb = torch.rand(n, device=a.device) * 0.01
    ref = _activation_int8_reference(a, b, sb, torch.float32)
    out = torch.empty(m, n, device=a.device)
    flag_gems.mm_w8a8_int8_out(a, b, sb, out=out)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("dtype,bits", [(torch.float16, 10), (torch.bfloat16, 7)])
def test_mm_w8a8_int8_quantization_mantissas(dtype, bits):
    # Enumerate activation mantissas around all relevant exponent differences.
    mant = torch.arange(2**bits, 2 ** (bits + 1), dtype=torch.float32) / (2**bits)
    pool = (
        mant[None, :] * 2.0 ** (-torch.arange(12, dtype=torch.float32)[:, None])
    ).flatten()
    pool = torch.cat([pool, -pool])
    peaks = mant[:: max(1, mant.numel() // 16)]
    a = pool[None, :].expand(peaks.numel(), -1).clone()
    a[a.abs() > peaks[:, None]] = 0
    a = torch.cat([peaks[:, None], a], dim=1).to(dtype)
    af = a.float()
    peak = af.abs().amax(1).clamp_min(1e-10)
    ref = torch.round((af / peak[:, None]) * 127).clamp(-127, 127).to(torch.int8)
    a = a.to(flag_gems.device)
    b = torch.zeros(a.shape[1], 1, device=a.device, dtype=torch.int8)
    sb = torch.ones(1, device=a.device)
    aq, _, _, _ = _backend._prepare_mm_w8a8_int8_inputs(a, b, sb)
    torch.testing.assert_close(aq.cpu(), ref, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mm_w8a8_int8_low_precision_ties(dtype):
    # A reciprocal alone can move a true half-integer across the rounding tie.
    a = torch.tensor(
        [[1.736328125, 0.0888671875, -0.0888671875, 0.0]],
        dtype=dtype,
        device=flag_gems.device,
    )
    b = torch.eye(4, device=a.device, dtype=torch.int8)
    sb = torch.ones(4, device=a.device)
    out = flag_gems.mm_w8a8_int8(a, b, sb, out_dtype=torch.float32)
    torch.testing.assert_close(
        out.cpu(), _activation_int8_reference(a, b, sb, torch.float32), rtol=0, atol=0
    )


@pytest.mark.parametrize("magnitude", [1e-30, 1e-12, 1e-10, 1e20, 1e30])
def test_mm_w8a8_int8_bfloat16_exponents(magnitude):
    a = (
        torch.tensor([[1.0, 0.5, -0.25, 0.0]], device=flag_gems.device) * magnitude
    ).bfloat16()
    b = torch.eye(4, device=a.device, dtype=torch.int8)
    sb = torch.ones(4, device=a.device)
    out = flag_gems.mm_w8a8_int8(a, b, sb, out_dtype=torch.float32)
    torch.testing.assert_close(
        out.cpu(),
        _activation_int8_reference(a, b, sb, torch.float32),
        rtol=1e-5,
        atol=1e-35,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mm_w8a8_int8_persistent_reduction(dtype):
    m, k = 257, 33001
    a = torch.randn(m, k, device=flag_gems.device, dtype=dtype)
    b = torch.randint(-128, 128, (k, 1), device=a.device, dtype=torch.int8)
    sb = torch.tensor([0.03], device=a.device)
    ref = _activation_int8_reference(a, b, sb, torch.float32)
    out = flag_gems.mm_w8a8_int8(a, b, sb, out_dtype=torch.float32)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-5, atol=1e-4)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            flag_gems.mm_w8a8_int8_out(a, b, sb, out=out)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flag_gems.mm_w8a8_int8_out(a, b, sb, out=out)
    for av, bv, sv in [(1, 2, 0.5), (0, 2, 0.5), (2, -128, 0.03)]:
        a.fill_(av)
        b.fill_(bv)
        sb.fill_(sv)
        graph.replay()
        torch.testing.assert_close(out, torch.full_like(out, k * av * bv * sv))


def test_mm_w8a8_int8_unaligned_weight():
    m, n, k = 2, 16, 128
    a = torch.randn(m, k, device=flag_gems.device, dtype=torch.bfloat16)
    storage = torch.randint(-128, 128, (n * k + 1,), device=a.device, dtype=torch.int8)
    b = storage[1:].view(n, k).t()
    sb = torch.rand(n, device=a.device)
    out = flag_gems.mm_w8a8_int8(a, b, sb, out_dtype=torch.float32)
    torch.testing.assert_close(
        out.cpu(),
        _activation_int8_reference(a, b, sb, torch.float32),
        rtol=1e-5,
        atol=1e-4,
    )


def test_mm_w8a8_int8_quantizer_cache_keeps_warp_configuration(monkeypatch):
    # Alternating small and large batches must not reuse the first launch's
    # warp count: LibEntry otherwise caches a 32-warp kernel for large batches.
    quantizer = _backend._quantize_activation_rows
    launched_warps = []

    class RecordLaunch:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                result = quantizer[grid](*args, **kwargs)
                launched_warps.append(result[0].metadata.num_warps)
                return result

            return launch

    monkeypatch.setattr(_backend, "_quantize_activation_rows", RecordLaunch())
    b = torch.ones(4096, 128, device=flag_gems.device, dtype=torch.int8)
    sb = torch.ones(128, device=b.device)
    for m in (8, 256, 8, 256):
        a = torch.ones(m, 4096, device=b.device, dtype=torch.bfloat16)
        aq, _, _, _ = _backend._prepare_mm_w8a8_int8_inputs(a, b, sb)
        torch.testing.assert_close(aq, torch.full_like(aq, 127))
    assert launched_warps == [32, 4, 32, 4]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 32769, 2048),
        (3, 32769, 2049),
        (8, 32768, 3584),
        (256, 32768, 2048),
        (1025, 65, 2048),
    ],
)
def test_mm_w8a8_int8_optimized_shapes_and_graph(dtype, shape):
    m, n, k = shape
    a = torch.randn(m, k, device=flag_gems.device, dtype=dtype)
    b = torch.randint(-128, 128, (n, k), device=a.device, dtype=torch.int8).t()
    sb = torch.rand(n, device=a.device) * 0.01
    rows = torch.tensor(sorted({0, m // 2, m - 1}), device=a.device)
    cols = torch.tensor(sorted({0, n // 2, n - 1}), device=a.device)
    out = flag_gems.mm_w8a8_int8(a, b, sb, out_dtype=torch.float32)
    reference = _activation_int8_reference(a[rows], b[:, cols], sb[cols], out.dtype)
    torch.testing.assert_close(
        out[rows][:, cols].cpu(), reference, rtol=1e-5, atol=1e-4
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            flag_gems.mm_w8a8_int8_out(a, b, sb, out=out)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flag_gems.mm_w8a8_int8_out(a, b, sb, out=out)
    for av, bv, sv in [(1, 2, 0.5), (2, 2, 0.5), (2, -128, 0.03), (0, 1, 1)]:
        a.fill_(av)
        b.fill_(bv)
        sb.fill_(sv)
        graph.replay()
        torch.testing.assert_close(out, torch.full_like(out, k * av * bv * sv))
