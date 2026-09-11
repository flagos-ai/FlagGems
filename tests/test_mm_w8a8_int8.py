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

import itertools
import sys

import pytest
import torch
import triton

import flag_gems

if hasattr(flag_gems, "mm_w8a8_int8_out"):
    _backend = sys.modules[flag_gems.mm_w8a8_int8.__module__]
    _mm_w8a8_int8_prequantized = _backend._mm_w8a8_int8_prequantized
    _mm_w8a8_int8_prequantized_out = _backend._mm_w8a8_int8_prequantized_out


pytestmark = [
    pytest.mark.mm_w8a8_int8,
    pytest.mark.skipif(
        not hasattr(flag_gems, "mm_w8a8_int8_out"),
        reason="mm_w8a8_int8 is not implemented by the active backend",
    ),
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


DTYPES = [torch.float16, torch.bfloat16, torch.float32]


def _dynamic_reference(a, b, scale_b, dtype=torch.float32):
    a = a.detach().float().cpu()
    b = b.cpu().long()
    if not a.shape[0] or not b.shape[1] or not a.shape[1]:
        return torch.zeros((a.shape[0], b.shape[1]), dtype=dtype)
    peak = a.abs().amax(1).clamp_min(1e-10)
    aq = (a / peak[:, None] * 127).round().clamp(-127, 127).long()
    return ((aq @ b).float() * (peak[:, None] / 127) * scale_b.cpu().reshape(1, -1)).to(
        dtype
    )


@pytest.mark.parametrize("dtype,out_dtype", list(itertools.product(DTYPES, DTYPES)))
@pytest.mark.parametrize(
    "shape", [(1, 17, 32), (17, 35, 67), (64, 128, 128), (2, 7, 4097)]
)
@pytest.mark.parametrize("column_major", [False, True])
def test_dynamic_accuracy(dtype, out_dtype, shape, column_major):
    m, n, k = shape
    a = torch.randn((m, k), device=flag_gems.device, dtype=dtype)
    b = torch.randint(-128, 128, (k, n), device=a.device, dtype=torch.int8)
    if column_major:
        b = b.t().contiguous().t()
    sb = torch.rand(n, device=a.device) * 0.01
    expected = _dynamic_reference(a, b, sb, out_dtype)
    actual = flag_gems.mm_w8a8_int8(a, b, sb, out_dtype=out_dtype)
    torch.testing.assert_close(
        actual.cpu(),
        expected,
        rtol={torch.float16: 0.001, torch.bfloat16: 0.008, torch.float32: 1e-5}[
            out_dtype
        ],
        atol=1e-4,
    )
    out = torch.empty_like(actual)
    assert flag_gems.mm_w8a8_int8_out(a, b, sb, out=out) is out
    torch.testing.assert_close(out, actual, rtol=0, atol=0)


@pytest.mark.parametrize("layout", ["transpose", "slice", "broadcast"])
def test_dynamic_strides(layout):
    a = torch.randn(17, 66, device=flag_gems.device)
    b = torch.randint(-128, 128, (66, 35), device=a.device, dtype=torch.int8)
    if layout == "transpose":
        a = a.t().contiguous().t()
        b = b.t().contiguous().t()
    elif layout == "slice":
        a, b = a[:, 1::2], b[1::2, :]
    else:
        a, b = a[:1].expand(17, 66), b[:, :1].expand(66, 35)
    sb = torch.rand(1, 35, device=a.device)
    actual = flag_gems.mm_w8a8_int8(a, b, sb, out_dtype=torch.float32)
    torch.testing.assert_close(
        actual.cpu(), _dynamic_reference(a, b, sb), rtol=1e-5, atol=1e-4
    )


@pytest.mark.parametrize(
    "use_graph,use_out", list(itertools.product([False, True], repeat=2))
)
@pytest.mark.parametrize(
    "shape", [(1, 512, 1024), (17, 35, 67), (2, 7, 4097), (4, 17, 8193)]
)
def test_dynamic_updates(use_graph, use_out, shape):
    m, n, k = shape
    a = torch.randn(m, k, device=flag_gems.device)
    b = torch.randint(-128, 128, (k, n), device=a.device, dtype=torch.int8)
    sb = torch.rand(n, device=a.device)
    out = torch.empty(m, n, device=a.device)

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
            actual = call()
    # Change individual elements as well as row peaks, so stale activation
    # codes or scales cannot pass by relying on uniform rescaling alone.
    for change in ["activation", "weight", "scale", "zero"]:
        if change == "activation":
            a.normal_()
            a[:, 0] = 17
        elif change == "weight":
            b.random_(-128, 128)
        elif change == "scale":
            sb.mul_(0.5)
        else:
            a.zero_()
        if use_graph:
            graph.replay()
        else:
            actual = call()
        torch.testing.assert_close(
            actual.cpu(), _dynamic_reference(a, b, sb), rtol=1e-5, atol=1e-3
        )


@pytest.mark.parametrize("shape", [(2, 3, 0), (0, 3, 8), (2, 0, 8), (0, 0, 0)])
def test_dynamic_empty(shape):
    m, n, k = shape
    a = torch.empty(m, k, device=flag_gems.device)
    b = torch.empty(k, n, device=a.device, dtype=torch.int8)
    sb = torch.ones(n, device=a.device)
    actual = flag_gems.mm_w8a8_int8(a, b, sb)
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, torch.zeros_like(actual))
    out = torch.full((m, n), float("nan"), device=a.device)
    assert flag_gems.mm_w8a8_int8_out(a, b, sb, out=out) is out
    torch.testing.assert_close(out, torch.zeros_like(out))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("k", [32, 4097])
def test_dynamic_rounding_and_tiny_values(dtype, k):
    a = torch.zeros((3, k), device=flag_gems.device, dtype=dtype)
    vals = torch.tensor(
        [-127, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 127],
        device=a.device,
        dtype=dtype,
    )
    a[0, :10] = vals
    a[1, :10] = vals * 1e-13
    b = torch.zeros(k, 10, device=a.device, dtype=torch.int8)
    b[:10] = torch.eye(10, device=a.device, dtype=torch.int8)
    sb = torch.ones(10, device=a.device)
    actual = flag_gems.mm_w8a8_int8(a, b, sb, out_dtype=torch.float32)
    torch.testing.assert_close(
        actual.cpu(), _dynamic_reference(a, b, sb), rtol=1e-6, atol=1e-20
    )


def test_dynamic_alias():
    a = torch.randn(17, 67, device=flag_gems.device)
    b = torch.randint(-128, 128, (67, 67), device=a.device, dtype=torch.int8)
    sb = torch.rand(67, device=a.device)
    expected = _dynamic_reference(a, b, sb)
    assert flag_gems.mm_w8a8_int8_out(a, b, sb, out=a) is a
    torch.testing.assert_close(a.cpu(), expected, rtol=1e-5, atol=1e-4)
    # Reject shared weight/scale storage even for shifted views.
    with pytest.raises(ValueError, match="alias"):
        flag_gems.mm_w8a8_int8_out(a[:1], b, sb, out=sb[None, :])
    storage = torch.empty(67 * 68, device=a.device)
    weight = storage.view(torch.int8)[: 67 * 67].reshape(67, 67)
    with pytest.raises(ValueError, match="alias"):
        flag_gems.mm_w8a8_int8_out(
            a, weight, sb, out=storage[: 17 * 67].reshape(17, 67)
        )


@pytest.mark.parametrize(
    "case",
    [
        "a_dtype",
        "b_dtype",
        "matrix",
        "shape",
        "device",
        "scale_dtype",
        "scale_shape",
        "scale_stride",
        "out_dtype",
        "out_shape",
        "out_stride",
    ],
)
def test_dynamic_invalid(case):
    a = torch.ones(2, 3, device=flag_gems.device)
    b = torch.ones(3, 4, device=a.device, dtype=torch.int8)
    sb = torch.ones(4, device=a.device)
    with pytest.raises((TypeError, ValueError)):
        if case == "a_dtype":
            flag_gems.mm_w8a8_int8(a.int(), b, sb)
        elif case == "b_dtype":
            flag_gems.mm_w8a8_int8(a, b.float(), sb)
        elif case == "matrix":
            flag_gems.mm_w8a8_int8(a[0], b, sb)
        elif case == "shape":
            flag_gems.mm_w8a8_int8(a, b[:2], sb)
        elif case == "device":
            flag_gems.mm_w8a8_int8(a, b, sb.cpu())
        elif case == "scale_dtype":
            flag_gems.mm_w8a8_int8(a, b, sb.half())
        elif case == "scale_shape":
            flag_gems.mm_w8a8_int8(a, b, sb[:, None])
        elif case == "scale_stride":
            flag_gems.mm_w8a8_int8(a, b, torch.ones(8, device=a.device)[::2])
        elif case == "out_dtype":
            flag_gems.mm_w8a8_int8(a, b, sb, out_dtype=torch.int8)
        elif case == "out_shape":
            flag_gems.mm_w8a8_int8_out(a, b, sb, out=torch.empty(1, 4, device=a.device))
        else:
            flag_gems.mm_w8a8_int8_out(
                a, b, sb, out=torch.empty(4, 2, device=a.device).t()
            )


@pytest.mark.parametrize(
    "shape",
    [
        (2, 1024, 4096),
        (4, 512, 3584),
        (8, 1024, 4096),
        (98, 2049, 1024),
        (98, 4096, 4096),
        (98, 3584, 3584),
        (98, 4608, 3584),
        (256, 1024, 1024),
        (2048, 1024, 1024),
        (4096, 1024, 1024),
        (8192, 1024, 1024),
        (8192, 8192, 1024),
        (2048, 2048, 2048),
        (256, 18944, 3584),
        (256, 28672, 4096),
        (1, 3584, 18944),
        (4, 3584, 18944),
        (4, 14336, 4096),
        (4, 37888, 3584),
        (2, 128256, 4096),
        (8192, 512, 3584),
    ],
)
def test_dynamic_dispatch(shape):
    m, n, k = shape
    a = torch.randn(m, k, device=flag_gems.device, dtype=torch.bfloat16)
    b = torch.randint(-128, 128, (n, k), device=a.device, dtype=torch.int8).t()
    sb = torch.rand(n, device=a.device) * 0.01
    actual = flag_gems.mm_w8a8_int8(a, b, sb, out_dtype=torch.float32)
    rows, cols = [0, m // 2, m - 1], [0, n // 2, n - 1]
    expected = _dynamic_reference(a[rows], b[:, cols], sb[cols])
    torch.testing.assert_close(
        actual[rows][:, cols].cpu(), expected, rtol=1e-5, atol=1e-4
    )


@pytest.mark.parametrize("k", [131071, 131072, 151936, 152064])
def test_dynamic_long_k(k):
    a = torch.full((2, k), 127.0, device=flag_gems.device)
    b = torch.full((5, k), -128, device=a.device, dtype=torch.int8).t()
    sb = torch.full((5,), 0.5, device=a.device)
    actual = flag_gems.mm_w8a8_int8(a, b, sb, out_dtype=torch.float32)
    torch.testing.assert_close(
        actual, torch.full_like(actual, float(k * 127 * -128) * 0.5), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "shape",
    [
        (0, 3, 5),
        (3, 0, 5),
        (3, 5, 0),
        (2, 1024, 4096),
        (4, 512, 3584),
        (8, 512, 3584),
        (8, 1024, 4096),
        (98, 2049, 1024),
        (256, 1024, 1024),
    ],
)
def test_prequantized_dispatch(shape):
    backend = _backend
    m, n, k = shape
    a = torch.randint(-128, 128, (m, k), device=flag_gems.device, dtype=torch.int8)
    b = torch.randint(-128, 128, (n, k), device=flag_gems.device, dtype=torch.int8).t()
    sa = torch.ones(m, device=flag_gems.device)
    sb = torch.ones(n, device=flag_gems.device)
    actual = backend._mm_w8a8_int8_prequantized(a, b, sa, sb, out_dtype=torch.float32)
    expected = (a.cpu().long() @ b.cpu().long()).float()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.skipif(
    flag_gems.vendor_name != "hygon", reason="requires overflow-safe long-K reduction"
)
def test_large_tile_long_k_overflow():
    backend = _backend
    m, n, k = 1025, 1025, 131073
    a = torch.full((m, k), -128, device=flag_gems.device, dtype=torch.int8)
    b = torch.full((n, k), -128, device=flag_gems.device, dtype=torch.int8).t()
    sa = torch.full((m,), 0.5, device=flag_gems.device)
    sb = torch.full((n,), -0.25, device=flag_gems.device)
    actual = backend._mm_w8a8_int8_prequantized(a, b, sa, sb, out_dtype=torch.float32)
    expected = torch.full_like(actual, float(k * 128 * 128) * -0.125)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "dtype,mantissa", [(torch.float16, 1024), (torch.bfloat16, 128)]
)
def test_half_quantization_mantissas(dtype, mantissa):
    # Exhaust all significand pairs whose normalized magnitude can round nonzero.
    # Larger exponent gaps are strictly below 0.5. Compare against CPU FP32
    # division followed by multiplication, including their intermediate rounding.
    values = 1.0 + torch.arange(mantissa, dtype=torch.float32) / mantissa
    for shift in range(9):
        positive = (values[None, :].expand(mantissa, -1) / (2.0**shift)).minimum(
            values[:, None]
        )
        x = torch.cat([positive, -positive, values[:, None]], 1).to(dtype)
        xf = x.float()
        peak = xf.abs().amax(1).clamp_min(1e-10)
        expected = (xf / peak[:, None] * 127).round().to(torch.int8)
        a = x.to(flag_gems.device)
        m, k = a.shape
        q = torch.empty_like(a, dtype=torch.int8)
        scale = torch.empty(m, device=a.device)
        _backend._quantize_half_rows[(triton.cdiv(m, 2),)](
            a,
            q,
            scale,
            m,
            k,
            *a.stride(),
            2,
            triton.next_power_of_2(k),
            num_warps=4,
            enable_fp_fusion=False
        )
        torch.testing.assert_close(q.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(scale.cpu(), peak * (1.0 / 127), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_quantization_extremes(dtype):
    exponents = (
        [-24, -20, -10, 0, 10]
        if dtype == torch.float16
        else [-120, -34, 0, 100, 120, 127]
    )
    for exponent in exponents:
        for significand in [1.0, 1.5, 127.0 / 64]:
            peak_value = torch.tensor(2.0**exponent * significand, dtype=dtype)
            x = torch.linspace(-1, 1, 257).mul(peak_value.float()).to(dtype)[None, :]
            a = x.to(flag_gems.device)
            q = torch.empty_like(a, dtype=torch.int8)
            scale = torch.empty(1, device=a.device)
            _backend._quantize_half_rows[(1,)](
                a,
                q,
                scale,
                1,
                a.shape[1],
                *a.stride(),
                1,
                512,
                num_warps=4,
                enable_fp_fusion=False
            )
            p = x.float().abs().amax(1).clamp_min(1e-10)
            expected = (x.float() / p[:, None] * 127).round().to(torch.int8)
            torch.testing.assert_close(q.cpu(), expected, rtol=0, atol=0)
            torch.testing.assert_close(scale.cpu(), p * (1.0 / 127), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("use_graph", [False, True])
def test_dynamic_long_vector_updates(dtype, use_graph):
    m, k = 65, 152064
    a = torch.rand(m, k, device=flag_gems.device, dtype=dtype)
    a[:, -1] = 7.9375
    b = torch.randint(-128, 128, (k, 1), device=a.device, dtype=torch.int8)
    sb = torch.full((1,), 0.5, device=a.device)
    out = torch.empty(m, 1, device=a.device)

    def call():
        return flag_gems.mm_w8a8_int8_out(a, b, sb, out=out)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    if use_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
    for phase in range(3):
        if phase == 1:
            a.uniform_(-1, 1)
            a[:, -1] = 4.5
        elif phase == 2:
            a.fill_(127)
            b.fill_(-128)
        if use_graph:
            graph.replay()
        else:
            call()
        if phase == 2:
            torch.testing.assert_close(
                out, torch.full_like(out, float(k * 127 * -128) * 0.5), rtol=0, atol=0
            )
        else:
            torch.testing.assert_close(
                out.cpu(), _dynamic_reference(a, b, sb), rtol=1e-5, atol=1e-3
            )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dynamic_long_vector_alias(dtype):
    m, k = 65, 65537
    a = torch.ones(m, k, device=flag_gems.device, dtype=dtype)
    b = torch.ones(k, 1, device=a.device, dtype=torch.int8)
    sb = torch.full((1,), 0.001, device=a.device)
    expected = _dynamic_reference(a, b, sb, dtype)
    out = a.flatten()[:m].view(m, 1)
    assert flag_gems.mm_w8a8_int8_out(a, b, sb, out=out) is out
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
