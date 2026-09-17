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

pytestmark = pytest.mark.mm_w8a8_int8


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


@pytest.mark.skipif(flag_gems.vendor_name != "thead", reason="thead only")
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.mm_w8a8_int8
def test_mm_w8a8_int8(shape, dtype):
    m, n, k = shape
    a = torch.randint(-128, 128, (m, k), device=flag_gems.device, dtype=torch.int8)
    b = torch.randint(-128, 128, (n, k), device=flag_gems.device, dtype=torch.int8).t()
    sa = torch.rand(m, device=a.device) * 0.01
    sb = torch.rand(n, device=a.device) * 0.01
    y = flag_gems.mm_w8a8_int8(a, b, sa, sb, out_dtype=dtype)
    _assert_reference(a, b, sa, sb, y)
    out = torch.empty_like(y)
    assert flag_gems.mm_w8a8_int8_out(a, b, sa, sb, out=out) is out
    torch.testing.assert_close(out, y, rtol=0, atol=0)


@pytest.mark.skipif(flag_gems.vendor_name != "thead", reason="thead only")
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
    y = flag_gems.mm_w8a8_int8(a, b, sa, sb, out_dtype=torch.float32)
    torch.testing.assert_close(y, a.float() @ b.float(), rtol=0, atol=0)


@pytest.mark.skipif(flag_gems.vendor_name != "thead", reason="thead only")
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
            return flag_gems.mm_w8a8_int8_out(a, b, sa, sb, out=out)
        return flag_gems.mm_w8a8_int8(a, b, sa, sb, out_dtype=torch.float32)

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


@pytest.mark.skipif(flag_gems.vendor_name != "thead", reason="thead only")
@pytest.mark.parametrize("shape", [(2, 3, 0), (0, 3, 8), (2, 0, 8), (0, 0, 0)])
@pytest.mark.mm_w8a8_int8
def test_mm_w8a8_int8_empty(shape):
    m, n, k = shape
    a = torch.empty((m, k), device=flag_gems.device, dtype=torch.int8)
    b = torch.empty((k, n), device=a.device, dtype=torch.int8)
    sa = torch.ones(m, device=a.device)
    sb = torch.ones(n, device=a.device)
    y = flag_gems.mm_w8a8_int8(a, b, sa, sb)
    torch.testing.assert_close(y, torch.zeros_like(y))


@pytest.mark.skipif(flag_gems.vendor_name != "thead", reason="thead only")
@pytest.mark.mm_w8a8_int8
def test_mm_w8a8_int8_reject_float():
    a = torch.ones((2, 3), device=flag_gems.device)
    b = torch.ones((3, 4), device=a.device)
    with pytest.raises(TypeError, match="prequantized"):
        flag_gems.mm_w8a8_int8(
            a, b, torch.ones(2, device=a.device), torch.ones(4, device=a.device)
        )


def _assert_reference(a, b, sa, sb, out, bias=None):
    # Check all entries of small cases and a deterministic subset of large
    # cases against an independent CPU INT64 product, not a floating GEMM.
    m, k = a.shape
    n = b.shape[1]
    rows = list(range(m)) if m * n * k <= 1000000 else sorted({0, m // 2, m - 1})
    cols = list(range(n)) if m * n * k <= 1000000 else sorted({0, n // 2, n - 1})
    aa = a[rows].cpu().to(torch.int64)
    bb = b[:, cols].cpu().to(torch.int64)
    sa = sa.cpu().reshape(-1)
    sb = sb.cpu().reshape(-1)
    sa = sa.expand(m) if sa.numel() == 1 else sa
    sb = sb.expand(n) if sb.numel() == 1 else sb
    ref = (aa @ bb).float() * sa[rows, None] * sb[None, cols]
    if bias is not None:
        ref += bias.cpu()[cols].float()[None, :]
    torch.testing.assert_close(
        out[rows][:, cols].cpu(),
        ref.to(out.dtype),
        rtol=1e-5 if out.dtype == torch.float32 else 1e-2,
        atol=1e-4,
    )


@pytest.mark.skipif(flag_gems.vendor_name != "thead", reason="thead only")
@pytest.mark.parametrize(
    "shape",
    [
        (1, 17, 32),
        (17, 1, 128),
        (3, 35, 128),
        (17, 35, 67),
        (64, 128, 128),
        (1, 32769, 2048),
        (256, 16384, 2048),
        (1025, 65, 2048),
    ],
)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("scales", ["tensor", "axis", "mixed_a", "mixed_b"])
def test_scaled_bias(shape, out_dtype, scales):
    m, n, k = shape
    a = torch.randint(-128, 128, (m, k), device=flag_gems.device, dtype=torch.int8)
    b = torch.randint(-128, 128, (n, k), device=a.device, dtype=torch.int8).t()
    sa = torch.rand(1 if scales in ("tensor", "mixed_a") else m, device=a.device) * 0.01
    sb = torch.rand(1 if scales in ("tensor", "mixed_b") else n, device=a.device) * 0.02
    bias = torch.randn(n, device=a.device, dtype=out_dtype)
    # Exercise the vLLM-style positional output dtype and bias arguments.
    y = flag_gems.mm_w8a8_int8(a, b, sa, sb, out_dtype, bias)
    _assert_reference(a, b, sa, sb, y, bias)
    out = torch.empty_like(y)
    assert flag_gems.mm_w8a8_int8_out(a, b, sa, sb, out=out, bias=bias) is out
    torch.testing.assert_close(out, y, rtol=0, atol=0)


@pytest.mark.skipif(flag_gems.vendor_name != "thead", reason="thead only")
@pytest.mark.parametrize("shape", [(2, 3, 0), (0, 3, 8), (2, 0, 8), (0, 0, 0)])
def test_scaled_empty_bias(shape):
    m, n, k = shape
    a = torch.empty(m, k, device=flag_gems.device, dtype=torch.int8)
    b = torch.empty(k, n, device=a.device, dtype=torch.int8)
    scale = torch.ones(1, device=a.device)
    bias = torch.randn(n, device=a.device, dtype=torch.bfloat16)
    y = flag_gems.mm_w8a8_int8(a, b, scale, scale, bias=bias)
    assert y.dtype == torch.bfloat16
    torch.testing.assert_close(y, bias[None, :].expand(m, n))


@pytest.mark.skipif(flag_gems.vendor_name != "thead", reason="thead only")
@pytest.mark.parametrize(
    "shape", [(3, 17, 128), (1, 32, 128), (32, 1, 128), (17, 35, 67), (1025, 65, 2048)]
)
@pytest.mark.parametrize("scalar", [False, True])
def test_scaled_graph_bias_updates(shape, scalar):
    m, n, k = shape
    a = torch.ones(m, k, device=flag_gems.device, dtype=torch.int8)
    b = torch.ones(n, k, device=a.device, dtype=torch.int8).t()
    sa = torch.ones(1 if scalar else m, device=a.device)
    sb = torch.ones(1 if scalar else n, device=a.device)
    bias = torch.zeros(n, device=a.device)
    out = torch.empty(m, n, device=a.device)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            flag_gems.mm_w8a8_int8_out(a, b, sa, sb, out=out, bias=bias)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flag_gems.mm_w8a8_int8_out(a, b, sa, sb, out=out, bias=bias)
    for av, bv, sav, sbv, bv_bias in [
        (1, 1, 1, 1, 0),
        (2, -3, 0.5, 0.25, 2),
        (0, 2, 2, 1, -3),
    ]:
        a.fill_(av)
        b.fill_(bv)
        sa.fill_(sav)
        sb.fill_(sbv)
        bias.fill_(bv_bias)
        graph.replay()
        torch.testing.assert_close(
            out, torch.full_like(out, k * av * bv * sav * sbv + bv_bias)
        )


@pytest.mark.skipif(flag_gems.vendor_name != "thead", reason="thead only")
@pytest.mark.parametrize(
    "invalid",
    [
        "a_dtype",
        "b_dtype",
        "scale_dtype",
        "scale_shape",
        "scale_stride",
        "scale_device",
        "bias_shape",
        "bias_dtype",
        "bias_device",
        "bias_stride",
        "output_dtype",
        "output_shape",
        "output_stride",
    ],
)
def test_scaled_validation(invalid):
    a = torch.ones(2, 3, device=flag_gems.device, dtype=torch.int8)
    b = torch.ones(3, 4, device=a.device, dtype=torch.int8)
    sa = torch.ones(2, device=a.device)
    sb = torch.ones(4, device=a.device)
    bias = torch.ones(4, device=a.device, dtype=torch.bfloat16)
    out = torch.empty(2, 4, device=a.device, dtype=torch.bfloat16)
    if invalid == "a_dtype":
        a = a.float()
    elif invalid == "b_dtype":
        b = b.float()
    elif invalid == "scale_dtype":
        sa = sa.half()
    elif invalid == "scale_shape":
        sa = torch.ones(2, 2, device=a.device)
    elif invalid == "scale_stride":
        sb = torch.ones(8, device=a.device)[::2]
    elif invalid == "scale_device":
        sa = sa.cpu()
    elif invalid == "bias_shape":
        bias = bias[None, :]
    elif invalid == "bias_dtype":
        bias = bias.float()
    elif invalid == "bias_device":
        bias = bias.cpu()
    elif invalid == "bias_stride":
        bias = torch.ones(8, device=a.device, dtype=out.dtype)[::2]
    elif invalid == "output_dtype":
        out = out.to(torch.int8)
    elif invalid == "output_shape":
        out = out[:1]
    elif invalid == "output_stride":
        out = torch.empty(2, 8, device=a.device, dtype=out.dtype)[:, ::2]
    with pytest.raises((TypeError, ValueError)):
        flag_gems.mm_w8a8_int8_out(a, b, sa, sb, out=out, bias=bias)


@pytest.mark.skipif(flag_gems.vendor_name != "thead", reason="thead only")
@pytest.mark.parametrize(
    "shape", [(1, 17, 32), (3, 35, 128), (17, 35, 67), (3, 17, 33001)]
)
def test_tensor_scales_without_bias(shape):
    m, n, k = shape
    a = torch.randint(-128, 128, (m, k), device=flag_gems.device, dtype=torch.int8)
    b = torch.randint(-128, 128, (n, k), device=a.device, dtype=torch.int8).t()
    sa = torch.tensor(0.0125, device=a.device)
    sb = torch.tensor([[0.025]], device=a.device)
    y = flag_gems.mm_w8a8_int8(a, b, sa, sb)
    assert y.dtype == torch.bfloat16
    _assert_reference(a, b, sa, sb, y)


@pytest.mark.skipif(flag_gems.vendor_name != "thead", reason="thead only")
def test_scaled_unaligned_weight():
    m, n, k = 2, 16, 128
    a = torch.randint(-128, 128, (m, k), device=flag_gems.device, dtype=torch.int8)
    storage = torch.randint(-128, 128, (n * k + 1,), device=a.device, dtype=torch.int8)
    b = storage[1:].view(n, k).t()
    sa = torch.rand(m, device=a.device)
    sb = torch.rand(n, device=a.device)
    y = flag_gems.mm_w8a8_int8(a, b, sa, sb, torch.float32)
    _assert_reference(a, b, sa, sb, y)


def inputs(m, n, k, scalar=False, layout=False):
    a = torch.randint(-128, 128, (m, k), device=flag_gems.device, dtype=torch.int8)
    b = torch.randint(-128, 128, (n, k), device=flag_gems.device, dtype=torch.int8).t()
    if layout:
        a = a.t().contiguous().t()
        b = b.contiguous()
    sa = torch.rand((1,) if scalar else (m, 1), device=flag_gems.device) * 0.01
    sb = torch.rand((1,) if scalar else (1, n), device=flag_gems.device) * 0.01
    return a, b, sa, sb


def reference(a, b, sa, sb, bias=None, dtype=torch.float32):
    value = (a.cpu().long() @ b.cpu().long()).float()
    value = value * sa.cpu().reshape(-1, 1) * sb.cpu().reshape(1, -1)
    if bias is not None:
        value += bias.cpu().float()
    return value.to(dtype)


@pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="hygon only")
@pytest.mark.parametrize(
    "shape",
    [
        (0, 4, 8),
        (3, 0, 8),
        (3, 4, 0),
        (1, 1, 1),
        (7, 13, 19),
        (1, 17, 8192),
        (2, 31, 2048),
        (4, 513, 1024),
        (3, 1025, 1024),
        (4, 1024, 4097),
        (98, 2048, 1024),
        (129, 17, 513),
        (2, 3, 131073),
        (64, 1, 65536),
        (2, 1, 262145),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "scalar,bias_on,layout",
    [(False, False, False), (False, True, True), (True, True, False)],
)
@pytest.mark.mm_w8a8_int8
def test_prequantized(shape, dtype, scalar, bias_on, layout):
    a, b, sa, sb = inputs(*shape, scalar=scalar, layout=layout)
    bias = (
        torch.randn(shape[1], device=flag_gems.device, dtype=dtype) if bias_on else None
    )
    expected = reference(a, b, sa, sb, bias, dtype)
    actual = flag_gems.mm_w8a8_int8(a, b, sa, sb, dtype, bias)
    gems_assert_equal(actual.cpu(), expected)
    out = torch.empty_like(actual)
    assert flag_gems.mm_w8a8_int8_out(a, b, sa, sb, out=out, bias=bias) is out
    gems_assert_equal(out.cpu(), expected)


@pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="hygon only")
@pytest.mark.parametrize("code", [-128, 127])
def test_long_k_overflow(code):
    a, b, sa, sb = inputs(2, 3, 262145)
    a.fill_(code)
    b.fill_(code)
    sa.fill_(1)
    sb.fill_(1)
    y = flag_gems.mm_w8a8_int8(a, b, sa, sb, torch.float32)
    gems_assert_equal(y.cpu(), reference(a, b, sa, sb))


@pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="hygon only")
@pytest.mark.parametrize("a_scalar,b_scalar", [(True, False), (False, True)])
def test_mixed_scales(a_scalar, b_scalar):
    a, b, sa, sb = inputs(17, 13, 31)
    sa = sa[:1] if a_scalar else sa.flatten()
    sb = sb[:, :1].contiguous() if b_scalar else sb.flatten()
    y = flag_gems.mm_w8a8_int8(a, b, sa, sb)
    assert y.dtype == torch.bfloat16
    gems_assert_equal(y.cpu(), reference(a, b, sa, sb, dtype=torch.bfloat16))


@pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="hygon only")
def test_graph_updates():
    a, b, sa, sb = inputs(17, 13, 31)
    bias = torch.randn(13, device=flag_gems.device)
    out = torch.empty((17, 13), device=flag_gems.device)
    for _ in range(3):
        flag_gems.mm_w8a8_int8_out(a, b, sa, sb, out=out, bias=bias)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flag_gems.mm_w8a8_int8_out(a, b, sa, sb, out=out, bias=bias)
    a.fill_(-128)
    b.fill_(127)
    sa.mul_(2)
    sb.mul_(3)
    bias.add_(1)
    graph.replay()
    gems_assert_equal(out.cpu(), reference(a, b, sa, sb, bias))


@pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="hygon only")
@pytest.mark.parametrize(
    "bad",
    [
        "a_dtype",
        "b_dtype",
        "shape",
        "sa_shape",
        "sb_shape",
        "scale_dtype",
        "scale_stride",
        "bias_shape",
        "bias_dtype",
        "bias_device",
        "out_dtype",
        "out_shape",
        "out_stride",
        "alias",
    ],
)
def test_invalid(bad):
    a, b, sa, sb = inputs(3, 5, 7)
    bias = None
    out = torch.empty((3, 5), device=flag_gems.device)
    if bad == "a_dtype":
        a = a.float()
    elif bad == "b_dtype":
        b = b.float()
    elif bad == "shape":
        b = b[:2]
    elif bad == "sa_shape":
        sa = sa[:2]
    elif bad == "sb_shape":
        sb = sb[:, :2]
    elif bad == "scale_dtype":
        sa = sa.half()
    elif bad == "scale_stride":
        sa = torch.ones(6, device=flag_gems.device)[::2]
    elif bad == "bias_shape":
        bias = torch.ones(4, device=flag_gems.device)
    elif bad == "bias_dtype":
        bias = torch.ones(5, device=flag_gems.device, dtype=torch.int8)
    elif bad == "bias_device":
        bias = torch.ones(5)
    elif bad == "out_dtype":
        out = out.to(torch.int8)
    elif bad == "out_shape":
        out = out[:2]
    elif bad == "out_stride":
        out = torch.empty((5, 3), device=flag_gems.device).t()
    elif bad == "alias":
        sb = out[0]
    with pytest.raises((ValueError, TypeError)):
        flag_gems.mm_w8a8_int8_out(a, b, sa, sb, out=out, bias=bias)
