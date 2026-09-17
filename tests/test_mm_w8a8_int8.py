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

pytestmark = [
    pytest.mark.mm_w8a8_int8,
    pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="Hygon only"),
]


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


@pytest.mark.parametrize("code", [-128, 127])
def test_long_k_overflow(code):
    a, b, sa, sb = inputs(2, 3, 262145)
    a.fill_(code)
    b.fill_(code)
    sa.fill_(1)
    sb.fill_(1)
    y = flag_gems.mm_w8a8_int8(a, b, sa, sb, torch.float32)
    gems_assert_equal(y.cpu(), reference(a, b, sa, sb))


@pytest.mark.parametrize("a_scalar,b_scalar", [(True, False), (False, True)])
def test_mixed_scales(a_scalar, b_scalar):
    a, b, sa, sb = inputs(17, 13, 31)
    sa = sa[:1] if a_scalar else sa.flatten()
    sb = sb[:, :1].contiguous() if b_scalar else sb.flatten()
    y = flag_gems.mm_w8a8_int8(a, b, sa, sb)
    assert y.dtype == torch.bfloat16
    gems_assert_equal(y.cpu(), reference(a, b, sa, sb, dtype=torch.bfloat16))


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
