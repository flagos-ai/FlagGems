# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

from itertools import product

import pytest
import torch

import flag_gems
from benchmark.test_bmm_w8a8 import make_channel_int8_bmm_inputs

from .accuracy_utils import gems_assert_close, gems_assert_equal

CONTRACT_SHAPES = [
    (1, 1, 1, 1),
    (2, 33, 65, 129),
    (4, 64, 128, 256),
    (2, 129, 17, 257),
    (1, 1, 1024, 4096),
    (1, 4, 1024, 4096),
    (1, 16, 1024, 4096),
    (1, 32, 1024, 4096),
    (1, 64, 1024, 4096),
    (1, 65, 1024, 4096),
    (1, 71, 1024, 4096),
    (1, 72, 1024, 4096),
    (1, 160, 1024, 4096),
    (1, 161, 1024, 4096),
    (1, 224, 1024, 4096),
    (1, 225, 1024, 4096),
    (1, 321, 1024, 4096),
    (1, 512, 1024, 4096),
    (1, 513, 1024, 4096),
]
BMM_CASES = list(
    product(
        CONTRACT_SHAPES,
        (torch.bfloat16, torch.float16, torch.float32),
        (2, 3),
        (False, True),
    )
)


requires_channel_int8_bmm = pytest.mark.skipif(
    "bmm_w8a8" not in vars(flag_gems),
    reason="backend does not export channel-wise INT8 BMM",
)


@pytest.mark.parametrize("shape,dtype,scale_rank,use_out", BMM_CASES)
@pytest.mark.bmm_w8a8
@requires_channel_int8_bmm
def test_channel_int8_bmm(shape, dtype, scale_rank, use_out):
    batch, rows, columns = shape[:3]
    activation, weight, activation_scale, weight_scale = make_channel_int8_bmm_inputs(
        shape, flag_gems.device
    )
    reference = torch.bmm(activation.cpu().long(), weight.cpu().long()).float()
    reference *= activation_scale.cpu()[:, :, None]
    reference *= weight_scale.cpu()[:, None, :]
    activation_scale = (
        activation_scale[:, :, None] if scale_rank == 3 else activation_scale
    )
    weight_scale = weight_scale[:, None, :] if scale_rank == 3 else weight_scale
    output = (
        torch.empty((batch, rows, columns), device=activation.device, dtype=dtype)
        if use_out
        else None
    )
    actual = flag_gems.bmm_w8a8(
        activation, weight, activation_scale, weight_scale, out_dtype=dtype, out=output
    )
    gems_assert_close(actual.cpu(), reference.to(dtype), dtype, atol=0)


@pytest.mark.parametrize("use_out", [False, True])
@pytest.mark.parametrize(
    "shape", [(0, 5, 7, 131072), (2, 0, 7, 32), (2, 5, 0, 32), (2, 5, 7, 0)]
)
@pytest.mark.bmm_w8a8
@requires_channel_int8_bmm
def test_channel_int8_bmm_empty(shape, use_out):
    batch, rows, columns, reduction = shape
    activation = torch.empty(
        shape[:2] + (reduction,), device=flag_gems.device, dtype=torch.int8
    )
    weight = torch.empty(
        (batch, columns, reduction), device=activation.device, dtype=torch.int8
    ).transpose(1, 2)
    activation_scale = torch.ones((batch, rows), device=activation.device)
    weight_scale = torch.ones((batch, columns), device=activation.device)
    output = (
        torch.empty(
            (batch, rows, columns), device=activation.device, dtype=torch.bfloat16
        )
        if use_out
        else None
    )
    actual = flag_gems.bmm_w8a8(
        activation, weight, activation_scale, weight_scale, out=output
    )
    reference = torch.zeros(
        (batch, rows, columns), device=activation.device, dtype=torch.bfloat16
    )
    gems_assert_equal(actual.cpu(), reference.cpu())


@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "shape",
    [
        (2, 17, 37, 65),
        (3, 64, 80, 128),
        (2, 11, 13, 0),
        (1, 0, 13, 19),
        (1, 31, 47, 257),
    ],
)
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.bmm_w8a8
@requires_channel_int8_bmm
@pytest.mark.skipif(
    flag_gems.vendor_name not in ("metax", "mthreads"),
    reason="backend does not support the arbitrary-stride layout extension",
)
def test_bmm_w8a8(out_dtype, shape, transpose):
    torch.manual_seed(20261019)
    batch, m, n, k = shape
    device = flag_gems.device
    b_shape = (batch, n, k * 2) if transpose else (batch, k, n * 2)
    a = torch.randint(-128, 128, (batch, m, k * 2), device=device, dtype=torch.int8)
    a = a[:, :, ::2]
    b = torch.randint(-128, 128, b_shape, device=device, dtype=torch.int8)[:, :, ::2]
    if transpose:
        b = b.transpose(1, 2)
    sa = (torch.rand(batch, m * 2, device=device) * 0.005)[:, ::2]
    sb = (torch.rand(batch, n * 2, device=device) * 0.005)[:, ::2]
    storage = torch.full((batch, n * 2, m * 2), 23, device=device, dtype=out_dtype)
    out = storage.transpose(1, 2)[:, ::2, ::2]
    # Match the specified INT32 accumulation followed by FP32 scale rounding.
    expected = torch.bmm(a.cpu().long(), b.cpu().long()).float()
    expected *= sa.cpu()[:, :, None]
    expected *= sb.cpu()[:, None, :]
    actual = flag_gems.bmm_w8a8(
        a, b, sa[:, :, None], sb[:, None, :], out_dtype=out_dtype, out=out
    )
    assert actual is out
    gems_assert_close(actual.cpu(), expected.to(out_dtype), dtype=out_dtype)
    untouched = torch.ones(storage.shape, dtype=torch.bool)
    untouched[:, ::2, ::2] = False
    untouched_values = storage.cpu()[untouched]
    gems_assert_equal(untouched_values, torch.full_like(untouched_values, 23))


@pytest.mark.parametrize(
    "shape",
    [
        (1, 97, 80, 128),
        (2, 64, 80, 128),
        (3, 129, 256, 256),
        (1, 257, 128, 7168),
        (2, 64, 32, 32),
        (1, 163, 2112, 1024),
    ],
)
@pytest.mark.bmm_w8a8
@requires_channel_int8_bmm
def test_bmm_w8a8_aligned(shape):
    torch.manual_seed(20261020)
    batch, m, n, k = shape
    a = torch.randint(
        -128, 128, (batch, m, k), device=flag_gems.device, dtype=torch.int8
    )
    b = torch.randint(
        -128, 128, (batch, n, k), device=flag_gems.device, dtype=torch.int8
    ).transpose(1, 2)
    sa = torch.rand(batch, m, device=flag_gems.device) * 0.001
    sb = torch.rand(batch, n, device=flag_gems.device) * 0.001
    expected = torch.bmm(a.cpu().long(), b.cpu().long()).float()
    expected *= sa.cpu()[:, :, None]
    expected *= sb.cpu()[:, None, :]
    actual = flag_gems.bmm_w8a8(a, b, sa, sb)
    gems_assert_close(actual.cpu(), expected.bfloat16(), dtype=torch.bfloat16)


@pytest.mark.skipif(flag_gems.vendor_name != "metax", reason="MetaX graph fast path")
@pytest.mark.bmm_w8a8
@requires_channel_int8_bmm
def test_bmm_w8a8_graph_replay():
    torch.manual_seed(20261020)
    a = torch.randint(
        -128, 128, (2, 64, 128), device=flag_gems.device, dtype=torch.int8
    )
    b = torch.randint(
        -128, 128, (2, 80, 128), device=flag_gems.device, dtype=torch.int8
    ).transpose(1, 2)
    sa = torch.rand(2, 64, device=flag_gems.device) * 0.001
    sb = torch.rand(2, 80, device=flag_gems.device) * 0.001
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            flag_gems.bmm_w8a8(a, b, sa, sb)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = flag_gems.bmm_w8a8(a, b, sa, sb)
    for _ in range(3):
        a.random_(-128, 128)
        sa.uniform_(0, 0.001)
        sb.uniform_(0, 0.001)
        graph.replay()
        torch.cuda.synchronize()
        expected = (
            torch.bmm(a.double(), b.double()) * sa[:, :, None] * sb[:, None, :]
        ).bfloat16()
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=0.01)


# FlagOSTune f4f65a7d779a0df3b017e9a80b75a7b9d651c639, DeepSeek-V4-Flash.
# Reuse contraction dimensions, with channel-wise rather than block-wise scales.
EINSUM_SHAPES = [
    (m, 1, 1024, 4096)
    for m in (
        16384,
        496,
        512,
        1,
        104,
        112,
        120,
        128,
        136,
        144,
        152,
        16,
        160,
        168,
        176,
        184,
        192,
        2,
        200,
        208,
        216,
        224,
        232,
        24,
        240,
        248,
        256,
        272,
        288,
        304,
        32,
        320,
        336,
        352,
        368,
        384,
        4,
        40,
        400,
        416,
        432,
        448,
        464,
        48,
        480,
        56,
        64,
        72,
        8,
        80,
        88,
        96,
        2048,
        4100,
        13021,
        1028,
        3805,
        3076,
        12991,
        13005,
    )
]


@pytest.mark.bmm_w8a8
@requires_channel_int8_bmm
@pytest.mark.parametrize("shape", EINSUM_SHAPES)
def test_bmm_w8a8_einsum_shapes(shape, monkeypatch):
    torch.manual_seed(20261027)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    b, h, d, r = shape
    device = flag_gems.device
    # bhr,hdr->bhd becomes (h,b,r) @ (h,r,d), followed by a view transpose.
    x = (torch.randn(b, h, r, device=device) * 3).to(torch.int8)
    y = (torch.randn(h, d, r, device=device) * 3).to(torch.int8)
    xs = torch.rand(b, h, device=device) * 0.01
    ys = torch.rand(h, d, device=device) * 0.01
    expected = (
        torch.einsum("bhr,hdr->bhd", x.float(), y.float())
        * xs[:, :, None]
        * ys[None, :, :]
    ).bfloat16()
    actual = flag_gems.bmm_w8a8(
        x.permute(1, 0, 2), y.transpose(1, 2), xs.transpose(0, 1), ys
    ).transpose(0, 1)
    assert actual.shape == (b, h, d)
    torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.01)


@pytest.mark.skipif(flag_gems.vendor_name != "metax", reason="MetaX graph fast paths")
@pytest.mark.bmm_w8a8
@requires_channel_int8_bmm
@pytest.mark.parametrize("m", [1, 2, 4, 8, 16, 64, 128, 256, 512])
def test_bmm_w8a8_einsum_graph_full_range(m):
    torch.manual_seed(20261027)
    n, k = 1024, 4096
    device = flag_gems.device
    codes = torch.arange(-128, 128, dtype=torch.int16).to(torch.int8)
    codes = codes.to(device)

    def sample(shape):
        return codes[torch.randint(codes.numel(), shape, device=device)].view(
            torch.int8
        )

    a = sample((1, m, k))
    b = sample((1, n, k)).transpose(1, 2)
    scale_limit = 1e-3
    sa = torch.rand(1, m, device=device) * scale_limit
    sb = torch.rand(1, n, device=device) * scale_limit
    out = torch.empty((1, m, n), device=device, dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            actual = flag_gems.bmm_w8a8(a, b, sa, sb, out=out)
    assert actual is out
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        flag_gems.bmm_w8a8(a, b, sa, sb, out=out)
    for _ in range(2):
        a.copy_(sample(a.shape))
        b.copy_(sample((1, n, k)).transpose(1, 2))
        sa.uniform_(0, scale_limit)
        sb.uniform_(0, scale_limit)
        graph.replay()
        torch.cuda.synchronize()
        expected = (
            torch.bmm(a.double(), b.double()) * sa[:, :, None] * sb[:, None, :]
        ).bfloat16()
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=0.01)


@pytest.mark.bmm_w8a8
@requires_channel_int8_bmm
@pytest.mark.parametrize("operand", ["a", "b"])
def test_bmm_w8a8_rejects_non_int8(operand):
    a = torch.zeros((1, 2, 32), device=flag_gems.device, dtype=torch.int8)
    b = torch.zeros((1, 32, 16), device=flag_gems.device, dtype=torch.int8)
    sa = torch.ones((1, 2), device=flag_gems.device)
    sb = torch.ones((1, 16), device=flag_gems.device)
    if operand == "a":
        a = a.float()
    else:
        b = b.float()
    with pytest.raises(TypeError, match="INT8"):
        flag_gems.bmm_w8a8(a, b, sa, sb)


@pytest.mark.bmm_w8a8
@requires_channel_int8_bmm
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("call_style", ["positional", "keyword"])
def test_channel_int8_bmm_output_arguments(out_dtype, call_style):
    activation, weight, activation_scale, weight_scale = make_channel_int8_bmm_inputs(
        (1, 3, 16, 32), flag_gems.device
    )
    out = torch.empty((1, 3, 16), device=flag_gems.device, dtype=out_dtype)
    if call_style == "positional":
        actual = flag_gems.bmm_w8a8(
            activation, weight, activation_scale, weight_scale, out_dtype, out
        )
    else:
        actual = flag_gems.bmm_w8a8(
            activation=activation,
            weight=weight,
            activation_scale=activation_scale,
            weight_scale=weight_scale,
            out_dtype=out_dtype,
            out=out,
        )
    expected = torch.bmm(activation.cpu().long(), weight.cpu().long()).float()
    expected *= activation_scale.cpu()[:, :, None]
    expected *= weight_scale.cpu()[:, None, :]
    assert actual is out
    gems_assert_close(actual.cpu(), expected.to(out_dtype), dtype=out_dtype)


@pytest.mark.bmm_w8a8
@requires_channel_int8_bmm
@pytest.mark.skipif(
    flag_gems.vendor_name not in ("metax", "mthreads"),
    reason="backend does not support the explicit generic-kernel config",
)
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_channel_int8_bmm_keyword_config(out_dtype):
    activation, weight, activation_scale, weight_scale = make_channel_int8_bmm_inputs(
        (1, 3, 17, 33), flag_gems.device
    )
    out = torch.empty((1, 3, 17), device=flag_gems.device, dtype=out_dtype)
    actual = flag_gems.bmm_w8a8(
        activation,
        weight,
        activation_scale,
        weight_scale,
        out_dtype,
        out,
        block_size=(128, 128, 128),
        config=(16, 32, 32, 4, 1, 1),
    )
    expected = torch.bmm(activation.cpu().long(), weight.cpu().long()).float()
    expected *= activation_scale.cpu()[:, :, None]
    expected *= weight_scale.cpu()[:, None, :]
    assert actual is out
    gems_assert_close(actual.cpu(), expected.to(out_dtype), dtype=out_dtype)


@pytest.mark.parametrize(
    "invalid_input", ["batch", "weight_layout", "scale_shape", "scale_dtype"]
)
@pytest.mark.bmm_w8a8
@requires_channel_int8_bmm
@pytest.mark.skipif(
    not flag_gems.bmm_w8a8.__module__.endswith("_thead.ops.bmm_w8a8_int8"),
    reason="THead layout and scale contract",
)
def test_channel_int8_bmm_input_contract(invalid_input):
    activation = torch.empty((2, 8, 16), device=flag_gems.device, dtype=torch.int8)
    weight = torch.empty(
        (2, 32, 16), device=activation.device, dtype=torch.int8
    ).transpose(1, 2)
    activation_scale = torch.ones((2, 8), device=activation.device)
    weight_scale = torch.ones((2, 32), device=activation.device)
    error_type = ValueError
    if invalid_input == "batch":
        weight = weight[:1]
    elif invalid_input == "weight_layout":
        weight = weight.contiguous()
    elif invalid_input == "scale_shape":
        activation_scale = activation_scale[:, :7]
    else:
        weight_scale = weight_scale.to(torch.int8)
        error_type = TypeError
    with pytest.raises(error_type):
        flag_gems.bmm_w8a8(activation, weight, activation_scale, weight_scale)


@pytest.mark.bmm_w8a8
@requires_channel_int8_bmm
@pytest.mark.parametrize("batch,m,n", [(2, 3, 3), (2, 65, 3), (1, 128, 1024)])
def test_channel_int8_bmm_int32_accumulation(batch, m, n):
    torch.manual_seed(20261112)
    k = 4096
    activation = torch.full((batch, m, k), 127, dtype=torch.int8)
    activation[:, 1, :] = (
        torch.arange(-128, 128, dtype=torch.int16).to(torch.int8).repeat(16)
    )
    activation[:, 2, :] = -128
    positive = torch.full((k // 2,), 127, dtype=torch.int8)
    positive[63::64] = 126
    weight = torch.full((batch, n, k), -128, dtype=torch.int8)
    weight[:, 0, :] = torch.cat((positive, -positive))
    weight[:, 2, :] = activation[:, 1, :]
    reference = torch.bmm(activation.long(), weight.transpose(1, 2).long()).float()
    activation = activation.to(flag_gems.device)
    weight = weight.to(flag_gems.device).transpose(1, 2)
    activation_scale = torch.ones((batch, m), device=flag_gems.device)
    weight_scale = torch.ones((batch, n), device=flag_gems.device)
    actual = flag_gems.bmm_w8a8(
        activation, weight, activation_scale, weight_scale, torch.float32
    )
    # Cancellation after large positive partial sums distinguishes INT32 from
    # an unbounded FP32 reduction; the other channels cover signed INT8 codes.
    gems_assert_equal(actual.cpu(), reference)


@pytest.mark.bmm_w8a8
@requires_channel_int8_bmm
@pytest.mark.skipif(
    flag_gems.vendor_name not in ("metax", "mthreads"),
    reason="backend does not support the arbitrary-stride layout extension",
)
def test_channel_int8_bmm_large_batch_stride():
    torch.manual_seed(20261112)
    # Each batch stride fits INT32, but the third batch crosses its signed range.
    stride = 2**30 + 128
    storage = torch.empty(2 * stride + 32, device=flag_gems.device, dtype=torch.int8)
    activation = storage.as_strided((3, 1, 32), (stride, 32, 1))
    activation.copy_(
        torch.tensor([7, 11, 13], dtype=torch.int8).view(3, 1, 1).expand(3, 1, 32)
    )
    weight = torch.full((3, 32, 1), 3, device=flag_gems.device, dtype=torch.int8)
    activation_scale = torch.ones((3, 1), device=flag_gems.device)
    weight_scale = torch.ones((3, 1), device=flag_gems.device)
    actual = flag_gems.bmm_w8a8(
        activation, weight, activation_scale, weight_scale, torch.float32
    )
    reference = torch.tensor([672, 1056, 1248], dtype=torch.float32).view(3, 1, 1)
    gems_assert_equal(actual.cpu(), reference)
