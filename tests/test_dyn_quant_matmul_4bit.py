# Copyright 2026, The FlagOS Contributors.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import flag_gems


def _reference(x, weights, scales, bias, group):
    """Independent CPU quantization and integer dot, with FP32 scale arithmetic."""
    x = x.float()
    lo = x.amin(1).clamp(max=0)
    hi = x.amax(1).clamp(min=0)
    mult = torch.where(lo == hi, 1.0, torch.div(torch.full_like(lo, 255.0), hi - lo))
    scale = torch.where(mult != 0, torch.div(torch.ones_like(mult), mult), 0.0)
    a, b = lo * mult, hi * mult
    zp = torch.where((-128.0 + a) + (127.0 + b) > 0, -128.0 - a, 127.0 - b)
    zp = zp.clamp(-128, 127).round().to(torch.int32)
    values = x * mult[:, None]
    rounded = values.sign() * (values.abs() + 0.5).floor()
    q = (rounded.to(torch.int32) + zp[:, None]).clamp(-128, 127) - zp[:, None]
    n, half_k = weights.shape
    w = torch.empty((n, half_k * 2), dtype=torch.int32)
    w[:, 0::2] = (weights.to(torch.int32) & 15) - 8
    w[:, 1::2] = (weights.to(torch.int32) >> 4) - 8
    result = torch.zeros((x.shape[0], n), dtype=torch.float32)
    for g in range(x.shape[1] // group):
        start = g * group
        dot = q[:, start : start + group].to(torch.int64) @ w[
            :, start : start + group
        ].T.to(torch.int64)
        result += dot.to(torch.int32).float() * scales[:, g].float()
    result *= scale[:, None]
    if bias is not None:
        result += bias.float()
    return result


def _case(m, n, k, group, dtype, bias, noncontiguous=False):
    gen = torch.Generator().manual_seed(43)
    x = torch.randn((m, k), generator=gen).to(dtype)
    w = torch.randint(0, 256, (n, k // 2), generator=gen, dtype=torch.uint8)
    scales = torch.rand((n, k // group), generator=gen) + 0.1
    b = torch.randn(n, generator=gen) if bias else None
    expected = (
        _reference(x, w, scales, b, group).to(dtype)
        if m
        else torch.empty((m, n), dtype=dtype)
    )
    xd = x.to(flag_gems.device)
    if noncontiguous:
        storage = torch.empty((k, m * 2), dtype=dtype, device=flag_gems.device)
        xd = storage[:, ::2].T
        xd.copy_(x)
    packed = flag_gems._dyn_quant_pack_4bit_weight(
        w.to(flag_gems.device),
        scales.to(flag_gems.device),
        None if b is None else b.to(flag_gems.device),
        group,
        k,
        n,
    )
    actual = flag_gems._dyn_quant_matmul_4bit(xd, packed, group, k, n)
    assert actual.dtype == dtype and actual.device == xd.device
    torch.testing.assert_close(
        actual.cpu(),
        expected,
        rtol=1e-5 if dtype == torch.float32 else 0.008,
        atol=2e-5 if dtype == torch.float32 else 0.008,
    )


@pytest.mark.dyn_quant_matmul_4bit
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize(
    "m,n,k",
    [
        (0, 31, 64),
        (1, 0, 64),
        (1, 1, 2),
        (7, 31, 30),
        (32, 48, 66),
        (1, 4096, 128),
        (7, 64, 1024),
        (32, 33, 4096),
        (1, 1, 16386),
    ],
)
def test_dyn_quant_matmul_4bit_channelwise(m, n, k, dtype, bias):
    _case(m, n, k, k, dtype, bias)


@pytest.mark.dyn_quant_matmul_4bit
@pytest.mark.parametrize(
    "m,n,k,group",
    [
        (1, 1, 32, 32),
        (7, 31, 64, 32),
        (32, 48, 96, 32),
        (7, 64, 128, 64),
        (1, 4096, 128, 32),
        (7, 33, 192, 96),
    ],
)
@pytest.mark.parametrize("bias", [False, True])
def test_dyn_quant_matmul_4bit_groupwise(m, n, k, group, bias):
    _case(m, n, k, group, torch.float32, bias)


@pytest.mark.dyn_quant_matmul_4bit
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_dyn_quant_matmul_4bit_noncontiguous(dtype):
    _case(7, 31, 66, 66, dtype, True, True)


@pytest.mark.dyn_quant_matmul_4bit
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_dyn_quant_matmul_4bit_rounding_and_nibbles(dtype):
    # Range [-128,127] makes multiplier exactly one; half-integers distinguish
    # away-from-zero activation rounding from nearest-even zero-point rounding.
    x = torch.tensor(
        [
            [-128, 127, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [3] * 8,
            [-3] * 8,
            [-127.5, 127.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5],
            [3e38, -3e38, 0, 1, -1, 2, -2, 0],
        ],
        dtype=dtype,
    )
    w = torch.tensor([[0xF0, 0x0F, 0x81, 0x18], [0xFF] * 4, [0] * 4], dtype=torch.uint8)
    s = torch.tensor([[1.0], [0.5], [2.0]])
    ref = _reference(x, w, s, None, 8).to(dtype)
    packed = flag_gems._dyn_quant_pack_4bit_weight(
        w.to(flag_gems.device), s.to(flag_gems.device), None, 8, 8, 3
    )
    actual = flag_gems._dyn_quant_matmul_4bit(x.to(flag_gems.device), packed, 8, 8, 3)
    torch.testing.assert_close(actual.cpu(), ref, rtol=0, atol=0)


@pytest.mark.dyn_quant_matmul_4bit
@pytest.mark.parametrize(
    "fault,match",
    [
        ("rank", "two-dimensional"),
        ("dtype", "float32 or bfloat16"),
        ("zero_k", "positive and even"),
        ("odd_k", "positive and even"),
        ("negative_n", "nonnegative"),
        ("mismatch_k", "inp.size"),
        ("zero_block", "block_size"),
        ("negative_block", "block_size"),
        ("bad_block", "block_size"),
        ("nondividing_block", "block_size"),
        ("bf16_group", "bfloat16 requires"),
        ("opaque", "opaque uint8"),
        ("packed_dtype", "float32 portable"),
        ("packed_rank", "one-dimensional"),
        ("packed_stride", "contiguous"),
        ("short", "length"),
        ("long", "length"),
        ("device", "same device"),
    ],
)
def test_dyn_quant_matmul_4bit_invalid(fault, match):
    x = torch.empty((2, 64), device=flag_gems.device)
    p = torch.empty((66,), device=flag_gems.device)
    block, k, n = 64, 64, 2
    if fault == "rank":
        x = x.unsqueeze(0)
    elif fault == "dtype":
        x = x.to(torch.float16)
    elif fault == "zero_k":
        k = 0
    elif fault == "odd_k":
        k = 63
    elif fault == "negative_n":
        n = -1
    elif fault == "mismatch_k":
        k = 32
    elif fault == "zero_block":
        block = 0
    elif fault == "negative_block":
        block = -32
    elif fault == "bad_block":
        block = 16
    elif fault == "nondividing_block":
        block = 96
    elif fault == "bf16_group":
        x = x.to(torch.bfloat16)
        block = 32
    elif fault == "opaque":
        p = p.to(torch.uint8)
    elif fault == "packed_dtype":
        p = p.to(torch.bfloat16)
    elif fault == "packed_rank":
        p = p.reshape(2, 33)
    elif fault == "packed_stride":
        p = torch.empty(132, device=flag_gems.device)[::2]
    elif fault == "short":
        p = p[:-1]
    elif fault == "long":
        p = torch.empty(69, device=flag_gems.device)
    elif fault == "device":
        p = p.cpu()
    with pytest.raises(RuntimeError, match=match):
        flag_gems._dyn_quant_matmul_4bit(x, p, block, k, n)


@pytest.mark.dyn_quant_matmul_4bit
def test_dyn_quant_matmul_4bit_compile():
    x = torch.randn((7, 64), device=flag_gems.device)
    w = torch.randint(0, 256, (31, 32), dtype=torch.uint8, device=flag_gems.device)
    s = torch.ones((31, 1), device=flag_gems.device)
    packed = flag_gems._dyn_quant_pack_4bit_weight(w, s, None, 64, 64, 31)
    compiled = torch.compile(
        flag_gems._dyn_quant_matmul_4bit, backend="eager", fullgraph=True
    )
    expected = flag_gems._dyn_quant_matmul_4bit(x, packed, 64, 64, 31)
    actual = compiled(x, packed, 64, 64, 31)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
