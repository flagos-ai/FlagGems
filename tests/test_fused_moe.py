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

import importlib

import pytest
import torch

import flag_gems

moe = importlib.import_module("flag_gems.fused.fused_moe")
pytestmark = pytest.mark.fused_experts_impl


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_w8a8_config_dtype(dtype):
    assert moe._get_config_dtype_str(dtype=dtype, use_int8_w8a8=True) == "int8_w8a8"


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"dtype": torch.float16}, "fp16"),
        ({"dtype": torch.bfloat16}, "bf16"),
        ({"dtype": torch.float32}, "float32"),
        ({"use_fp8_w8a8": True}, "fp8_w8a8"),
        ({"use_fp8_w8a16": True}, "fp8_w8a16"),
        ({"use_int8_w8a16": True}, "int8_w8a16"),
        ({"use_int4_w4a16": True}, "int4_w4a16"),
        ({"ocp_mx_scheme": "w_mxfp4"}, None),
    ],
)
def test_other_moe_config_dtypes_unchanged(kwargs, expected):
    assert moe._get_config_dtype_str(**kwargs) == expected


def test_w8a8_flag_reaches_config_lookup(monkeypatch):
    # Stop before device allocation/kernel launch: test the real entry point,
    # not just a direct call to the dtype helper.
    class ConfigReached(Exception):
        pass

    def check_config(**kwargs):
        assert kwargs["use_int8_w8a8"] is True
        assert kwargs["dtype"] == torch.bfloat16
        raise ConfigReached

    monkeypatch.setattr(moe, "_get_config_dtype_str", check_config)
    with pytest.raises(ConfigReached):
        moe.fused_experts_impl(
            torch.empty(4096, 128, dtype=torch.bfloat16, device="meta"),
            torch.empty(8, 256, 128, dtype=torch.int8, device="meta"),
            torch.empty(8, 128, 128, dtype=torch.int8, device="meta"),
            torch.empty(4096, 2, device="meta"),
            torch.empty(4096, 2, dtype=torch.int32, device="meta"),
            use_int8_w8a8=True,
            per_channel_quant=True,
        )


@pytest.mark.parametrize("num_tokens", [4095, 4096, 4097, 8192])
@pytest.mark.parametrize("gemm_stage", ["gemm1", "gemm2"])
def test_w8a8_excludes_plain_half_gemm_config(num_tokens, gemm_stage):
    dtype = moe._get_config_dtype_str(dtype=torch.bfloat16, use_int8_w8a8=True)
    assert dtype not in moe._PLAIN_HALF_CONFIG_DTYPES
    config = moe.get_default_config(
        num_tokens,
        8,
        256,
        128,
        2,
        dtype,
        gemm_stage=gemm_stage,
        enable_gemm_fast_path=True,
    )
    assert not config.get("PAIR_GATE_UP_DOT", False)
    # Enabling half-only tuning must have no effect on quantized configs.
    assert config == moe.get_default_config(
        num_tokens,
        8,
        256,
        128,
        2,
        dtype,
        gemm_stage=gemm_stage,
        enable_gemm_fast_path=False,
    )


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_plain_half_gemm_optimization_remains_enabled(dtype):
    config = moe.get_default_config(
        8192,
        8,
        256,
        128,
        2,
        dtype,
        gemm_stage="gemm1",
        enable_gemm_fast_path=True,
    )
    assert config["PAIR_GATE_UP_DOT"] is True


def _quantize_reference(x):
    scale = x.abs().amax(dim=-1, keepdim=True).float().clamp_min(1e-10) / 127
    quantized = (x.float() / scale).round().clamp(-128, 127)
    return quantized, scale


def _w8a8_reference(x, w1, w2, weights, ids, s1, s2):
    """Independent CPU oracle, including both dynamic quantizations.

    These cases have M > 8 and use fused SiLU: gate/up projections are not
    rounded to the output dtype until after the activation and multiplication.
    GEMM2 rounds each weighted expert output before the top-k reduction.
    """
    dtype = x.dtype
    x, w1, w2, weights, ids, s1, s2 = [
        t.cpu() for t in (x, w1, w2, weights, ids, s1, s2)
    ]
    qx, sx = _quantize_reference(x)
    routed = torch.zeros(*ids.shape, x.shape[1], dtype=dtype)
    for expert in range(w1.shape[0]):
        rows, slots = torch.where(ids == expert)
        gate_up = (qx[rows] @ w1[expert].float().T) * sx[rows] * s1[expert].T
        gate, up = gate_up.chunk(2, dim=-1)
        intermediate = (torch.nn.functional.silu(gate) * up).to(dtype)
        qi, si = _quantize_reference(intermediate)
        out = (qi @ w2[expert].float().T) * si * s2[expert].T
        routed[rows, slots] = (out * weights[rows, slots, None]).to(dtype)
    return routed.float().sum(dim=1).to(dtype)


@pytest.mark.parametrize("num_tokens", [32, 4095, 4096, 4097, 8192])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("execution_mode", ["eager", "graph"])
def test_w8a8_large_m_accuracy(monkeypatch, num_tokens, dtype, execution_mode):
    if execution_mode == "graph" and (
        flag_gems.device != "cuda" or not torch.cuda.is_available()
    ):
        pytest.skip("This graph regression requires the CUDA-compatible graph API")

    # Force the fallback configuration even on devices with embedded tuning.
    monkeypatch.setattr(moe, "get_moe_configs", lambda *args, **kwargs: None)
    generator = torch.Generator(device="cpu").manual_seed(20260904)
    e, k, i, topk = 8, 128, 128, 2
    # Isolate scale propagation from CPU/device INT8 rounding ties. Each
    # projection selects one feature: gate=32, up=code/4, hence SiLU(gate)*up
    # rounds to 8*code. With max code=127, the second scale is exactly 8.
    # Non-unit per-channel weight scales are essential to expose the old path.
    x = torch.full((num_tokens, k), 32, dtype=dtype)
    x[:, 1:] *= torch.randint(0, 2, (num_tokens, k - 1), generator=generator) * 2 - 1
    w1 = torch.zeros(e, 2 * i, k, dtype=torch.int8)
    w1[:, :i, 0] = 1
    channels = torch.arange(i)
    codes = (channels % 127 + 1).to(torch.int8)
    codes[0] = 127
    w1[:, i + channels, channels % k] = codes
    w2 = torch.randint(-8, 9, (e, k, i), generator=generator, dtype=torch.int8)
    s1 = torch.ones(e, 2 * i, 1)
    s1[:, i:] = 1 / 128
    s2 = ((torch.arange(e * k).reshape(e, k, 1) % 4 + 1).float() / 1024).contiguous()
    logits = torch.randn(num_tokens, e, generator=generator)
    _, ids = logits.topk(topk, dim=-1)
    weights = torch.tensor([0.25, 0.75]).expand(num_tokens, -1).contiguous()
    ids = ids.to(torch.int32)
    ref = _w8a8_reference(x, w1, w2, weights, ids, s1, s2)
    x, w1, w2, weights, ids, s1, s2 = [
        t.to(flag_gems.device) for t in (x, w1, w2, weights, ids, s1, s2)
    ]

    def call():
        return moe.fused_experts_impl(
            x,
            w1,
            w2,
            weights,
            ids,
            use_int8_w8a8=True,
            per_channel_quant=True,
            w1_scale=s1,
            w2_scale=s2,
        )

    if execution_mode == "eager":
        out = call()
        torch.testing.assert_close(out.cpu(), ref, atol=0.003, rtol=0.04)
        return

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()  # Compile/allocate before capture.
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = call()
    graph.replay()
    torch.testing.assert_close(out.cpu(), ref, atol=0.003, rtol=0.04)
    # Replay must observe changed values, not just the capture-time buffers.
    x.neg_()
    graph.replay()
    changed_ref = _w8a8_reference(x, w1, w2, weights, ids, s1, s2)
    torch.testing.assert_close(out.cpu(), changed_ref, atol=0.003, rtol=0.04)
