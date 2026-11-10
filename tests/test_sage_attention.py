# Copyright 2026, The FlagOS Contributors.
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
#

"""Quantization-aware references; dense SDPA is not an FP8 Sage oracle."""

import pytest
import torch

import flag_gems

pytestmark = pytest.mark.sage_attention


def quantized_reference(q, k, v):
    """Independent CPU per-thread INT8 QK / BF16 PV, one128-key tile."""
    q, k, v = [x.detach().cpu().permute(0, 2, 1, 3) for x in (q, k, v)]
    k = (k.float() - k.float().mean(2, keepdim=True).to(k.dtype).float()).to(k.dtype)
    nq, nk = q.shape[2], k.shape[2]
    qf, kf = q.float(), k.float()
    qs = torch.empty(q.shape[:-1])
    ks = torch.empty(k.shape[:-1])
    for start in range(0, nq, 64):
        for group in range(32):
            rows = [start + group // 8 * 16 + group % 8 + off for off in (0, 8)]
            rows = [r for r in rows if r < nq]
            if rows:
                scale = qf[:, :, rows].abs().amax((-2, -1)) / 127.0 + 1e-7
                qs[:, :, rows] = scale[..., None]
    for group in range(4):
        rows = [r for r in range(nk) if (r % 8) // 2 == group]
        if rows:
            scale = kf[:, :, rows].abs().amax((-2, -1)) / 127.0 + 1e-7
            ks[:, :, rows] = scale[..., None]
    qscaled, kscaled = qf / qs[..., None], kf / ks[..., None]
    qi = torch.trunc(qscaled + 0.5 * torch.where(qscaled >= 0, 1.0, -1.0))
    ki = torch.trunc(kscaled + 0.5 * torch.where(kscaled >= 0, 1.0, -1.0))
    score = (qi @ ki.transpose(-1, -2)) * (qs[..., None] * ks[..., None, :])
    score = score * (128**-0.5)
    # Probabilities are rounded before PV, but the denominator stays FP32.
    p = torch.exp(score - score.amax(-1, keepdim=True))
    out = (p.bfloat16().float() @ v.float()) / p.sum(-1, keepdim=True)
    return out.to(q.dtype).permute(0, 2, 1, 3)


@pytest.mark.parametrize("nq,nk", [(1, 1), (17, 31), (63, 127), (65, 128)])
@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_sage_bf16_quantized_reference(nq, nk, layout):
    torch.manual_seed(123)
    q = torch.randn((1, nq, 2, 128), device=flag_gems.device, dtype=torch.bfloat16)
    k, v = [
        torch.randn((1, nk, 2, 128), device=q.device, dtype=q.dtype) for _ in range(2)
    ]
    reference = quantized_reference(q, k, v)
    before = [x.clone() for x in (q, k, v)]
    inputs = [x.permute(0, 2, 1, 3) if layout == "HND" else x for x in (q, k, v)]
    actual = flag_gems.sage_attention(
        *inputs, tensor_layout=layout, implementation="baseline"
    )
    if layout == "HND":
        actual = actual.permute(0, 2, 1, 3)
    torch.testing.assert_close(actual.cpu(), reference, atol=1e-3, rtol=1e-3)
    for original, saved in zip((q, k, v), before):
        assert torch.equal(original, saved)


@pytest.mark.parametrize(
    "nq,nk,h",
    [
        (1, 1, 2),
        (17, 31, 2),
        (63, 63, 2),
        (64, 64, 2),
        (65, 65, 2),
        (127, 127, 2),
        (128, 128, 2),
        (129, 257, 2),
        (1108, 1792, 2),
    ]
    + [(44, 44, h) for h in (56, 28, 14, 7)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_sage_sm90_matches_native(nq, nk, h, dtype, layout):
    if flag_gems.vendor_name != "nvidia" or torch.cuda.get_device_capability() != (
        9,
        0,
    ):
        pytest.skip("Native FP8 Sage reference requires SM90")
    native = pytest.importorskip("sageattention").sageattn_qk_int8_pv_fp8_cuda_sm90
    torch.manual_seed(456)
    q = torch.randn((1, nq, h, 128), device="cuda", dtype=dtype)
    k, v = [
        torch.randn((1, nk, h, 128), device=q.device, dtype=q.dtype) for _ in range(2)
    ]
    inputs = [x.permute(0, 2, 1, 3) if layout == "HND" else x for x in (q, k, v)]
    expected = native(
        *inputs,
        tensor_layout=layout,
        is_causal=False,
        pv_accum_dtype="fp32+fp32",
        qk_quant_gran="per_thread",
    )
    actual = flag_gems.sage_attention(
        *inputs, tensor_layout=layout, pv_dtype="fp8", implementation="sm90"
    )
    assert torch.equal(actual, expected)


def test_sage_graph_refresh():
    if flag_gems.vendor_name != "nvidia" or torch.cuda.get_device_capability() != (
        9,
        0,
    ):
        pytest.skip("This test uses the CUDA Graph API")
    native = pytest.importorskip("sageattention").sageattn_qk_int8_pv_fp8_cuda_sm90
    inputs = [
        torch.randn((1, 44, 2, 128), device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    ]
    flag_gems.sage_attention(*inputs, pv_dtype="fp8")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = flag_gems.sage_attention(*inputs, pv_dtype="fp8")
    for _ in range(3):
        for x in inputs:
            x.normal_()
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        expected = native(
            *inputs,
            tensor_layout="NHD",
            is_causal=False,
            pv_accum_dtype="fp32+fp32",
            qk_quant_gran="per_thread",
        )
        assert torch.equal(output, expected)


def test_sage_unsupported_semantics_rejected():
    q = torch.empty((1, 1, 2, 128), device=flag_gems.device, dtype=torch.bfloat16)
    for kwargs in (
        {"is_causal": True},
        {"return_lse": True},
        {"tensor_layout": "bad"},
        {"sm_scale": 0},
    ):
        with pytest.raises((ValueError, NotImplementedError)):
            flag_gems.sage_attention(q, q, q, **kwargs)


def _sage_bytes_equal(actual, expected):
    # FP8 and nonfinite intermediates need bitwise comparison, not allclose.
    assert torch.equal(
        actual.contiguous().view(torch.uint8),
        expected.contiguous().view(torch.uint8),
    )


@pytest.mark.parametrize("n", [1, 17, 44, 63, 64, 65, 127, 128, 129])
@pytest.mark.parametrize("h", [7, 56])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("pattern", ["random", "zero", "nonfinite"])
def test_sage_sm90_short_quantization(n, h, dtype, layout, strided, pattern):
    """Fused paths must match the unfused device stages, including padding.

    Native output equality is checked separately. The unfused stages allow us
    to compare nonfinite values and weakly aligned strides without requiring
    the native extension to accept those layouts or define NaN output equality.
    """
    if flag_gems.vendor_name != "nvidia" or torch.cuda.get_device_capability() != (
        9,
        0,
    ):
        pytest.skip("Short FP8 Sage regression requires SM90")
    import triton

    from flag_gems.runtime.backend._nvidia.hopper.ops.sage_attention_impl import (
        quantization as quant,
    )
    from flag_gems.runtime.backend._nvidia.hopper.ops.sage_attention_impl.native_mean import (
        native_k_mean,
    )

    torch.manual_seed(234)
    if strided:
        tensors = [
            torch.randn((1, n, h, 132), device="cuda", dtype=dtype)[..., 1:129]
            for _ in range(3)
        ]
    else:
        tensors = [
            torch.randn((1, n, h, 128), device="cuda", dtype=dtype) for _ in range(3)
        ]
    if layout == "HND" and not strided:
        tensors = [
            x.permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3) for x in tensors
        ]
    for x in tensors[1:]:
        if pattern == "zero":
            x.zero_()
            x[..., ::2] = -0.0
        elif pattern == "nonfinite":
            x[..., 0] = float("nan")
            x[..., 1] = float("inf")
            x[..., 2] = -float("inf")
    inputs = [x.permute(0, 2, 1, 3) if layout == "HND" else x for x in tensors]
    _, _, ki, ks, vf, vs, mean = quant.quantize_sage_sm90(*inputs, tensor_layout=layout)
    key, value = tensors[1:]
    expected_mean = native_k_mean(key)
    _sage_bytes_equal(mean, expected_mean)
    expected_ki, expected_ks = torch.empty_like(ki), torch.empty_like(ks)
    quant._quant_k[(triton.cdiv(n, 128) * 4, h, 1)](
        key,
        expected_mean,
        expected_ki,
        expected_ks,
        *key.stride()[:3],
        expected_mean.stride(0),
        expected_mean.stride(2),
        h,
        n,
        ks.shape[-1],
        128,
    )
    _sage_bytes_equal(ki, expected_ki)
    _sage_bytes_equal(ks, expected_ks)
    nb, npad = triton.cdiv(n, 256), triton.cdiv(n, 128) * 128
    partial = torch.empty((1, h, nb, 128), device="cuda", dtype=torch.float32)
    expected_vs = torch.empty_like(vs)
    inverse = torch.empty_like(vs)
    expected_vf = torch.empty_like(vf)
    quant._v_partial[(nb, 8, h)](
        value, partial, *value.stride()[:3], h, n, nb, 128, 256, 16
    )
    quant._v_scales[(8, h)](
        partial, expected_vs, inverse, h, nb, 128, triton.next_power_of_2(nb), 16
    )
    safe_strided = value.data_ptr() % 16 != 0 or any(
        (stride * value.element_size()) % 16 for stride in value.stride()[:3]
    )
    quant._quant_v[(triton.cdiv(npad, 256), 8, h)](
        value,
        inverse,
        expected_vf,
        *value.stride()[:3],
        h,
        n,
        npad,
        128,
        256,
        16,
        SAFE_STRIDED=safe_strided,
    )
    _sage_bytes_equal(vs, expected_vs)
    _sage_bytes_equal(vf, expected_vf)
