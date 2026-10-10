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
from torch.nn.attention import SDPBackend, sdpa_kernel

import flag_gems

pytestmark = pytest.mark.attention


def _attention(api, q, k, v, bias):
    if api == "sdpa":
        return flag_gems.scaled_dot_product_attention(q, k, v, attn_mask=bias)
    if api == "scaled_cudnn":
        return flag_gems._scaled_dot_product_cudnn_attention(
            q, k, v, bias, False, 0.0, False, False
        )[0]
    return flag_gems.cudnn_attention_forward(
        q, k, v, bias, None, None, q.shape[2], k.shape[2], False
    )[0]


def _reference(q, k, v, bias):
    with sdpa_kernel(SDPBackend.MATH):
        return torch.nn.functional.scaled_dot_product_attention(
            q.float(), k.float(), v.float(), attn_mask=bias.float()
        )


def _inputs(dtype, mask_kind, strided):
    torch.manual_seed(42)
    b, h, sq, sk, d = 2, 4, 65, 97, 72

    def make_qkv(seq):
        # Model vision projections are sliced and transposed, not BHSD-contiguous.
        x = torch.randn(b, seq, h, 3 * d, device=flag_gems.device, dtype=dtype)
        x = x[..., :d].transpose(1, 2)
        return x if strided else x.contiguous()

    q, k, v = make_qkv(sq), make_qkv(sk), make_qkv(sk)
    shapes = {
        "padding": (b, 1, 1, sk),
        "batch": (1, h, sq, sk),
        "matrix": (sq, sk),
        "expanded": (b, 1, 1, sk),
    }
    # A sliced mask also checks that the non-broadcast strides are preserved.
    shape = shapes[mask_kind]
    bias = torch.zeros(*shape[:-1], 2 * sk, device=flag_gems.device, dtype=dtype)
    bias = bias[..., ::2]
    bias[..., sk // 2 :] = torch.finfo(dtype).min
    if mask_kind == "expanded":
        bias = bias.expand(b, h, sq, sk)
    return q, k, v, bias


@pytest.mark.scaled_dot_product_attention
@pytest.mark.scaled_dot_product_cudnn_attention
@pytest.mark.cudnn_attention_forward
@pytest.mark.skipif(flag_gems.device != "cuda", reason="cuDNN attention requires CUDA")
@pytest.mark.parametrize("api", ["sdpa", "scaled_cudnn", "cudnn"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("mask_kind", ["padding", "batch", "matrix", "expanded"])
def test_attention_mask_broadcast(api, dtype, strided, mask_kind):
    q, k, v, bias = _inputs(dtype, mask_kind, strided)
    expected = _reference(q, k, v, bias)
    actual = _attention(api, q, k, v, bias)
    torch.testing.assert_close(actual.float(), expected, atol=0.02, rtol=0.02)


@pytest.mark.scaled_dot_product_attention
@pytest.mark.scaled_dot_product_cudnn_attention
@pytest.mark.cudnn_attention_forward
@pytest.mark.skipif(flag_gems.device != "cuda", reason="cuDNN attention requires CUDA")
@pytest.mark.parametrize("api", ["sdpa", "scaled_cudnn", "cudnn"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_attention_mask_broadcast_graph(api, dtype):
    q, k, v, bias = _inputs(dtype, "padding", True)
    _attention(api, q, k, v, bias)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = _attention(api, q, k, v, bias)

    for seed in (43, 44):
        torch.manual_seed(seed)
        q.normal_()
        k.normal_()
        v.normal_()
        # Change mask data as well as Q/K/V between replays.
        bias.zero_()
        bias[..., (seed % 3 + 1) * 20 :] = torch.finfo(dtype).min
        graph.replay()
        torch.cuda.synchronize()
        expected = _reference(q, k, v, bias)
        torch.testing.assert_close(actual.float(), expected, atol=0.02, rtol=0.02)


@pytest.mark.scaled_dot_product_attention
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_attention_mask_broadcast_gqa(dtype):
    q, k, v, bias = _inputs(dtype, "padding", True)
    k, v = k[:, :2], v[:, :2]
    with sdpa_kernel(SDPBackend.MATH):
        expected = torch.nn.functional.scaled_dot_product_attention(
            q.float(), k.float(), v.float(), attn_mask=bias.float(), enable_gqa=True
        )
    actual = flag_gems.scaled_dot_product_attention(
        q, k, v, attn_mask=bias, enable_gqa=True
    )
    torch.testing.assert_close(actual.float(), expected, atol=0.02, rtol=0.02)
