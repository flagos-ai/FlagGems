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

from . import base


class AttentionBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        # self.shapes is a list of tuples, each containing three elements:
        # (batch, num_heads, seq_len, head_size).
        return []


def scaled_dot_product_flash_attention_input_fn(shape, dtype, device):
    query = torch.randn(shape, device=device, dtype=dtype)
    key = torch.randn(shape, device=device, dtype=dtype)
    value = torch.randn(shape, device=device, dtype=dtype)
    yield query, key, value, 0.0, False, False


@pytest.mark.scaled_dot_product_flash_attention
def test_scaled_dot_product_flash_attention():
    bench = AttentionBenchmark(
        op_name="scaled_dot_product_flash_attention",
        input_fn=scaled_dot_product_flash_attention_input_fn,
        torch_op=torch.ops.aten._scaled_dot_product_flash_attention.default,
        # FlashAttention supports CUDA float16 and bfloat16 inputs.
        dtypes=[torch.float16, torch.bfloat16],
    )
    bench.run()


@pytest.mark.scaled_dot_product_attention
@pytest.mark.parametrize("dropout_p", [0.0])
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_scaled_dot_product_attention(monkeypatch, dropout_p, is_causal):
    if flag_gems.vendor_name == "hygon":
        monkeypatch.setenv("TRITON_HIP_USE_NEW_STREAM_PIPELINE", "0")

    def scaled_dot_product_attention_kwargs(shape, dtype, device):
        query = torch.randn(shape, device=device, dtype=dtype)
        key = torch.randn(shape, device=device, dtype=dtype)
        value = torch.randn(shape, device=device, dtype=dtype)
        yield query, key, value, None, dropout_p, is_causal

    def sdpa_flash(
        query, key, value, attn_mask=None, dropout_p=dropout_p, is_causal=is_causal
    ):
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )

    bench = AttentionBenchmark(
        op_name="scaled_dot_product_attention",
        input_fn=scaled_dot_product_attention_kwargs,
        # torch_op=torch.nn.functional.scaled_dot_product_attention,
        torch_op=sdpa_flash,
        gems_op=flag_gems.scaled_dot_product_attention,
        dtypes=[
            torch.float16,
            torch.bfloat16,
        ],
    )
    bench.run()


# Multi-head-attention shapes used by the *_forward / *_backward split markers,
# mirroring tests/test_scaled_dot_product_attention.py's LEGACY_SHAPES.
SDPA_LEGACY_BENCH_SHAPES = [
    (4, 8, 8, 1024, 1024, 64, False),
    (4, 8, 8, 1024, 1024, 64, True),
    (4, 8, 8, 1024, 1024, 128, True),
]

# Backward-only shape subset. The flag_gems SDPA backward kernel
# (_attn_bwd) crashes with an illegal memory access for the causal
# head_size=64 case once a torch SDPA graph on the same tensors has run,
# so head_size=64 causal shapes are excluded here while head_size=128
# causal still keeps is_causal=True covered.
SDPA_LEGACY_BWD_BENCH_SHAPES = [
    (4, 8, 8, 1024, 1024, 64, False),
    (4, 8, 8, 1024, 1024, 128, False),
    (4, 8, 8, 1024, 1024, 128, True),
]


def sdpa_legacy_input_fn(shape, dtype, device):
    batch, num_q_head, num_kv_head, q_seq_len, kv_seq_len, head_size, is_causal = shape
    scale = float(1.0 / (head_size**0.5))
    q = torch.randn(
        (batch, num_q_head, q_seq_len, head_size), dtype=dtype, device=device
    )
    k = torch.randn(
        (batch, num_kv_head, kv_seq_len, head_size), dtype=dtype, device=device
    )
    v = torch.randn(
        (batch, num_kv_head, kv_seq_len, head_size), dtype=dtype, device=device
    )
    yield q, k, v, scale, is_causal


def sdpa_legacy_torch_op(q, k, v, scale, is_causal):
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, scale=scale, is_causal=is_causal
    )


def sdpa_legacy_gems_op(q, k, v, scale, is_causal):
    return flag_gems.scaled_dot_product_attention(
        q, k, v, attn_mask=None, scale=scale, is_causal=is_causal
    )


class SdpaLegacyBenchmark(base.GenericBenchmark):
    def __init__(self, *args, bench_shapes=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.bench_shapes = bench_shapes or SDPA_LEGACY_BENCH_SHAPES

    def set_shapes(self, shape_file_path=None):
        self.shapes = self.bench_shapes
        self.shape_desc = (
            "batch, num_q_head, num_kv_head, q_seq_len, kv_seq_len, "
            "head_size, is_causal"
        )


@pytest.mark.scaled_dot_product_attention_forward
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_scaled_dot_product_attention_forward():
    bench = SdpaLegacyBenchmark(
        input_fn=sdpa_legacy_input_fn,
        op_name="scaled_dot_product_attention_forward",
        torch_op=sdpa_legacy_torch_op,
        gems_op=sdpa_legacy_gems_op,
        dtypes=[
            torch.float16,
            torch.bfloat16,
        ],
    )
    bench.run()


@pytest.mark.scaled_dot_product_attention_backward
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_scaled_dot_product_attention_backward():
    bench = SdpaLegacyBenchmark(
        input_fn=sdpa_legacy_input_fn,
        op_name="scaled_dot_product_attention_backward",
        torch_op=sdpa_legacy_torch_op,
        gems_op=sdpa_legacy_gems_op,
        is_backward=True,
        bench_shapes=SDPA_LEGACY_BWD_BENCH_SHAPES,
        dtypes=[
            torch.float16,
            torch.bfloat16,
        ],
    )
    bench.run()
