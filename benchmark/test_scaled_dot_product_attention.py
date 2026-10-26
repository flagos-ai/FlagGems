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
from .test_scaled_dot_product_flash_attention import _ExternalH800Baseline


class AttentionBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        # self.shapes is a list of tuples, each containing three elements:
        # (batch, num_heads, seq_len, head_size).
        return []


class _ExternalFlashAttentionBenchmark(_ExternalH800Baseline, AttentionBenchmark):
    pass


def scaled_dot_product_flash_attention_input_fn(shape, dtype, device):
    query = torch.randn(shape, device=device, dtype=dtype)
    key = torch.randn(shape, device=device, dtype=dtype)
    value = torch.randn(shape, device=device, dtype=dtype)
    yield query, key, value, 0.0, False, False


def _flash_case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"query": list(dims), "key": list(dims), "value": list(dims)},
        params={"dropout_p": 0.0, "is_causal": False, "return_debug_mask": False},
        builder_args=(dims,),
    )


def _flash_build_inputs_fn(plan, dtype, device):
    return next(
        scaled_dot_product_flash_attention_input_fn(plan.builder_args[0], dtype, device)
    )


@pytest.mark.scaled_dot_product_flash_attention
@pytest.mark.underscore_scaled_dot_product_flash_attention
def test_scaled_dot_product_flash_attention():
    bench = _ExternalFlashAttentionBenchmark(
        op_name="_scaled_dot_product_flash_attention",
        case_fn=_flash_case_fn,
        build_inputs_fn=_flash_build_inputs_fn,
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
