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

import math

import pytest
import torch

import flag_gems
from benchmark.base import Benchmark
from benchmark.conftest import Config

try:
    from vllm.model_executor.layers.fla.ops import (
        fused_recurrent_gated_delta_rule as base_fused_recurrent_gated_delta_rule,
    )

    VLLM_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency guard
    base_fused_recurrent_gated_delta_rule = None
    VLLM_AVAILABLE = False


def rearrange_mixed_qkv(
    mixed_qkv, key_dim, value_dim, head_k_dim, head_v_dim, tp_size=1, contiguous=True
):
    query, key, value = torch.split(
        mixed_qkv,
        [
            key_dim // tp_size,
            key_dim // tp_size,
            value_dim // tp_size,
        ],
        dim=-1,
    )
    query = query.view(1, query.shape[0], -1, head_k_dim)
    key = key.view(1, key.shape[0], -1, head_k_dim)
    value = value.view(1, value.shape[0], -1, head_v_dim)
    if contiguous:
        return query.contiguous(), key.contiguous(), value.contiguous()
    else:
        return query, key, value


class FusedRecurrentGatedDeltaRuleBenchmark(Benchmark):
    DEFAULT_DTYPES = [torch.bfloat16]

    def __init__(self, qkv_contiguous: bool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qkv_contiguous = qkv_contiguous

    def set_more_shapes(self):
        # Test the full set of sequence lengths we saw from the runtime prints
        return [
            (1,),
            (2,),
            (4,),
            (8,),
            (16,),
            (24,),
            (32,),
            (40,),
            (48,),
            (56,),
            (72,),
            (80,),
            (88,),
            (96,),
            (104,),
            (112,),
            (120,),
            (128,),
            (136,),
            (144,),
            (152,),
            (160,),
            (168,),
            (176,),
            (192,),
            (200,),
            (208,),
            (216,),
            (224,),
            (232,),
            (240,),
            (248,),
            (272,),
            (288,),
            (304,),
            (320,),
            (336,),
            (352,),
            (368,),
            (384,),
            (400,),
            (416,),
            (432,),
            (448,),
            (464,),
            (480,),
            (496,),
        ]

    def get_input_iter(self, cur_dtype):
        for (T,) in self.shapes:
            yield self._build_inputs(T, cur_dtype)

    def _build_inputs(self, T: int, dtype: torch.dtype):
        device = flag_gems.device
        B = 1
        H, HV, K, V = 16, 32, 128, 128
        tp_size = 4
        key_dim = H * K
        value_dim = HV * V

        assert key_dim % tp_size == 0 and value_dim % tp_size == 0

        mixed_qkv_dim = (2 * key_dim + value_dim) // tp_size
        total_tokens = B * T
        mixed_qkv = torch.randn(
            (total_tokens, mixed_qkv_dim), device=device, dtype=dtype
        )

        q, k, v = rearrange_mixed_qkv(
            mixed_qkv,
            key_dim=key_dim,
            value_dim=value_dim,
            head_k_dim=K,
            head_v_dim=V,
            tp_size=tp_size,
            contiguous=self.qkv_contiguous,
        )

        HV_local = v.shape[2]
        g = torch.nn.functional.logsigmoid(
            torch.randn((B, T, HV_local), device=device, dtype=dtype)
        )
        beta = torch.rand(B, T, HV_local, device=device, dtype=dtype).sigmoid()
        cu_seqlens = torch.arange(T + 1, device=device, dtype=torch.long)
        initial_state = torch.zeros((1024, HV_local, K, V), device=device, dtype=dtype)
        ssm_state_indices = torch.zeros(T, device=device, dtype=torch.long)
        scale = 0.08838834764831845

        # positional args follow fused_recurrent_gated_delta_rule_fwd signature
        return (
            q,
            k,
            v,
            g,
            beta,
            scale,
            initial_state,
            True,
            cu_seqlens,
            ssm_state_indices,
            None,
            True,
        )


def _torch_op_wrapper(*args, **kwargs):
    if VLLM_AVAILABLE:
        return base_fused_recurrent_gated_delta_rule(*args, **kwargs)
    return flag_gems.fused_recurrent_gated_delta_rule_fwd(*args, **kwargs)


@pytest.mark.fused_recurrent_gated_delta_rule_fwd
@pytest.mark.fused_recurrent_gated_delta_rule
@pytest.mark.parametrize("qkv_contiguous", [False])
def test_perf_fused_recurrent_gated_delta_rule(qkv_contiguous):
    bench = FusedRecurrentGatedDeltaRuleBenchmark(
        qkv_contiguous,
        op_name="fused_recurrent_gated_delta_rule",
        torch_op=_torch_op_wrapper,
    )
    bench.set_gems(flag_gems.fused_recurrent_gated_delta_rule_fwd)
    bench.run()


class FusedRecurrentGatedDeltaRuleFP8Benchmark(Benchmark):
    DEFAULT_DTYPES = [torch.bfloat16]
    DEFAULT_SHAPES = [
        (1,),
        (2,),
        (4,),
        (8,),
        (16,),
        (32,),
        (64,),
        (128,),
        (256,),
        (512,),
    ]
    DEFAULT_SHAPE_DESC = "num_sequences"

    def init_user_config(self):
        self.mode = Config.mode
        self.set_dtypes(Config.user_desired_dtypes)
        self.set_metrics(Config.user_desired_metrics)
        self.shapes = self.DEFAULT_SHAPES
        self.shape_desc = self.DEFAULT_SHAPE_DESC

    def get_input_iter(self, cur_dtype):
        for (num_sequences,) in self.shapes:
            yield self._build_inputs(num_sequences, cur_dtype)

    @staticmethod
    def _build_inputs(num_sequences: int, dtype: torch.dtype):
        H, HV, K, V = 4, 8, 128, 128
        mixed = torch.randn(
            num_sequences,
            2 * H * K + HV * V,
            device=flag_gems.device,
            dtype=dtype,
        )
        q, k, v = torch.split(mixed, (H * K, H * K, HV * V), dim=-1)
        q = q.view(1, num_sequences, H, K)
        k = k.view(1, num_sequences, H, K)
        v = (0.125 * v).view(1, num_sequences, HV, V)
        g = torch.empty(
            1,
            num_sequences,
            HV,
            device=flag_gems.device,
            dtype=dtype,
        ).uniform_(math.log(0.98), math.log(0.995))
        beta = torch.rand(
            1,
            num_sequences,
            HV,
            device=flag_gems.device,
            dtype=dtype,
        )
        state_bf16 = torch.zeros(
            num_sequences,
            HV,
            K,
            V,
            device=flag_gems.device,
            dtype=dtype,
        )
        state_fp8 = flag_gems.quantize_gdn_state_fp8(state_bf16)
        cu_seqlens = torch.arange(
            num_sequences + 1, device=flag_gems.device, dtype=torch.long
        )
        state_indices = torch.arange(
            num_sequences, device=flag_gems.device, dtype=torch.long
        )
        return (
            q,
            k,
            v,
            g,
            beta,
            state_bf16,
            state_fp8,
            K**-0.5,
            cu_seqlens,
            state_indices,
        )


def _bf16_decode_wrapper(
    q,
    k,
    v,
    g,
    beta,
    state_bf16,
    _state_fp8,
    scale,
    cu_seqlens,
    state_indices,
):
    return flag_gems.fused_recurrent_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=state_bf16,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=state_indices,
        num_accepted_tokens=None,
        use_qk_l2norm_in_kernel=True,
    )


def _fp8_w8a16_decode_wrapper(
    q,
    k,
    v,
    g,
    beta,
    _state_bf16,
    state_fp8,
    scale,
    cu_seqlens,
    state_indices,
):
    return flag_gems.fused_recurrent_gated_delta_rule_fp8_w8a16_decode(
        q,
        k,
        v,
        g,
        beta,
        state_fp8,
        scale,
        cu_seqlens,
        state_indices,
        True,
    )


def _fp8_decode_available():
    return (
        flag_gems.device == "cuda"
        and torch.cuda.is_available()
        and hasattr(torch, "float8_e4m3fn")
        and torch.cuda.get_device_capability()[0] >= 9
    )


@pytest.mark.skipif(
    not _fp8_decode_available(), reason="FP8 GDN decode requires SM90 or newer"
)
@pytest.mark.fused_recurrent_gated_delta_rule
def test_perf_fused_recurrent_gated_delta_rule_fp8_w8a16():
    torch.manual_seed(0)
    bench = FusedRecurrentGatedDeltaRuleFP8Benchmark(
        op_name="fused_recurrent_gated_delta_rule",
        torch_op=_bf16_decode_wrapper,
    )
    bench.set_gems(_fp8_w8a16_decode_wrapper)
    bench.run()
