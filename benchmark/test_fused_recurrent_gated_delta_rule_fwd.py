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
import torch.nn.functional as F

import flag_gems

from . import base

try:
    from vllm.model_executor.layers.fla.ops import (
        fused_recurrent_gated_delta_rule as base_fused_recurrent_gated_delta_rule,
    )

    HAS_VLLM_FLA = True
except ImportError:
    HAS_VLLM_FLA = False

B, H, HV, K, V = 1, 4, 8, 64, 64
SEQ_LENS = [128, 512, 2048]


def _build_inputs(T, dtype, device):
    key_dim, value_dim = H * K, HV * V
    mixed_qkv_dim = 2 * key_dim + value_dim
    mixed_qkv = torch.randn((B * T, mixed_qkv_dim), device=device, dtype=dtype)
    q, k, v = torch.split(mixed_qkv, [key_dim, key_dim, value_dim], dim=-1)
    query = q.view(B, T, H, K)
    key = k.view(B, T, H, K)
    value = v.view(B, T, HV, V)
    g = F.logsigmoid(torch.randn((B, T, HV), device=device, dtype=dtype))
    beta = torch.rand(B, T, HV, device=device, dtype=dtype).sigmoid()
    cu_seqlens = torch.arange(T + 1, device=device, dtype=torch.long)
    # The state buffer must cover every index in ssm_state_indices.
    ssm_state_len = max(128, T)
    initial_state = torch.zeros((ssm_state_len, HV, K, V), device=device, dtype=dtype)
    ssm_state_indices = torch.arange(T, device=device, dtype=torch.long)
    return (
        query,
        key,
        value,
        g,
        beta,
        K**-0.5,
        initial_state,
        cu_seqlens,
        ssm_state_indices,
    )


def _input_fn(case, dtype, device):
    (T,) = case
    (
        query,
        key,
        value,
        g,
        beta,
        scale,
        initial_state,
        cu_seqlens,
        ssm_state_indices,
    ) = _build_inputs(T, dtype, device)
    kwargs = {
        "q": query,
        "k": key,
        "v": value,
        "g": g,
        "beta": beta,
        "scale": scale,
        "initial_state": initial_state.clone(),
        "inplace_final_state": True,
        "cu_seqlens": cu_seqlens,
        "ssm_state_indices": ssm_state_indices,
        "num_accepted_tokens": None,
        "use_qk_l2norm_in_kernel": True,
    }
    yield (kwargs,)


def _vllm_baseline(**kwargs):
    out, _ = base_fused_recurrent_gated_delta_rule(
        q=kwargs["q"],
        k=kwargs["k"],
        v=kwargs["v"],
        g=kwargs["g"],
        beta=kwargs["beta"],
        scale=kwargs["scale"],
        initial_state=kwargs["initial_state"].clone(),
        inplace_final_state=kwargs["inplace_final_state"],
        cu_seqlens=kwargs["cu_seqlens"],
        ssm_state_indices=kwargs["ssm_state_indices"],
        use_qk_l2norm_in_kernel=kwargs["use_qk_l2norm_in_kernel"],
    )
    return out


class FusedRecurrentGatedDeltaRuleBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = [(T,) for T in SEQ_LENS]
    DEFAULT_SHAPE_DESC = "T"

    def init_default_config(self):
        self.shapes = self.DEFAULT_SHAPES

    def init_user_config(self):
        self.mode = base.Config.mode
        self.set_dtypes(base.Config.user_desired_dtypes)
        self.set_metrics(base.Config.user_desired_metrics)
        # Each case bundles T with the recurrent state geometry; shape-only
        # configuration files are not compatible with this benchmark.
        self.shapes = self.DEFAULT_SHAPES

    def get_input_iter(self, dtype):
        for case in self.shapes:
            yield from self.input_fn(case, dtype, self.device)


@pytest.mark.fused_recurrent_gated_delta_rule_fwd
@pytest.mark.skipif(
    not HAS_VLLM_FLA,
    reason="vLLM FLA fused_recurrent_gated_delta_rule is not available",
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_benchmark_fused_recurrent_gated_delta_rule_fwd():
    bench = FusedRecurrentGatedDeltaRuleBenchmark(
        op_name="fused_recurrent_gated_delta_rule_fwd",
        torch_op=_vllm_baseline,
        gems_op=flag_gems.fused_recurrent_gated_delta_rule_fwd,
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
    )
    bench.run()
