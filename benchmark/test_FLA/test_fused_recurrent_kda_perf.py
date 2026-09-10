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
import triton

import flag_gems
from benchmark.base import Benchmark
from flag_gems.fused.FLA import fused_recurrent


def _vllm_024_kda_baseline(
    q,
    k,
    v,
    g,
    beta,
    scale,
    baseline_state,
    candidate_state,
    cu_seqlens,
    ssm_state_indices,
    legacy_out,
    new_out,
):
    """Launch the vLLM 0.24 recurrent KDA Triton configuration directly."""
    del candidate_state, new_out

    B, T, H, K = q.shape
    HV, V = v.shape[2:]
    N = cu_seqlens.numel() - 1
    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 8)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1

    grid = (NK, NV, N * HV)
    fused_recurrent.fused_recurrent_gated_delta_rule_large_t_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        o=legacy_out,
        h0=baseline_state,
        ht=baseline_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=None,
        scale=scale,
        N=N,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        stride_init_state_token=baseline_state.stride(0),
        stride_final_state_token=baseline_state.stride(0),
        stride_indices_seq=ssm_state_indices.stride(0),
        stride_indices_tok=1,
        IS_BETA_HEADWISE=beta.ndim == v.ndim,
        USE_QK_L2NORM_IN_KERNEL=True,
        INPLACE_FINAL_STATE=True,
        IS_KDA=True,
        num_warps=1,
        num_stages=3,
    )
    return legacy_out, baseline_state


def _flag_gems_kda_candidate(
    q,
    k,
    v,
    g,
    beta,
    scale,
    baseline_state,
    candidate_state,
    cu_seqlens,
    ssm_state_indices,
    legacy_out,
    new_out,
):
    del baseline_state, legacy_out
    return flag_gems.fused_recurrent_kda_decode(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=candidate_state,
        ssm_state_indices=ssm_state_indices,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
        out=new_out,
    )


class FusedRecurrentKDABenchmark(Benchmark):
    DEFAULT_DTYPES = [torch.bfloat16]
    DEFAULT_SHAPES = [(1,), (32,), (64,), (96,), (128,)]
    DEFAULT_SHAPE_DESC = "N"

    def set_shapes(self, shape_file_path=None):
        self.shapes = self.DEFAULT_SHAPES
        self.shape_desc = self.DEFAULT_SHAPE_DESC

    def get_input_iter(self, cur_dtype):
        device = flag_gems.device
        B, H, HV, K, V = 1, 4, 4, 128, 128
        scale = K**-0.5

        for (N,) in self.shapes:
            torch.manual_seed(2026 + N)
            q = torch.randn(B, N, H, K, dtype=cur_dtype, device=device)
            k = torch.randn_like(q)
            v = torch.randn(B, N, HV, V, dtype=cur_dtype, device=device)

            # These are the already-processed natural-log decay and sigmoid beta
            # tensors consumed by fused_recurrent_kda_fwd in serving decode.
            g = -5.0 * torch.sigmoid(
                torch.randn(B, N, HV, K, dtype=torch.float32, device=device)
            )
            beta = torch.sigmoid(
                torch.randn(B, N, HV, dtype=torch.float32, device=device)
            )

            state = 0.01 * torch.randn(
                N + 1, HV, V, K, dtype=torch.float32, device=device
            )
            baseline_state = state.clone()
            candidate_state = state.clone()
            cu_seqlens = torch.arange(N + 1, dtype=torch.int32, device=device)
            ssm_state_indices = torch.arange(1, N + 1, dtype=torch.int32, device=device)
            legacy_out = torch.empty_like(v)
            new_out = torch.empty_like(v)

            yield (
                q,
                k,
                v,
                g,
                beta,
                scale,
                baseline_state,
                candidate_state,
                cu_seqlens,
                ssm_state_indices,
                legacy_out,
                new_out,
            )


@pytest.mark.fused_recurrent_kda
def test_perf_fused_recurrent_kda():
    bench = FusedRecurrentKDABenchmark(
        op_name="fused_recurrent_kda",
        torch_op=_vllm_024_kda_baseline,
    )
    bench.set_gems(_flag_gems_kda_candidate)
    bench.run()
