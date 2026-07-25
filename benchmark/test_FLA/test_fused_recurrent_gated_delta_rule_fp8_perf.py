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
import triton

import flag_gems
from benchmark.conftest import Config

H = 4
HV = 8
K = 128
V = 128
BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


def _fp8_decode_available() -> bool:
    return (
        flag_gems.device == "cuda"
        and torch.cuda.is_available()
        and hasattr(torch, "float8_e4m3fn")
        and torch.cuda.get_device_capability()[0] >= 9
    )


def _make_inputs(num_sequences: int):
    mixed = torch.randn(
        num_sequences,
        2 * H * K + HV * V,
        device=flag_gems.device,
        dtype=torch.bfloat16,
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
        dtype=torch.bfloat16,
    ).uniform_(math.log(0.98), math.log(0.995))
    beta = torch.rand(
        1,
        num_sequences,
        HV,
        device=flag_gems.device,
        dtype=torch.bfloat16,
    )
    state = torch.zeros(
        num_sequences,
        HV,
        K,
        V,
        device=flag_gems.device,
        dtype=torch.bfloat16,
    )
    cu_seqlens = torch.arange(
        num_sequences + 1, device=flag_gems.device, dtype=torch.long
    )
    state_indices = torch.arange(
        num_sequences, device=flag_gems.device, dtype=torch.long
    )
    return q, k, v, g, beta, state, cu_seqlens, state_indices


def _bench_cuda_graph(fn) -> float:
    return triton.testing.do_bench_cudagraph(
        fn,
        rep=Config.repetition,
        return_mode="median",
    )


@pytest.mark.skipif(
    not _fp8_decode_available(), reason="FP8 GDN decode requires SM90 or newer"
)
@pytest.mark.fused_recurrent_gated_delta_rule
def test_perf_fused_recurrent_gated_delta_rule_fp8_w8a16_cuda_graph():
    torch.manual_seed(0)
    rows = []
    for num_sequences in BATCH_SIZES:
        q, k, v, g, beta, state, cu_seqlens, state_indices = _make_inputs(num_sequences)
        state_bf16 = state.clone()
        state_fp8 = flag_gems.quantize_gdn_state_fp8(state)

        def bf16_fn():
            return flag_gems.fused_recurrent_gated_delta_rule_fwd(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                scale=K**-0.5,
                initial_state=state_bf16,
                inplace_final_state=True,
                cu_seqlens=cu_seqlens,
                ssm_state_indices=state_indices,
                num_accepted_tokens=None,
                use_qk_l2norm_in_kernel=True,
            )

        def fp8_fn():
            return flag_gems.fused_recurrent_gated_delta_rule_fp8_w8a16_decode(
                q,
                k,
                v,
                g,
                beta,
                state_fp8,
                K**-0.5,
                cu_seqlens,
                state_indices,
                True,
            )

        bf16_ms = _bench_cuda_graph(bf16_fn)
        fp8_ms = _bench_cuda_graph(fp8_fn)
        rows.append((num_sequences, bf16_ms, fp8_ms, bf16_ms / fp8_ms))

    print("\nbatch\tBF16 GDN ms\tFP8-W8A16 GDN ms\tspeedup")
    for num_sequences, bf16_ms, fp8_ms, speedup in rows:
        print(f"{num_sequences}\t{bf16_ms:.6f}\t{fp8_ms:.6f}\t{speedup:.3f}")
    average_speedup = sum(row[3] for row in rows) / len(rows)
    print(f"average\t-\t-\t{average_speedup:.3f}")
