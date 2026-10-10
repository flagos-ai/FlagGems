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

# The substituted baseline below is scoped to Kunlunxin, where the official vLLM
# baseline cannot be constructed (see NOTE). Every other backend keeps the
# original vLLM fused_experts_impl comparison.
IS_KUNLUNXIN = flag_gems.vendor_name == "kunlunxin"

try:
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        fused_experts_impl as vllm_fused_experts_impl,
    )

    HAS_VLLM_FUSED_MOE = True
except ImportError:
    HAS_VLLM_FUSED_MOE = False


class OutplaceFusedExpertsBenchmark(base.Benchmark):
    """
    Benchmark for outplace_fused_experts comparing FlagGems Triton kernel vs vLLM.
    """

    def __init__(self, op_name, torch_op, dtypes):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)

    def set_shapes(self, shape_file_path=None):
        # (num_tokens, num_experts, hidden_size, intermediate_size, topk)
        self.shapes = [
            # Mixtral-like shapes
            (1, 8, 4096, 14336, 2),
            (4, 8, 4096, 14336, 2),
            (16, 8, 4096, 14336, 2),
            (64, 8, 4096, 14336, 2),
            (128, 8, 4096, 14336, 2),
            (256, 8, 4096, 14336, 2),
            (512, 8, 4096, 14336, 2),
            # DeepSeek-V3-like shapes (TP=8 shard)
            (1, 256, 7168, 2048, 8),
            (4, 256, 7168, 2048, 8),
            (16, 256, 7168, 2048, 8),
            (64, 256, 7168, 2048, 8),
            (128, 256, 7168, 2048, 8),
            (256, 256, 7168, 2048, 8),
        ]

    def get_input_iter(self, cur_dtype):
        for config in self.shapes:
            yield from self._fused_moe_input_fn(config, cur_dtype)

    def _fused_moe_input_fn(self, config, dtype):
        num_tokens, num_experts, hidden_size, intermediate_size, topk = config
        device = flag_gems.device

        hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
        w1 = torch.randn(
            num_experts,
            intermediate_size * 2,
            hidden_size,
            device=device,
            dtype=dtype,
        )
        w2 = torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=dtype,
        )

        gating = torch.randn(
            num_tokens, num_experts, device=device, dtype=torch.float32
        )
        topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(dtype)

        yield (hidden_states, w1, w2, topk_weights, topk_ids)


# ---------------------------------------------------------------------------
# SUBSTITUTED DEVICE BASELINE -- NOT THE OFFICIAL vLLM BASELINE.
#
# The official contract for this marker is FlagGems vs vLLM's own
# fused_experts_impl. That baseline is NOT constructible on this Kunlunxin/XPU
# stack, measured 2026-10-08:
#   * vllm._C / vllm._moe_C cannot load -- ABI mismatch. vLLM needs
#     c10::cuda::c10_cuda_check_implementation(int, char const*, char const*,
#     unsigned int, bool) (mangled ...EiPKcS2_jb); this torch provides the
#     (..., int, bool) form (...EiPKcS2_ib). Preloading libc10_cuda.so does not
#     resolve it. Both .so are NVIDIA-only (sm_52/80/89/90/100/120,
#     NEEDED libcuda.so.1 + libcudart.so.12) and cannot execute on XPU anyway.
#   * Supplying the three missing native glue ops as device implementations
#     (moe_align_block_size fused_moe.py:1839, silu_and_mul activation.py:115,
#     moe_sum fused_moe.py:1919) does get vLLM's own Triton fused_experts_impl
#     to launch, but that kernel then fails on XPU both ways: at M=8 it compiles,
#     launches and faults with an illegal memory access (XPU status 700); at M=1
#     (naive_block_assignment, which bypasses the align helper) it fails to
#     compile -- PassManager::run failed ... [TritonSDNNLegalize], vLLM
#     fused_moe.py:315.
#
# Therefore the number reported here is measured against a SUBSTITUTED device
# baseline: a straightforward unfused per-expert index_select + matmul
# implementation. It is recorded so the comparison stays visible and reviewable.
# It is NOT an official-contract measurement and must not be counted as a pass
# without an explicit test-contract decision.
# ---------------------------------------------------------------------------
def _unfused_moe_experts_device_baseline(hidden_states, w1, w2, topk_weights, topk_ids):
    """Unfused per-expert MoE (silu gating), all device ops, no vLLM.

    Routing is computed once with a single stable argsort (one host sync per
    call); the expert loop then runs sync-free so E=256 shapes stay tractable.
    """
    num_experts = w1.shape[0]
    top_k = topk_ids.shape[1]
    out = torch.zeros_like(hidden_states)
    flat_expert = topk_ids.reshape(-1)
    counts = torch.bincount(flat_expert, minlength=num_experts)
    bounds = (torch.cumsum(counts, 0) - counts).tolist()
    ends = torch.cumsum(counts, 0).tolist()
    order = torch.argsort(flat_expert, stable=True)
    for ei in range(num_experts):
        lo, hi = bounds[ei], ends[ei]
        if hi <= lo:
            continue
        sel = order[lo:hi]
        tok = torch.div(sel, top_k, rounding_mode="floor")
        kth = sel - tok * top_k
        x = hidden_states.index_select(0, tok)
        a = x @ w1[ei].t()
        inter = a.shape[-1] // 2
        act = torch.nn.functional.silu(a[:, :inter]) * a[:, inter:]
        y = act @ w2[ei].t()
        w = topk_weights[tok, kth].to(hidden_states.dtype).unsqueeze(1)
        out.index_add_(0, tok, y * w)
    return out


def _vllm_outplace_fused_experts_wrapper(hidden_states, w1, w2, topk_weights, topk_ids):
    """Baseline for the comparison, per vendor.

    On Kunlunxin/XPU the official vLLM baseline is not constructible (see NOTE
    above), so a SUBSTITUTED unfused device baseline is used there. All other
    backends keep the original vLLM fused_experts_impl baseline unchanged.
    """
    if IS_KUNLUNXIN:
        return _unfused_moe_experts_device_baseline(
            hidden_states.clone(), w1, w2, topk_weights, topk_ids
        )
    return vllm_fused_experts_impl(
        hidden_states.clone(),
        w1,
        w2,
        topk_weights,
        topk_ids,
        inplace=False,
        activation="silu",
    )


@pytest.mark.outplace_fused_experts
@pytest.mark.skipif(not HAS_VLLM_FUSED_MOE, reason="vLLM not installed")
def test_outplace_fused_experts_gems_vs_vllm():
    """
    Benchmark FlagGems outplace_fused_experts vs vLLM outplace fused_experts_impl (bf16/fp16).
    """
    bench = OutplaceFusedExpertsBenchmark(
        op_name="outplace_fused_experts",
        torch_op=_vllm_outplace_fused_experts_wrapper,
        dtypes=[torch.bfloat16, torch.float16],
    )
    bench.set_gems(flag_gems.outplace_fused_experts)
    bench.run()
