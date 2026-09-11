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

# vLLM imports (baseline). Optional: when vllm is not installed (e.g. in CI),
# the entire benchmark is skipped via the skipif marker below.
try:
    from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (
        fused_marlin_moe as vllm_fused_marlin_moe,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
        marlin_quantize,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        quantize_weights,
    )
    from vllm.scalar_type import scalar_types

    VLLM_QUANT_TYPE = scalar_types.uint4b8
    HAS_VLLM_FUSED_MARLIN_MOE = True
except ImportError:
    HAS_VLLM_FUSED_MARLIN_MOE = False

import flag_gems

# FlagGems wrapper under test
from flag_gems.fused.fused_marlin_moe import QUANT_TYPE_UINT4B8
from flag_gems.fused.fused_marlin_moe import fused_marlin_moe as gems_fused_marlin_moe

from . import base


def is_cuda_available():
    if flag_gems.device != "cuda":
        return False
    major, minor = torch.cuda.get_device_capability()
    sm_version_num = major * 10 + minor
    return sm_version_num >= 90 and sm_version_num < 100


CUDA_AVAILABLE = is_cuda_available()

GROUP_SIZE = 128


def _wna16_quantize_per_expert(w_fp):
    """
    Per-expert GPTQ-style INT4 quantization for FlagGems wna16 kernel layout.

    Input  w_fp: (E, out_dim, in_dim), bf16/fp16
    Output w_q:   (E, out_dim, in_dim // 2), uint8 (two nibbles per byte)
           scales: (E, out_dim, in_dim // GROUP_SIZE), same dtype as w_fp
    """
    E, out_dim, in_dim = w_fp.shape
    assert in_dim % GROUP_SIZE == 0
    w_q = torch.empty(E, out_dim, in_dim // 2, device=w_fp.device, dtype=torch.uint8)
    scales = torch.empty(
        E, out_dim, in_dim // GROUP_SIZE, device=w_fp.device, dtype=w_fp.dtype
    )
    for e in range(E):
        _, q_e, sc_e, _ = quantize_weights(
            w_fp[e].T, VLLM_QUANT_TYPE, GROUP_SIZE, False, False
        )
        q_e = q_e.T.contiguous().to(torch.uint8)
        sc_e = sc_e.T
        w_q[e] = q_e[:, 1::2] * 16 + q_e[:, ::2]
        scales[e] = sc_e
    return w_q, scales


def _marlin_quantize_per_expert(w_fp):
    """
    Per-expert Marlin-layout INT4 quantization for vLLM's fused_marlin_moe.

    Input  w_fp: (E, out_dim, in_dim), bf16/fp16
    Output qweight: stacked (E, ...), int32 (Marlin packed layout)
           scales:  stacked (E, ...), same dtype as w_fp
    """
    qweight_l, scales_l = [], []
    E = w_fp.shape[0]
    for e in range(E):
        # marlin_quantize expects (in_dim, out_dim)
        _, qw, sc, _, _, _ = marlin_quantize(
            w_fp[e].T.contiguous(), VLLM_QUANT_TYPE, GROUP_SIZE, act_order=False
        )
        qweight_l.append(qw)
        scales_l.append(sc)
    qweight = torch.stack(qweight_l, dim=0).contiguous()
    scales = torch.stack(scales_l, dim=0).contiguous()
    return qweight, scales


class FusedMarlinMoEW4A16INT4Benchmark(base.Benchmark):
    """
    Benchmark for fused_marlin_moe W4A16 INT4 (fused-dequant MoE GEMM).

    Compares FlagGems' Triton wna16 kernel against vLLM's Marlin CUDA kernel.
    Both consume per-group-128 GPTQ uint4b8 weights (different packed layouts).
    """

    def __init__(self, op_name, torch_op, dtypes):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)

    def set_shapes(self, shape_file_path=None):
        # The three production MoE architectures from profile_fused_marlin_moe.py
        # over the decode token range (1 .. 256).
        self.shapes = [
            # Mixtral-8x7B
            (1, 8, 4096, 14336, 2),
            (4, 8, 4096, 14336, 2),
            (8, 8, 4096, 14336, 2),
            (16, 8, 4096, 14336, 2),
            (32, 8, 4096, 14336, 2),
            (64, 8, 4096, 14336, 2),
            (128, 8, 4096, 14336, 2),
            (256, 8, 4096, 14336, 2),
            # DeepSeek-V3 (TP=8 shard)
            (1, 256, 7168, 2048, 8),
            (4, 256, 7168, 2048, 8),
            (8, 256, 7168, 2048, 8),
            (16, 256, 7168, 2048, 8),
            (32, 256, 7168, 2048, 8),
            (64, 256, 7168, 2048, 8),
            (128, 256, 7168, 2048, 8),
            (256, 256, 7168, 2048, 8),
            # Qwen3-5-397B-A17B
            (1, 512, 4096, 1024, 10),
            (4, 512, 4096, 1024, 10),
            (8, 512, 4096, 1024, 10),
            (16, 512, 4096, 1024, 10),
            (32, 512, 4096, 1024, 10),
            (64, 512, 4096, 1024, 10),
            (128, 512, 4096, 1024, 10),
            (256, 512, 4096, 1024, 10),
            # DeepSeek-V4-Flash
            (1, 256, 4096, 2048, 6),
            (4, 256, 4096, 2048, 6),
            (8, 256, 4096, 2048, 6),
            (16, 256, 4096, 2048, 6),
            (32, 256, 4096, 2048, 6),
            (64, 256, 4096, 2048, 6),
            (128, 256, 4096, 2048, 6),
            (256, 256, 4096, 2048, 6),
        ]

    def get_input_iter(self, cur_dtype):
        for config in self.shapes:
            yield from self._gen(config, cur_dtype)

    def _gen(self, config, dtype):
        num_tokens, num_experts, hidden_size, intermediate_size, topk = config
        device = flag_gems.device

        hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)

        # Original FP weights (kept only as source for both quantizers).
        w1_fp = (
            torch.randn(
                num_experts,
                intermediate_size * 2,
                hidden_size,
                device=device,
                dtype=dtype,
            )
            / 10.0
        )
        w2_fp = (
            torch.randn(
                num_experts,
                hidden_size,
                intermediate_size,
                device=device,
                dtype=dtype,
            )
            / 10.0
        )

        # FlagGems wna16 layout
        w1_q_wna16, w1_scale_wna16 = _wna16_quantize_per_expert(w1_fp)
        w2_q_wna16, w2_scale_wna16 = _wna16_quantize_per_expert(w2_fp)

        # vLLM Marlin layout
        w1_q_marlin, w1_scale_marlin = _marlin_quantize_per_expert(w1_fp)
        w2_q_marlin, w2_scale_marlin = _marlin_quantize_per_expert(w2_fp)

        del w1_fp, w2_fp
        torch.cuda.empty_cache()

        # Routing
        gating = torch.randn(
            num_tokens, num_experts, device=device, dtype=torch.float32
        )
        topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        # vLLM requires fp32 topk_weights; FlagGems wrapper is dtype-agnostic.

        # Both ops get the same tuple; each picks what it needs.
        yield (
            hidden_states,
            w1_q_wna16,
            w2_q_wna16,
            w1_scale_wna16,
            w2_scale_wna16,
            w1_q_marlin,
            w2_q_marlin,
            w1_scale_marlin,
            w2_scale_marlin,
            topk_weights,
            topk_ids,
        )


def _vllm_baseline(
    hidden_states,
    w1_q_wna16,
    w2_q_wna16,
    w1_scale_wna16,
    w2_scale_wna16,
    w1_q_marlin,
    w2_q_marlin,
    w1_scale_marlin,
    w2_scale_marlin,
    topk_weights,
    topk_ids,
):
    """Baseline: vLLM's CUDA Marlin fused_marlin_moe."""
    return vllm_fused_marlin_moe(
        hidden_states=hidden_states,
        w1=w1_q_marlin,
        w2=w2_q_marlin,
        bias1=None,
        bias2=None,
        w1_scale=w1_scale_marlin,
        w2_scale=w2_scale_marlin,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        quant_type_id=VLLM_QUANT_TYPE.id,
    )


def _gems_call(
    hidden_states,
    w1_q_wna16,
    w2_q_wna16,
    w1_scale_wna16,
    w2_scale_wna16,
    w1_q_marlin,
    w2_q_marlin,
    w1_scale_marlin,
    w2_scale_marlin,
    topk_weights,
    topk_ids,
):
    """FlagGems' Triton wna16 fused_marlin_moe (Phase 2)."""
    return gems_fused_marlin_moe(
        hidden_states=hidden_states,
        w1=w1_q_wna16,
        w2=w2_q_wna16,
        bias1=None,
        bias2=None,
        w1_scale=w1_scale_wna16,
        w2_scale=w2_scale_wna16,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        quant_type_id=QUANT_TYPE_UINT4B8,
    )


@pytest.mark.fused_marlin_moe
@pytest.mark.skipif(
    not HAS_VLLM_FUSED_MARLIN_MOE, reason="vllm not installed; baseline unavailable"
)
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires NVIDIA Hopper architecture")
def test_fused_marlin_moe_w4a16_int4():
    """
    Benchmark FlagGems fused_marlin_moe (Triton wna16) vs vLLM fused_marlin_moe
    (CUDA Marlin). Both run GPTQ uint4b8 + per-group-128 W4A16 GEMM.
    """
    bench = FusedMarlinMoEW4A16INT4Benchmark(
        op_name="fused_marlin_moe_w4a16_int4",
        torch_op=_vllm_baseline,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_gems_call)
    bench.run()
