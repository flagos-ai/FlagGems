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
    import vllm._custom_ops as vllm_ops
    from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (
        fused_marlin_moe as vllm_fused_marlin_moe,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_permute_scales,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        mxfp4_marlin_process_scales,
    )
    from vllm.scalar_type import scalar_types

    VLLM_QUANT_TYPE_FP4 = scalar_types.float4_e2m1f
    HAS_VLLM_FUSED_MARLIN_MOE = True
except ImportError:
    HAS_VLLM_FUSED_MARLIN_MOE = False

import flag_gems

# FlagGems wrapper under test
from flag_gems.fused.fused_marlin_moe import QUANT_TYPE_FP4_E2M1
from flag_gems.fused.fused_marlin_moe import fused_marlin_moe as gems_fused_marlin_moe

from . import base


def is_cuda_available():
    if flag_gems.device != "cuda":
        return False
    major, minor = torch.cuda.get_device_capability()
    sm_version_num = major * 10 + minor
    return sm_version_num >= 90 and sm_version_num < 100


CUDA_AVAILABLE = is_cuda_available()

# =============================================================================
# MXFP4 (FP4 E2M1 + per-32 E8M0) benchmark: FlagGems Triton vs vLLM Marlin.
# Both consume the same FP4 weights + E8M0 scale in their respective layouts.
# =============================================================================
MXFP4_GROUP_SIZE = 32
_E2M1_POS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
_E2M1_MID = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
_E2M1_MAX = 6.0


def _quantize_mxfp4_2d(w_2d, group_size):
    """Round-to-nearest MXFP4. Returns nibbles (uint8 [0,15]) + E8M0 scale."""
    out_dim, in_dim = w_2d.shape
    ng = in_dim // group_size
    device = w_2d.device
    wg = w_2d.reshape(out_dim, ng, group_size).to(torch.float32)
    amax = wg.abs().amax(dim=-1, keepdim=True)
    exp = torch.ceil(torch.log2((amax / _E2M1_MAX).clamp(min=1e-30))).clamp(-127, 127)
    scale = torch.exp2(exp)
    e8m0_byte = (exp + 127.0).to(torch.uint8)
    wn = wg / scale
    sign = wn < 0
    a = wn.abs().clamp(max=_E2M1_MAX)
    mag = torch.bucketize(a, torch.tensor(_E2M1_MID, device=device))
    nibbles = (sign.to(torch.uint8) * 8 + mag.to(torch.uint8)).reshape(out_dim, in_dim)
    scale_e8m0 = e8m0_byte.squeeze(-1).view(torch.float8_e8m0fnu)
    return nibbles, scale_e8m0


def _mxfp4_quantize_per_expert(w_fp):
    """FlagGems MXFP4 layout: packed uint8 (E, out, in//2) + E8M0 scale."""
    E, out_dim, in_dim = w_fp.shape
    w_q = torch.empty(E, out_dim, in_dim // 2, device=w_fp.device, dtype=torch.uint8)
    scales = torch.empty(
        E,
        out_dim,
        in_dim // MXFP4_GROUP_SIZE,
        device=w_fp.device,
        dtype=torch.float8_e8m0fnu,
    )
    for e in range(E):
        nib, sc = _quantize_mxfp4_2d(w_fp[e], MXFP4_GROUP_SIZE)
        w_q[e] = nib[:, 1::2] * 16 + nib[:, ::2]
        scales[e] = sc
    return w_q, scales


def _marlin_mxfp4_quantize_per_expert(w_fp, dtype):
    """vLLM Marlin MXFP4 layout from the same nibbles/scale (numerically aligned)."""
    E, out_dim, in_dim = w_fp.shape
    qweight_l, scales_l = [], []
    for e in range(E):
        nib, sc = _quantize_mxfp4_2d(w_fp[e], MXFP4_GROUP_SIZE)
        packed = (nib[:, 1::2] * 16 + nib[:, ::2]).to(torch.uint8)
        perm = torch.empty(0, dtype=torch.int, device=w_fp.device)
        qw = vllm_ops.gptq_marlin_repack(
            packed.view(torch.int32).T.contiguous(), perm, in_dim, out_dim, 4, False
        )
        ms = marlin_permute_scales(
            sc.T.to(dtype), in_dim, out_dim, MXFP4_GROUP_SIZE, False
        )
        ms = mxfp4_marlin_process_scales(ms, input_dtype=None).to(torch.float8_e8m0fnu)
        qweight_l.append(qw)
        scales_l.append(ms)
    return torch.stack(qweight_l, 0).contiguous(), torch.stack(scales_l, 0).contiguous()


class FusedMarlinMoEW4A16MXFP4Benchmark(base.Benchmark):
    """MXFP4 (FP4 E2M1 + E8M0) MoE: FlagGems Triton vs vLLM Marlin."""

    def __init__(self, op_name, torch_op, dtypes):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            # Mixtral-8x7B
            (1, 8, 4096, 14336, 2),
            (16, 8, 4096, 14336, 2),
            (64, 8, 4096, 14336, 2),
            (256, 8, 4096, 14336, 2),
            # DeepSeek-V3 (TP=8 shard)
            (1, 256, 7168, 2048, 8),
            (16, 256, 7168, 2048, 8),
            (64, 256, 7168, 2048, 8),
            (256, 256, 7168, 2048, 8),
            # Qwen3-5-397B-A17B
            (1, 512, 4096, 1024, 10),
            (16, 512, 4096, 1024, 10),
            (64, 512, 4096, 1024, 10),
            (256, 512, 4096, 1024, 10),
            # DeepSeek-V4-Flash
            (1, 256, 4096, 2048, 6),
            (16, 256, 4096, 2048, 6),
            (64, 256, 4096, 2048, 6),
            (256, 256, 4096, 2048, 6),
        ]

    def get_input_iter(self, cur_dtype):
        for config in self.shapes:
            yield from self._gen(config, cur_dtype)

    def _gen(self, config, dtype):
        num_tokens, num_experts, hidden_size, intermediate_size, topk = config
        device = flag_gems.device

        hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
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
                num_experts, hidden_size, intermediate_size, device=device, dtype=dtype
            )
            / 10.0
        )

        w1_q_fg, w1_scale_fg = _mxfp4_quantize_per_expert(w1_fp)
        w2_q_fg, w2_scale_fg = _mxfp4_quantize_per_expert(w2_fp)
        w1_q_marlin, w1_scale_marlin = _marlin_mxfp4_quantize_per_expert(w1_fp, dtype)
        w2_q_marlin, w2_scale_marlin = _marlin_mxfp4_quantize_per_expert(w2_fp, dtype)

        del w1_fp, w2_fp
        torch.cuda.empty_cache()

        gating = torch.randn(
            num_tokens, num_experts, device=device, dtype=torch.float32
        )
        topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        yield (
            hidden_states,
            w1_q_fg,
            w2_q_fg,
            w1_scale_fg,
            w2_scale_fg,
            w1_q_marlin,
            w2_q_marlin,
            w1_scale_marlin,
            w2_scale_marlin,
            topk_weights,
            topk_ids,
        )


def _vllm_baseline_mxfp4(
    hidden_states,
    w1_q_fg,
    w2_q_fg,
    w1_scale_fg,
    w2_scale_fg,
    w1_q_marlin,
    w2_q_marlin,
    w1_scale_marlin,
    w2_scale_marlin,
    topk_weights,
    topk_ids,
):
    """Baseline: vLLM's CUDA Marlin fused_marlin_moe (MXFP4)."""
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
        quant_type_id=VLLM_QUANT_TYPE_FP4.id,
    )


def _gems_call_mxfp4(
    hidden_states,
    w1_q_fg,
    w2_q_fg,
    w1_scale_fg,
    w2_scale_fg,
    w1_q_marlin,
    w2_q_marlin,
    w1_scale_marlin,
    w2_scale_marlin,
    topk_weights,
    topk_ids,
):
    """FlagGems' Triton MXFP4 fused_marlin_moe."""
    return gems_fused_marlin_moe(
        hidden_states=hidden_states,
        w1=w1_q_fg,
        w2=w2_q_fg,
        bias1=None,
        bias2=None,
        w1_scale=w1_scale_fg,
        w2_scale=w2_scale_fg,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        quant_type_id=QUANT_TYPE_FP4_E2M1,
        group_size=MXFP4_GROUP_SIZE,
    )


@pytest.mark.fused_marlin_moe
@pytest.mark.skipif(
    not HAS_VLLM_FUSED_MARLIN_MOE, reason="vllm not installed; baseline unavailable"
)
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires NVIDIA Hopper architecture")
def test_fused_marlin_moe_w4a16_mxfp4():
    """
    Benchmark FlagGems MXFP4 fused_marlin_moe (Triton) vs vLLM MXFP4
    fused_marlin_moe (CUDA Marlin). Both run FP4 E2M1 + per-32 E8M0 W4A16 GEMM.
    """
    bench = FusedMarlinMoEW4A16MXFP4Benchmark(
        op_name="fused_marlin_moe_w4a16_mxfp4",
        torch_op=_vllm_baseline_mxfp4,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_gems_call_mxfp4)
    bench.run()
