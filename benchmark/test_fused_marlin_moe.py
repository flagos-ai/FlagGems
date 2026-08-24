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

from unittest import mock

import pytest
import torch

try:
    import vllm._custom_ops as vllm_ops
    from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (
        fused_marlin_moe as vllm_fused_marlin_moe,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_permute_scales,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
        fp8_fused_exponent_bias_into_scales,
        pack_fp8_to_int32,
    )
    from vllm.scalar_type import scalar_types

    VLLM_QUANT_TYPE_FP8 = scalar_types.float8_e4m3fn
    HAS_VLLM_FUSED_MARLIN_MOE = True
except ImportError:
    HAS_VLLM_FUSED_MARLIN_MOE = False

import flag_gems
from flag_gems.fused.fused_marlin_moe import QUANT_TYPE_FLOAT8_E4M3FN
from flag_gems.fused.fused_marlin_moe import fused_marlin_moe as gems_fused_marlin_moe

from . import base, consts
from .conftest import Config


def is_cuda_available():
    if flag_gems.device != "cuda":
        return False
    major, minor = torch.cuda.get_device_capability()
    sm_version_num = major * 10 + minor
    return 90 <= sm_version_num < 100


CUDA_AVAILABLE = is_cuda_available()
GROUP_SIZE = 128


def _quantize_per_expert_fp8(w_fp):
    """Quantize each expert to E4M3 with one scale per 128 weights."""
    num_experts, out_dim, in_dim = w_fp.shape
    assert in_dim % GROUP_SIZE == 0
    fp8_dtype = torch.float8_e4m3fn
    fp8_info = torch.finfo(fp8_dtype)
    num_groups = in_dim // GROUP_SIZE
    w_q = torch.empty(num_experts, out_dim, in_dim, device=w_fp.device, dtype=fp8_dtype)
    scales = torch.empty(
        num_experts,
        out_dim,
        num_groups,
        device=w_fp.device,
        dtype=w_fp.dtype,
    )
    for expert in range(num_experts):
        w_grouped = w_fp[expert].reshape(out_dim, num_groups, GROUP_SIZE).float()
        scales_fp = (w_grouped.abs().amax(dim=-1, keepdim=True) / fp8_info.max).clamp(
            min=1e-8
        )
        q_expert = (
            (w_grouped / scales_fp).clamp(fp8_info.min, fp8_info.max).to(fp8_dtype)
        )
        w_q[expert] = q_expert.reshape(out_dim, in_dim)
        scales[expert] = scales_fp.squeeze(-1).to(w_fp.dtype)
    return w_q, scales.contiguous()


def _marlin_repack_per_expert_fp8(w_q, scales, dtype):
    """Convert E4M3 weights and per-group scales to vLLM Marlin layout."""
    num_experts, out_dim, in_dim = w_q.shape
    perm = torch.empty(0, dtype=torch.int, device=w_q.device)
    qweight_list = []
    scale_list = []
    for expert in range(num_experts):
        qweight = pack_fp8_to_int32(w_q[expert], size_k_first=False)
        qweight = vllm_ops.gptq_marlin_repack(
            b_q_weight=qweight.T.contiguous(),
            perm=perm,
            size_k=in_dim,
            size_n=out_dim,
            num_bits=8,
        )
        marlin_scales = marlin_permute_scales(
            s=scales[expert].T.to(dtype).contiguous(),
            size_k=in_dim,
            size_n=out_dim,
            group_size=GROUP_SIZE,
        )
        marlin_scales = fp8_fused_exponent_bias_into_scales(marlin_scales)
        qweight_list.append(qweight)
        scale_list.append(marlin_scales)
    return (
        torch.stack(qweight_list, dim=0).contiguous(),
        torch.stack(scale_list, dim=0).contiguous(),
    )


class FusedMarlinMoEW8A16FP8Benchmark(base.Benchmark):
    """FlagGems W(fp8)A16 against vLLM Marlin W(fp8)A16."""

    def __init__(self, op_name, torch_op, dtypes):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)
        self._weight_cache = {}

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (1, 512, 4096, 1024, 10),
            (4, 512, 4096, 1024, 10),
            (16, 512, 4096, 1024, 10),
            (64, 512, 4096, 1024, 10),
            (128, 512, 4096, 1024, 10),
            (256, 512, 4096, 1024, 10),
            (512, 512, 4096, 1024, 10),
            (1024, 512, 4096, 1024, 10),
            (4096, 512, 4096, 1024, 10),
            (16384, 512, 4096, 1024, 10),
            (32768, 512, 4096, 1024, 10),
        ]

    def get_input_iter(self, cur_dtype):
        for config in self.shapes:
            yield from self._gen(config, cur_dtype)

    def _get_quantized_weights(
        self, dtype, device, num_experts, hidden_size, intermediate_size
    ):
        cache_key = (dtype, str(device), num_experts, hidden_size, intermediate_size)
        cached = self._weight_cache.get(cache_key)
        if cached is not None:
            return cached

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
        w1_q_fp8, w1_scale_fp8 = _quantize_per_expert_fp8(w1_fp)
        w2_q_fp8, w2_scale_fp8 = _quantize_per_expert_fp8(w2_fp)
        w1_q_marlin, w1_scale_marlin = _marlin_repack_per_expert_fp8(
            w1_q_fp8, w1_scale_fp8, dtype
        )
        w2_q_marlin, w2_scale_marlin = _marlin_repack_per_expert_fp8(
            w2_q_fp8, w2_scale_fp8, dtype
        )
        cached = (
            w1_q_marlin,
            w1_scale_marlin,
            w2_q_marlin,
            w2_scale_marlin,
            w1_q_fp8,
            w1_scale_fp8,
            w2_q_fp8,
            w2_scale_fp8,
        )
        self._weight_cache[cache_key] = cached
        del w1_fp, w2_fp
        torch.cuda.empty_cache()
        return cached

    def _gen(self, config, dtype):
        num_tokens, num_experts, hidden_size, intermediate_size, topk = config
        device = flag_gems.device
        hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
        (
            w1_q_marlin,
            w1_scale_marlin,
            w2_q_marlin,
            w2_scale_marlin,
            w1_q_fp8,
            w1_scale_fp8,
            w2_q_fp8,
            w2_scale_fp8,
        ) = self._get_quantized_weights(
            dtype, device, num_experts, hidden_size, intermediate_size
        )

        gating = torch.randn(
            num_tokens, num_experts, device=device, dtype=torch.float32
        )
        topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        inputs = (
            hidden_states,
            w1_q_marlin,
            w2_q_marlin,
            w1_scale_marlin,
            w2_scale_marlin,
            w1_q_fp8,
            w2_q_fp8,
            w1_scale_fp8,
            w2_scale_fp8,
            topk_weights,
            topk_ids,
        )
        if Config.mode == consts.BenchMode.CUDAGRAPH:
            _vllm_baseline_fp8(*inputs)
            # Compile/autotune the same fixed-capacity routing path that is
            # selected during capture. PyTorch's device ops still execute
            # normally because this warmup itself is outside the graph.
            with mock.patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=True
            ):
                _gems_call_fp8(*inputs)
            torch.cuda.synchronize()
        yield inputs


def _vllm_baseline_fp8(
    hidden_states,
    w1_q_marlin,
    w2_q_marlin,
    w1_scale_marlin,
    w2_scale_marlin,
    w1_q_fp8,
    w2_q_fp8,
    w1_scale_fp8,
    w2_scale_fp8,
    topk_weights,
    topk_ids,
):
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
        quant_type_id=VLLM_QUANT_TYPE_FP8.id,
    )


def _gems_call_fp8(
    hidden_states,
    w1_q_marlin,
    w2_q_marlin,
    w1_scale_marlin,
    w2_scale_marlin,
    w1_q_fp8,
    w2_q_fp8,
    w1_scale_fp8,
    w2_scale_fp8,
    topk_weights,
    topk_ids,
):
    return gems_fused_marlin_moe(
        hidden_states=hidden_states,
        w1=w1_q_fp8,
        w2=w2_q_fp8,
        bias1=None,
        bias2=None,
        w1_scale=w1_scale_fp8,
        w2_scale=w2_scale_fp8,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        quant_type_id=QUANT_TYPE_FLOAT8_E4M3FN,
    )


@pytest.mark.fused_marlin_moe
@pytest.mark.skipif(
    not HAS_VLLM_FUSED_MARLIN_MOE, reason="vllm not installed; baseline unavailable"
)
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires NVIDIA Hopper architecture")
def test_fused_marlin_moe_fp8_weight_mxq():
    """Compare identical E4M3 weights and per-group-128 scales."""
    bench = FusedMarlinMoEW8A16FP8Benchmark(
        op_name="fused_marlin_moe_fp8_weight_mxq",
        torch_op=_vllm_baseline_fp8,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_gems_call_fp8)
    bench.run()
