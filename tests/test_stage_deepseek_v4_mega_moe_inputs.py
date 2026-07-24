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

import flag_gems.testing as fg_testing
from flag_gems.fused.stage_deepseek_v4_mega_moe_inputs import (
    stage_deepseek_v4_mega_moe_inputs,
)


def _supports_fp8e4nv():
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


def _ceil_to_ue8m0(x: torch.Tensor):
    bits = x.abs().float().view(torch.int32)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
    return (exp.clamp(1, 254) << 23).view(torch.float32)


def _pack_ue8m0_to_int(x: torch.Tensor):
    assert x.dtype == torch.float32 and x.size(-1) % 4 == 0
    assert (x.view(torch.int32) & ((1 << 23) - 1) == 0).all()
    return (x.view(torch.int32) >> 23).to(torch.uint8).view(torch.int32)


def _reference_stage_inputs(hidden_states, topk_weights, topk_ids):
    num_tokens, hidden_size = hidden_states.shape
    gran_k = 32
    x_view = hidden_states.view(num_tokens, hidden_size // gran_k, gran_k)
    x_amax = x_view.abs().float().amax(dim=2).clamp(1e-4)
    scales = _ceil_to_ue8m0(x_amax / 448.0)
    x_fp8 = (x_view * (1.0 / scales.unsqueeze(2))).to(torch.float8_e4m3fn)
    x_fp8 = x_fp8.view(num_tokens, hidden_size).contiguous()
    x_sf = _pack_ue8m0_to_int(scales)
    return x_fp8, x_sf, topk_ids.to(torch.int64), topk_weights.clone()


def _make_inputs(num_tokens, hidden_size, top_k):
    device = "cuda"
    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    hidden_states = (
        torch.randn(
            num_tokens,
            hidden_size,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        * 17.0
    ).to(torch.bfloat16)
    if num_tokens >= 3 and hidden_size >= 96:
        hidden_states[0, :32] = 0
        hidden_states[1, 32:64] = 1.0e-6
        hidden_states[2, 64:96] = -1.0e-6

    topk_ids = torch.randint(
        0,
        256,
        (num_tokens, top_k),
        device=device,
        dtype=torch.int32,
        generator=generator,
    )
    topk_weights = torch.randn(
        num_tokens,
        top_k,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    return hidden_states, topk_weights, topk_ids


@pytest.mark.parametrize("num_tokens, hidden_size, top_k", [(1, 128, 1), (7, 256, 8)])
@pytest.mark.skipif(
    not _supports_fp8e4nv(), reason="requires cuda with fp8e4nv support"
)
def test_stage_deepseek_v4_mega_moe_inputs_accuracy(num_tokens, hidden_size, top_k):
    hidden_states, topk_weights, topk_ids = _make_inputs(num_tokens, hidden_size, top_k)
    ref_x, ref_x_sf, ref_topk_idx, ref_topk_weights = _reference_stage_inputs(
        hidden_states, topk_weights, topk_ids
    )

    x_fp8 = torch.empty_like(ref_x)
    x_sf = torch.empty_like(ref_x_sf)
    topk_idx_out = torch.empty_like(ref_topk_idx)
    topk_weights_out = torch.empty_like(ref_topk_weights)

    stage_deepseek_v4_mega_moe_inputs(
        hidden_states,
        topk_weights,
        topk_ids,
        x_fp8,
        x_sf,
        topk_idx_out,
        topk_weights_out,
    )
    torch.cuda.synchronize()

    fg_testing.assert_equal(x_fp8.view(torch.uint8), ref_x.view(torch.uint8))
    fg_testing.assert_equal(x_sf, ref_x_sf)
    fg_testing.assert_equal(topk_idx_out, ref_topk_idx)
    fg_testing.assert_equal(
        topk_weights_out.view(torch.uint8), ref_topk_weights.view(torch.uint8)
    )


@pytest.mark.skipif(
    not _supports_fp8e4nv(), reason="requires cuda with fp8e4nv support"
)
def test_stage_deepseek_v4_mega_moe_inputs_rejects_bad_hidden_size():
    hidden_states, topk_weights, topk_ids = _make_inputs(1, 256, 1)
    bad_hidden_states = hidden_states[:, :129]
    x_fp8 = torch.empty_like(bad_hidden_states, dtype=torch.float8_e4m3fn)
    x_sf = torch.empty((1, 2), device="cuda", dtype=torch.int32)
    topk_idx_out = torch.empty((1, 1), device="cuda", dtype=torch.int64)
    topk_weights_out = torch.empty((1, 1), device="cuda", dtype=torch.float32)

    with pytest.raises(ValueError, match="multiple of 128"):
        stage_deepseek_v4_mega_moe_inputs(
            bad_hidden_states,
            topk_weights,
            topk_ids,
            x_fp8,
            x_sf,
            topk_idx_out,
            topk_weights_out,
        )
