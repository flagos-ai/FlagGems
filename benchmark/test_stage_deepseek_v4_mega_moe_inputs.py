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

from flag_gems.fused.stage_deepseek_v4_mega_moe_inputs import (
    stage_deepseek_v4_mega_moe_inputs,
)

from . import base


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
    return (x.view(torch.int32) >> 23).to(torch.uint8).view(torch.int32)


def _reference_stage_deepseek_v4_mega_moe_inputs(
    hidden_states,
    topk_weights,
    topk_ids,
    x_fp8,
    x_sf,
    topk_idx_out,
    topk_weights_out,
):
    num_tokens, hidden_size = hidden_states.shape
    gran_k = 32
    x_view = hidden_states.view(num_tokens, hidden_size // gran_k, gran_k)
    x_amax = x_view.abs().float().amax(dim=2).clamp(1e-4)
    scales = _ceil_to_ue8m0(x_amax / 448.0)
    x_fp8.copy_(
        (x_view * (1.0 / scales.unsqueeze(2)))
        .to(torch.float8_e4m3fn)
        .view(num_tokens, hidden_size)
    )
    x_sf.copy_(_pack_ue8m0_to_int(scales))
    topk_idx_out.copy_(topk_ids.to(torch.int64))
    topk_weights_out.copy_(topk_weights)


class StageDeepseekV4MegaMoeInputsBenchmark(base.Benchmark):
    def __init__(self):
        super().__init__(
            "stage_deepseek_v4_mega_moe_inputs",
            _reference_stage_deepseek_v4_mega_moe_inputs,
            [torch.bfloat16],
            gems_op=stage_deepseek_v4_mega_moe_inputs,
        )

    def set_shapes(self, shape_file_path=None):
        _ = shape_file_path
        self.shapes = [
            (7, 256, 8),
        ]
        self.shape_desc = "num_tokens, hidden_size, top_k"

    def get_input_iter(self, dtype):
        for num_tokens, hidden_size, top_k in self.shapes:
            generator = torch.Generator(device="cuda")
            generator.manual_seed(0)
            hidden_states = (
                torch.randn(
                    num_tokens,
                    hidden_size,
                    device="cuda",
                    dtype=torch.float32,
                    generator=generator,
                )
                * 17.0
            ).to(dtype)
            hidden_states[0, :32] = 0
            hidden_states[1, 32:64] = 1.0e-6
            hidden_states[2, 64:96] = -1.0e-6
            topk_ids = torch.randint(
                0,
                256,
                (num_tokens, top_k),
                device="cuda",
                dtype=torch.int32,
                generator=generator,
            )
            topk_weights = torch.randn(
                num_tokens,
                top_k,
                device="cuda",
                dtype=torch.float32,
                generator=generator,
            )
            x_fp8 = torch.empty(
                num_tokens,
                hidden_size,
                device="cuda",
                dtype=torch.float8_e4m3fn,
            )
            x_sf = torch.empty(
                num_tokens,
                hidden_size // 128,
                device="cuda",
                dtype=torch.int32,
            )
            topk_idx_out = torch.empty(
                num_tokens, top_k, device="cuda", dtype=torch.int64
            )
            topk_weights_out = torch.empty_like(topk_weights)
            yield (
                hidden_states,
                topk_weights,
                topk_ids,
                x_fp8,
                x_sf,
                topk_idx_out,
                topk_weights_out,
            )


@pytest.mark.stage_deepseek_v4_mega_moe_inputs
@pytest.mark.skipif(
    not _supports_fp8e4nv(), reason="requires cuda with fp8e4nv support"
)
def test_stage_deepseek_v4_mega_moe_inputs_benchmark():
    StageDeepseekV4MegaMoeInputsBenchmark().run()
