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

import importlib
import itertools

import pytest
import torch

import flag_gems
from flag_gems.fused.moe_sum import moe_sum_ep

from . import accuracy_utils as utils
from . import conftest as cfg

fused_moe = importlib.import_module("flag_gems.fused.fused_moe")

if cfg.QUICK_MODE:
    M_VALUES = [1, 64]
    TOP_KS = [2]
    K_VALUES = [128]
else:
    M_VALUES = [1, 33, 64, 222]
    TOP_KS = [2, 6]
    K_VALUES = [128, 511, 1024]
MOE_SHAPES = list(itertools.product(M_VALUES, TOP_KS, K_VALUES))


@pytest.mark.moe_sum
@pytest.mark.parametrize("shape", MOE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_moe_sum(shape, dtype):
    m, topk, k = shape
    inp1 = torch.randn((m, topk, k), dtype=dtype, device=flag_gems.device)
    res_out = torch.empty((m, k), dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1)
    ref_out = torch.sum(ref_inp1, dim=1)

    with flag_gems.use_gems():
        flag_gems.moe_sum(inp1, res_out)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.moe_sum
@pytest.mark.parametrize("routing", ["uniform", "all_local", "no_local", "invalid"])
def test_moe_sum_ep_skips_nonlocal_routes(routing):
    device = flag_gems.device
    dtype = torch.bfloat16
    m, topk, hidden_size = 17, 8, 511
    global_experts, local_experts = 288, 18
    shard_begin = 7 * local_experts
    torch.manual_seed(20260824)
    inp = torch.randn((m, topk, hidden_size), dtype=dtype, device=device)
    expert_map = torch.full((global_experts,), -1, dtype=torch.int32, device=device)
    expert_map[shard_begin : shard_begin + local_experts] = torch.arange(
        local_experts, dtype=torch.int32, device=device
    )

    if routing == "uniform":
        topk_ids = torch.randint(
            0, global_experts, (m, topk), dtype=torch.int64, device=device
        )
    elif routing == "all_local":
        topk_ids = (
            torch.arange(m * topk, dtype=torch.int64, device=device)
            .remainder(local_experts)
            .add(shard_begin)
            .view(m, topk)
        )
    elif routing == "no_local":
        topk_ids = torch.zeros((m, topk), dtype=torch.int64, device=device)
    else:
        topk_ids = torch.full(
            (m, topk), 2**32 + shard_begin, dtype=torch.int64, device=device
        )
        topk_ids[:, 0] = shard_begin + 3
        topk_ids[:, 1] = -1

    valid_global = (topk_ids >= 0) & (topk_ids < global_experts)
    safe_ids = torch.where(valid_global, topk_ids, 0)
    mapped = expert_map[safe_ids]
    local_mask = valid_global & (mapped >= 0) & (mapped < local_experts)
    reference = (inp.float() * local_mask[:, :, None]).sum(dim=1).to(dtype)
    output = torch.empty((m, hidden_size), dtype=dtype, device=device)
    moe_sum_ep(inp, output, topk_ids, expert_map, local_experts)
    torch.testing.assert_close(output, reference, rtol=1e-2, atol=1e-2)


@pytest.mark.moe_sum
@pytest.mark.parametrize(
    ("num_tokens", "expected"),
    [
        (64, (256, 2)),
        (96, (512, 2)),
        (128, (1024, 4)),
        (1, None),
        (63, None),
        (65, None),
        (95, None),
        (97, None),
        (127, None),
    ],
)
def test_fused_moe_ep_sum_fixed_config_selection(num_tokens, expected):
    strict_ep_config = {"BLOCK_SIZE_M": 16}
    expert_map = torch.empty(1)
    assert (
        fused_moe._get_ep_sum_fixed_config(
            strict_ep_config,
            expert_map,
            num_tokens,
        )
        == expected
    )
    assert (
        fused_moe._get_ep_sum_fixed_config(
            None,
            expert_map,
            num_tokens,
        )
        is None
    )
    assert (
        fused_moe._get_ep_sum_fixed_config(
            strict_ep_config,
            None,
            num_tokens,
        )
        is None
    )


@pytest.mark.moe_sum
@pytest.mark.parametrize(
    ("fixed_block_size", "fixed_num_warps", "error"),
    [
        (256, None, "must be provided together"),
        (None, 2, "must be provided together"),
        (256, 4, "unsupported fixed moe_sum_ep config"),
    ],
)
def test_moe_sum_ep_rejects_invalid_fixed_config(
    fixed_block_size,
    fixed_num_warps,
    error,
):
    inp = torch.empty((1, 1, 1))
    output = torch.empty((1, 1))
    topk_ids = torch.zeros((1, 1), dtype=torch.int32)
    expert_map = torch.zeros(1, dtype=torch.int32)
    with pytest.raises(ValueError, match=error):
        moe_sum_ep(
            inp,
            output,
            topk_ids,
            expert_map,
            1,
            fixed_block_size=fixed_block_size,
            fixed_num_warps=fixed_num_warps,
        )


@pytest.mark.moe_sum
@pytest.mark.parametrize(
    ("m", "fixed_block_size", "fixed_num_warps"),
    [
        (64, 256, 2),
        (96, 512, 2),
        (128, 1024, 4),
    ],
)
def test_moe_sum_ep_fixed_config_bitwise_reference(
    m,
    fixed_block_size,
    fixed_num_warps,
):
    device = flag_gems.device
    dtype = torch.bfloat16
    topk, hidden_size = 8, 4096
    global_experts, local_experts = 288, 18
    token_offsets = torch.arange(m, dtype=torch.int32, device=device)[:, None]
    route_offsets = torch.arange(topk, dtype=torch.int32, device=device)[None, :]
    topk_ids = local_experts + (token_offsets + route_offsets).remainder(
        global_experts - local_experts
    )
    topk_ids[:, :4] = (token_offsets + route_offsets[:, :4]).remainder(local_experts)
    expert_map = torch.full((global_experts,), -1, dtype=torch.int32, device=device)
    expert_map[:local_experts] = torch.arange(
        local_experts, dtype=torch.int32, device=device
    )
    local_mask = expert_map[topk_ids.to(torch.int64)] >= 0
    torch.manual_seed(20260824 + m)
    inp = torch.randn((m, topk, hidden_size), dtype=dtype, device=device)
    inp.masked_fill_(~local_mask[:, :, None], float("nan"))
    reference = torch.zeros((m, hidden_size), dtype=torch.float32, device=device)
    for route_idx in range(topk):
        reference += torch.where(
            local_mask[:, route_idx, None],
            inp[:, route_idx].float(),
            torch.zeros((), dtype=torch.float32, device=device),
        )
    reference = reference.to(dtype)
    output = torch.empty_like(reference)

    moe_sum_ep(
        inp,
        output,
        topk_ids,
        expert_map,
        local_experts,
        fixed_block_size=fixed_block_size,
        fixed_num_warps=fixed_num_warps,
    )
    assert torch.equal(output, reference)


@pytest.mark.moe_sum
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Graph requires CUDA")
@pytest.mark.parametrize(
    ("m", "hidden_size", "fixed_config"),
    [
        (8, 128, None),
        (64, 4096, (256, 2)),
        (96, 4096, (512, 2)),
        (128, 4096, (1024, 4)),
    ],
)
def test_moe_sum_ep_cuda_graph_replays_updated_routing(
    m,
    hidden_size,
    fixed_config,
):
    device = flag_gems.device
    dtype = torch.bfloat16
    topk = 8
    global_experts, local_experts = 288, 18
    inp = torch.randn((m, topk, hidden_size), dtype=dtype, device=device)
    output = torch.empty((m, hidden_size), dtype=dtype, device=device)
    topk_ids = torch.zeros((m, topk), dtype=torch.int32, device=device)
    expert_map = torch.full((global_experts,), -1, dtype=torch.int32, device=device)
    expert_map[:local_experts] = torch.arange(
        local_experts, dtype=torch.int32, device=device
    )

    fixed_kwargs = {}
    if fixed_config is not None:
        fixed_kwargs = {
            "fixed_block_size": fixed_config[0],
            "fixed_num_warps": fixed_config[1],
        }
    for _ in range(3):
        moe_sum_ep(
            inp,
            output,
            topk_ids,
            expert_map,
            local_experts,
            **fixed_kwargs,
        )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        moe_sum_ep(
            inp,
            output,
            topk_ids,
            expert_map,
            local_experts,
            **fixed_kwargs,
        )

    for local_routes in (0, 2, 8):
        new_ids = torch.full_like(topk_ids, local_experts)
        if local_routes:
            new_ids[:, :local_routes] = torch.arange(
                local_routes, dtype=torch.int32, device=device
            )
        topk_ids.copy_(new_ids)
        graph.replay()
        torch.cuda.synchronize()
        first_replay = output.clone()
        for _ in range(20):
            graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(output, first_replay)
        local_mask = topk_ids < local_experts
        reference = (inp.float() * local_mask[:, :, None]).sum(dim=1).to(dtype)
        torch.testing.assert_close(output, reference, rtol=1e-2, atol=1e-2)
