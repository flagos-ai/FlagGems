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
from flag_gems.ops.flash_kernel import flash_fwd_splitkv_combine_kernel
from flag_gems.runtime import torch_device_fn

SENTINEL = -123.0
LAYOUTS = ["bshd", "bhsd_storage", "gapped"]
DTYPES = [torch.float16, torch.bfloat16]


def make_output(batch, seq, heads, dim, dtype, layout):
    if layout == "bshd":
        strides = (seq * heads * dim, heads * dim, dim, 1)
    elif layout == "bhsd_storage":
        strides = (heads * seq * dim, dim, seq * dim, 1)
    else:
        strides = (seq * (heads * dim + 16) + 32, heads * dim + 16, dim, 1)
    extent = (batch - 1) * strides[0] + (seq - 1) * strides[1]
    extent += (heads - 1) * strides[2] + dim
    # Keep even the old, incorrect writes inside this allocation. Check both
    # the trailing guard and any holes in the logical output view afterwards.
    size = extent + batch * heads * seq * dim + 16 * strides[1]
    backing = torch.full((size,), SENTINEL, dtype=dtype, device=flag_gems.device)
    out = backing.as_strided((batch, seq, heads, dim), strides)
    occupied = torch.zeros(size, dtype=torch.bool)
    occupied.as_strided(out.shape, strides).fill_(True)
    return out, backing, occupied


def assert_guards(backing, occupied):
    assert torch.all(backing.cpu()[~occupied] == SENTINEL)


def make_partials(batch, seq, heads, dim, splits):
    generator = torch.Generator().manual_seed(17)
    partial = torch.randn((splits, batch, heads, seq, dim), generator=generator)
    lse = torch.randn((splits, batch, heads, seq), generator=generator)
    # One empty KV partition must contribute zero to the weighted output.
    lse[-1, ..., 0] = -float("inf")
    return partial, lse


def combine_reference(partial, lse):
    weights = torch.softmax(lse, dim=0)
    out = (partial * weights[..., None]).sum(0).permute(0, 2, 1, 3)
    return out, torch.logsumexp(lse, dim=0)


def launch_combine(out, final_lse, partial, lse, block_m):
    splits, batch, heads, seq, dim = partial.shape
    flash_fwd_splitkv_combine_kernel[(triton.cdiv(batch * heads * seq, block_m),)](
        out_ptr=out,
        lse_ptr=final_lse,
        out_splits_ptr=partial,
        lse_splits_ptr=lse,
        head_size=dim,
        out_split_stride=partial.stride(0),
        lse_split_stride=lse.stride(0),
        out_b_stride=out.stride(0),
        out_s_stride=out.stride(1),
        out_h_stride=out.stride(2),
        n_splits=splits,
        BLOCK_M=block_m,
        BLOCK_K=triton.next_power_of_2(dim),
        q_total=batch * heads * seq,
        MAX_N_SPLITS=triton.next_power_of_2(splits),
        num_heads=heads,
        seqlen_q=seq,
    )


@pytest.mark.flash_attention_forward
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize(
    "batch,seq,heads,dim,splits,block_m",
    [
        (1, 256, 16, 96, 2, 8),
        (2, 17, 3, 64, 3, 8),
        (2, 5, 3, 32, 5, 16),
        (1, 9, 2, 128, 2, 4),
    ],
)
def test_splitkv_combine_layout(batch, seq, heads, dim, splits, block_m, layout, dtype):
    partial_cpu, lse_cpu = make_partials(batch, seq, heads, dim, splits)
    expected, expected_lse = combine_reference(partial_cpu, lse_cpu)
    out, backing, occupied = make_output(batch, seq, heads, dim, dtype, layout)
    partial = partial_cpu.to(flag_gems.device)
    lse = lse_cpu.to(flag_gems.device)
    final_lse = torch.empty((batch, heads, seq), device=flag_gems.device)
    launch_combine(out, final_lse, partial, lse, block_m)
    torch.testing.assert_close(out.cpu(), expected.to(dtype), atol=0.016, rtol=0.016)
    torch.testing.assert_close(final_lse.cpu(), expected_lse, atol=1e-5, rtol=1e-5)
    assert_guards(backing, occupied)


@pytest.mark.flash_attention_forward
@pytest.mark.skipif(
    flag_gems.device != "cuda", reason="Requires CUDA-compatible graphs"
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", ["bshd", "bhsd_storage"])
def test_splitkv_combine_graph_replay(layout, dtype):
    partial_cpu, lse_cpu = make_partials(2, 17, 3, 96, 3)
    partial, lse = [x.to(flag_gems.device) for x in (partial_cpu, lse_cpu)]
    out, backing, occupied = make_output(2, 17, 3, 96, dtype, layout)
    final_lse = torch.empty((2, 3, 17), device=flag_gems.device)
    # Compile outside capture and warm up on the capture stream.
    launch_combine(out, final_lse, partial, lse, 8)
    stream = torch_device_fn.Stream()
    stream.wait_stream(torch_device_fn.current_stream())
    with torch_device_fn.stream(stream):
        launch_combine(out, final_lse, partial, lse, 8)
    torch_device_fn.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        launch_combine(out, final_lse, partial, lse, 8)
    for step in range(2):
        changed_partial = partial_cpu + step + 1
        changed_lse = lse_cpu.clone()
        changed_lse[0] += step + 0.5
        partial.copy_(changed_partial)
        lse.copy_(changed_lse)
        backing.fill_(SENTINEL)
        graph.replay()
        expected, expected_lse = combine_reference(changed_partial, changed_lse)
        torch.testing.assert_close(
            out.cpu(), expected.to(dtype), atol=0.016, rtol=0.016
        )
        torch.testing.assert_close(final_lse.cpu(), expected_lse, atol=1e-5, rtol=1e-5)
        assert_guards(backing, occupied)
