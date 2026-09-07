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

pytestmark = pytest.mark.skipif(flag_gems.device != "npu", reason="Ascend only")
CASES = [
    ((4, 128), 8),
    ((8, 256), 16),
    ((64, 1024), 32),
    ((64, 4096), 64),
    ((64, 8192), 128),
    ((128, 32768), 256),
    ((2, 33, 128), 5),
    ((3, 257), 17),
]


def _check(q, s, k, group_size, largest, values, indices):
    n = q.shape[-1]
    ref = q.float() * s.float().repeat_interleave(group_size, -1)[..., :n]
    values, indices = values.cpu(), indices.cpu()
    torch.testing.assert_close(
        values, torch.topk(ref, k, largest=largest).values.bfloat16(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        values, torch.gather(ref, -1, indices).bfloat16(), rtol=0, atol=0
    )
    assert indices.dtype == torch.int64
    ordered = torch.sort(indices).values
    assert (ordered[..., 1:] != ordered[..., :-1]).all()


def _run(q, s, k, group_size, largest):
    result = flag_gems.topk_w8a16_fp8(
        q.npu(), s.npu(), k, group_size=group_size, largest=largest
    )
    _check(q, s, k, group_size, largest, *result)


@pytest.mark.topk_w8a16_fp8
@pytest.mark.parametrize("shape,k", CASES)
@pytest.mark.parametrize("largest", [True, False])
@pytest.mark.parametrize("row_scale", [True, False])
def test_topk_fp8(shape, k, largest, row_scale):
    torch.manual_seed(127)
    n = shape[-1]
    g = n if row_scale else 128
    q = torch.randn(shape).to(torch.float8_e4m3fn)
    s = (torch.rand(shape[:-1] + ((n + g - 1) // g,)) * 1.5 + 0.1).bfloat16()
    _run(q, s, k, g, largest)


@pytest.mark.topk_w8a16_fp8
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("largest", [True, False])
@pytest.mark.parametrize(
    "kind", ["ties", "subnormal", "negative_scale", "zero_k", "all_k"]
)
def test_topk_fp8_edges(dtype, largest, kind):
    torch.manual_seed(98)
    q = torch.randn((3, 128)).to(dtype)
    if kind == "ties":
        q = torch.ones((3, 128)).to(dtype)
    if kind == "subnormal":
        q = (torch.arange(128).float().repeat(3, 1) / 65536).to(dtype)
    s = (
        torch.tensor([[-0.3], [0.0], [1.3]])
        if kind == "negative_scale"
        else torch.ones((3, 1), dtype=torch.bfloat16)
    )
    k = 0 if kind == "zero_k" else 128 if kind == "all_k" else 16
    _run(q, s, k, 128, largest)


@pytest.mark.topk_w8a16_fp8
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("largest", [True, False])
def test_topk_fp8_maximum_row(dtype, largest):
    torch.manual_seed(33)
    _run(
        torch.randn((2, 32768)).to(dtype),
        torch.ones((2, 1), dtype=torch.bfloat16),
        512,
        32768,
        largest,
    )


@pytest.mark.topk_w8a16_fp8
@pytest.mark.parametrize(
    "dtype,max_code", [(torch.float8_e4m3fn, 126), (torch.float8_e5m2, 123)]
)
@pytest.mark.parametrize("largest", [True, False])
def test_topk_fp8_all_finite_encodings(dtype, max_code, largest):
    codes = torch.cat(
        [torch.arange(max_code + 1), torch.arange(max_code + 1) + 128]
    ).to(torch.uint8)
    q = codes.view(dtype).reshape(1, -1)
    _run(q, torch.ones((1, 1), dtype=torch.bfloat16), q.shape[-1], q.shape[-1], largest)


@pytest.mark.topk_w8a16_fp8
def test_topk_fp8_graph_replay_changed_inputs():
    torch.manual_seed(811)
    q_cpu = torch.randn((64, 4096)).to(torch.float8_e4m3fn)
    s_cpu = torch.ones((64, 1), dtype=torch.bfloat16)
    q, s = q_cpu.npu(), s_cpu.npu()
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        flag_gems.topk_w8a16_fp8(q, s, 64, group_size=4096)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        values, indices = flag_gems.topk_w8a16_fp8(q, s, 64, group_size=4096)
    for changed in (False, True):
        if changed:
            q_cpu = (torch.randn((64, 4096)) * 2).to(torch.float8_e4m3fn)
            s_cpu = torch.linspace(-1, 1, 64).reshape(64, 1).bfloat16()
            q.view(torch.uint8).copy_(q_cpu.view(torch.uint8).npu())
            s.copy_(s_cpu.npu())
        graph.replay()
        torch.npu.synchronize()
        _check(q_cpu, s_cpu, 64, 4096, True, values, indices)


@pytest.mark.topk_w8a16_fp8
def test_topk_fp8_large_cutoff_ties_and_zero_scale():
    q = torch.ones((64, 4096)).to(torch.float8_e4m3fn)
    s = ((torch.arange(64) % 3) - 1).reshape(64, 1).bfloat16()
    for largest in (True, False):
        _run(q, s, 64, 4096, largest)


@pytest.mark.topk_w8a16_fp8
def test_topk_fp8_unaligned_storage_and_current_stream():
    torch.manual_seed(953)
    q_cpu = torch.randn((3, 4096)).to(torch.float8_e4m3fn)
    storage = torch.empty(q_cpu.numel() + 1, device="npu", dtype=torch.uint8)
    q = storage[1:].view(torch.float8_e4m3fn).reshape(q_cpu.shape)
    scale_storage = torch.empty(4, device="npu", dtype=torch.bfloat16)
    s = scale_storage[1:].reshape(3, 1)
    s_cpu = torch.tensor([[0.5], [-1.5], [2.0]], dtype=torch.bfloat16)
    for _ in range(2):
        q.view(torch.uint8).copy_(q_cpu.view(torch.uint8).npu())
        s.copy_(s_cpu.npu())
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            values, indices = flag_gems.topk_w8a16_fp8(q, s, 64, group_size=4096)
        stream.synchronize()
        _check(q_cpu, s_cpu, 64, 4096, True, values, indices)
        q_cpu = (-torch.randn(q_cpu.shape)).to(torch.float8_e4m3fn)
