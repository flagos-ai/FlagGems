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

"""Packed word conversion is independent of a tensor's register layout."""

import pytest
import torch
import triton
import triton.language as tl

from flag_gems.fused.dsv41.fp4_fp8_tle import decode_fp4_e2m1_words_to_fp8x4

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (9, 0)
    or triton.__version__.split("+")[0] not in ("3.6.0", "3.7.1"),
    reason="validated SM90 Triton 3.6.0 / 3.7.1 packed word conversion",
)


@triton.jit
def _decode(words, output):
    x = tl.program_id(0) * 256 + tl.arange(0, 256)
    word = tl.load(words + x).to(tl.uint32)
    tl.store(output + x, decode_fp4_e2m1_words_to_fp8x4(word))


def _expected(words):
    codes = (
        words.to(torch.int32)[:, None] >> (4 * torch.arange(4, device="cuda"))[None, :]
    ) & 15
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device="cuda")
    value = lut[codes & 7] * torch.where((codes & 8) != 0, -1.0, 1.0)
    return (
        torch.where((codes & 7) == 0, 0.0, value)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )


@pytest.mark.parametrize("dtype", [torch.uint16, torch.int32])
def test_all_words_and_graph_refresh(dtype):
    a = torch.arange(65536, device="cuda", dtype=torch.int32).to(dtype)
    storage = a.clone()
    output = torch.empty(65536, device="cuda", dtype=torch.int32)
    _decode[(256,)](storage, output)
    assert torch.equal(output.view(torch.uint8).reshape(-1, 4), _expected(a))
    # The storage sign bit of FP4 zero is intentionally canonicalized to +0.
    assert int(output.view(torch.uint8).reshape(-1, 4)[8, 0]) == 0
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _decode[(256,)](storage, output)
    for value in [a, a.flip(0), a]:
        storage.copy_(value)
        output.fill_(0x12345678)
        graph.replay()
        assert torch.equal(output.view(torch.uint8).reshape(-1, 4), _expected(value))
