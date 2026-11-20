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

import hashlib
import inspect
from pathlib import Path

import pytest
import torch

import flag_gems


@pytest.mark.index
@pytest.mark.skipif(flag_gems.vendor_name != "metax", reason="MetaX index backend")
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("use_graph", [False, True])
def test_metax_index_large_state_stride(index_dtype, use_graph, record_property):
    # Keep valid state indices wide before multiplying their row stride.
    import gc

    source = Path(inspect.getsourcefile(flag_gems.index))
    assert source.as_posix().endswith("runtime/backend/_metax/ops/index.py")
    record_property("metax_index_source", str(source))
    record_property(
        "metax_index_sha256", hashlib.sha256(source.read_bytes()).hexdigest()
    )

    rows = 21041
    row_stride = 102400
    row_elements = 6 * 128 * 128
    guard = 2**31
    offset = guard + 3200
    elements = offset + (rows - 1) * row_stride + row_elements
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < elements * 4 + 1024**3:
        pytest.skip("requires memory for a guarded FP32 large-offset regression")

    backing = state = packed = indices = actual = expected = graph = None
    try:
        backing = torch.empty(elements, dtype=torch.float32, device=flag_gems.device)
        state = torch.as_strided(
            backing, (rows, 6, 128, 128), (row_stride, 16384, 128, 1), offset
        )
        slots = [0, 20971, 20972, 21040]
        # Only initialize selected rows and the old wrapped addresses. The
        # prefix keeps an unpatched int32 read inside this test allocation.
        for slot in slots:
            product = slot * row_stride
            wrapped = (product + 2**31) % 2**32 - 2**31
            if product != wrapped:
                start = offset + wrapped
                assert 0 <= start < guard
                backing[start : start + row_elements].fill_(-7)
        host_rows = {}
        tail = torch.arange(row_elements, dtype=torch.int64).reshape(6, 128, 128)
        for slot in slots:
            host_rows[slot] = ((tail + slot * 7) % 251).float()
            state[slot].copy_(host_rows[slot])
        packed = torch.tensor(
            [20971, 0, 20972, 0], dtype=index_dtype, device=flag_gems.device
        )
        indices = packed[::2]
        expected = torch.stack([host_rows[20971], host_rows[20972]])
        actual = flag_gems.index(state, [indices])
        torch.cuda.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

        if use_graph:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    actual = flag_gems.index(state, [indices])
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = flag_gems.index(state, [indices])
            batches = [
                [20972, 21040],
                [0, 20971],
                [21040, 20972],
                [20971, 20971],
                [21040, 0],
                [20972, 20972],
            ]
            for _ in range(3):
                for pair in batches:
                    indices.copy_(torch.tensor(pair, dtype=index_dtype))
                    graph.replay()
                    torch.cuda.synchronize()
                    expected = torch.stack([host_rows[slot] for slot in pair])
                    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    finally:
        graph = actual = expected = state = backing = indices = packed = None
        gc.collect()
        torch.cuda.empty_cache()
