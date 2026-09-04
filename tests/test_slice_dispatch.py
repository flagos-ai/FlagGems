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

from . import accuracy_utils as utils
from . import conftest as cfg


def _slice_dispatch(inp, case):
    if case == "empty":
        # Regression: vLLM MinPLogitsProcessor initializes its buffer with x[:0].
        return inp[:0]
    if case == "full":
        return inp[:]
    if case == "range":
        return inp[1:3]
    if case == "aten_default_step":
        return torch.ops.aten.slice.Tensor(inp, 0, 1, 3)
    if case == "explicit_step":
        return inp[::2]
    raise AssertionError(f"Unknown slice case: {case}")


def _gems_slice(inp, case):
    if case == "empty":
        return flag_gems.slice(inp, end=0)
    if case == "full":
        return flag_gems.slice(inp)
    if case == "range":
        return flag_gems.slice(inp, start=1, end=3)
    if case == "aten_default_step":
        return flag_gems.slice(inp, 0, 1, 3)
    if case == "explicit_step":
        return flag_gems.slice(inp, step=2)
    raise AssertionError(f"Unknown slice case: {case}")


@pytest.mark.slice
@pytest.mark.parametrize(
    "case", ["empty", "full", "range", "aten_default_step", "explicit_step"]
)
@pytest.mark.parametrize(
    "dtype", [torch.float32] if cfg.QUICK_MODE else utils.FLOAT_DTYPES
)
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("mode", ["eager", "graph"])
def test_slice_dispatch_defaults(case, dtype, strided, mode):
    if mode == "graph" and flag_gems.device != "cuda":
        pytest.skip("CUDA Graph regression requires CUDA")

    # Deterministic data; no random state or model/service dependency.
    base = torch.arange(16, dtype=dtype, device=flag_gems.device)
    inp = base[::2] if strided else base
    ref_out = _slice_dispatch(utils.to_reference(inp), case)

    def assert_output(out, reference):
        assert out.shape == reference.shape
        assert out.dtype == inp.dtype
        assert out.device == inp.device
        utils.gems_assert_equal(out, reference)

    if mode == "eager":
        out = _gems_slice(inp, case)
    else:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                _gems_slice(inp, case)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = _gems_slice(inp, case)
        graph.replay()
        torch.cuda.synchronize()

    assert_output(out, ref_out)
    if mode == "graph":
        # Replay must use new values at the same input address, not cached values.
        inp.add_(17)
        ref_out = _slice_dispatch(utils.to_reference(inp), case)
        graph.replay()
        torch.cuda.synchronize()
        assert_output(out, ref_out)
