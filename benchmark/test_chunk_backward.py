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
"""Benchmark for the gradient of ``chunk``.

``chunk`` returns a tuple, so the harness's ``is_backward`` path (which times
``torch.autograd.grad((out,), xs, ...)`` on a single output tensor) does not
apply. Both sides here instead run the *same* shape of workload: a forward
``chunk`` that builds the graph, then ``backward`` over it. Timing the
FlagGems backward helper directly against native ``autograd.grad`` would not
be like-for-like -- it would compare a bare kernel launch against the whole
autograd engine.
"""

import pytest
import torch

import flag_gems

from . import base, consts

CHUNK_SHAPES = [
    (22, 64),
    (342, 1024),
    (1366, 4096),
    (22, 512, 512),
    (342, 1024, 1024),
]
CHUNKS = 3


class ChunkBackwardBenchmark(base.Benchmark):
    """Times forward-chunk + backward for one input shape at a time."""

    DEFAULT_METRICS = consts.DEFAULT_METRICS
    DEFAULT_SHAPE_DESC = "input shape"
    # ``shapes`` holds the input shapes; the chunk count is fixed so the
    # per-chunk gradient shapes stay stable across runs.
    DEFAULT_SHAPES = CHUNK_SHAPES

    def set_shapes(self, shape_file_path=None):
        self.shapes = list(CHUNK_SHAPES)

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            inp = torch.randn(shape, dtype=dtype, device=self.device)
            # One upstream gradient per chunk: ``chunk`` returns a tuple, so the
            # grads have to match the per-chunk shapes rather than the input.
            grads = [
                torch.randn(c.shape, dtype=dtype, device=self.device)
                for c in torch.chunk(inp, CHUNKS, 0)
            ]
            yield inp, grads, CHUNKS, 0

    def _benchmark_callable(self, op, inp, grads, chunks, dim):
        """Build one graph via ``op`` and time its backward.

        The graph is built once per (op, case) and reused with
        ``retain_graph=True``, so the timing measures the backward only -- the
        same contract the harness's ``is_backward`` path uses for
        single-output operators. Keeping the forward out of the timed region
        matters here because ``chunk``'s forward is a zero-copy view op.
        """
        inp = inp.detach().requires_grad_(True)
        out = op(inp, chunks, dim)
        return (
            lambda: torch.autograd.grad(out, inp, grad_outputs=grads, retain_graph=True)
        ), [inp]


@pytest.mark.chunk_backward
def test_chunk_backward():
    bench = ChunkBackwardBenchmark(
        op_name="chunk",
        torch_op=torch.chunk,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.chunk)
    bench.run()
