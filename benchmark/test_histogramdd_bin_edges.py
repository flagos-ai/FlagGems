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

from flag_gems.ops._histogramdd_bin_edges import _histogramdd_bin_edges as gems_op

from . import base

# PyTorch only ships a CPU implementation for ``_histogramdd_bin_edges``, so the
# baseline runs there (the input is moved to CPU on each call) while the GEMS
# path runs on the accelerator device.
HIST_DTYPES = [torch.float32]
HIST_BINS = (5, 3, 2)

# Fixed shapes whose innermost dim matches the bin count above. These keep the
# recorded shape meaningful (rather than the framework's generic 2D shapes).
HIST_SHAPES = [
    (1024, 3),
    (4096, 3),
    (16384, 3),
    (65536, 3),
    (262144, 3),
]


class _FixedShapesBenchmark(base.GenericBenchmark2DOnly):
    """Generic 2D benchmark that pins the shape list to ``HIST_SHAPES``."""

    def init_user_config(self):
        # Bypass the yaml-based shape loading; dtypes/metrics still use Config.
        self.mode = base.Config.mode
        self.set_dtypes(base.Config.user_desired_dtypes)
        self.set_metrics(base.Config.user_desired_metrics)
        # Pin shapes whose innermost dim matches HIST_BINS so the recorded shape
        # stays meaningful instead of the framework's generic 2-D sweep.
        self.shapes = [tuple(s) for s in HIST_SHAPES]


def _input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    yield inp, {"bins": list(HIST_BINS)}


def _torch_op(inp, *, bins, range=None, weight=None, density=False):
    # Fall back to the CPU implementation since aten has no CUDA kernel.
    ref_inp = inp.to("cpu")
    out = torch._histogramdd_bin_edges(
        ref_inp, bins, range=range, weight=weight, density=density
    )
    return [t.to(inp.device) for t in out]


@pytest.mark.histogramdd_bin_edges
def test_histogramdd_bin_edges():
    bench = _FixedShapesBenchmark(
        op_name="histogramdd_bin_edges",
        input_fn=_input_fn,
        torch_op=_torch_op,
        gems_op=gems_op,
        dtypes=HIST_DTYPES,
    )
    bench.run()


@pytest.mark.histogramdd_bin_edges_out
def test_histogramdd_bin_edges_out():
    def _torch_out_op(inp, *, bins, out=None, range=None, weight=None, density=False):
        ref_inp = inp.to("cpu")
        ref_out = [t.to("cpu") for t in out]
        torch.ops.aten._histogramdd_bin_edges.out(
            ref_inp, bins, range=range, weight=weight, density=density, out=ref_out
        )
        for gpu, cpu in zip(out, ref_out):
            if gpu.numel() != cpu.numel():
                gpu.resize_(cpu.numel())
            gpu.copy_(cpu.to(inp.device))

    def _gems_out_op(inp, *, bins, out=None, range=None, weight=None, density=False):
        from flag_gems.ops._histogramdd_bin_edges import _histogramdd_bin_edges_out

        return _histogramdd_bin_edges_out(
            inp, bins, range=range, weight=weight, density=density, out=out
        )

    def _out_input_fn(shape, dtype, device):
        n_dims = len(HIST_BINS)
        inp = torch.randn(shape, dtype=dtype, device=device)
        out = [torch.empty(0, dtype=dtype, device=device) for _ in range(n_dims)]
        yield inp, {"bins": list(HIST_BINS), "out": out}

    bench = _FixedShapesBenchmark(
        op_name="histogramdd_bin_edges_out",
        input_fn=_out_input_fn,
        torch_op=_torch_out_op,
        gems_op=_gems_out_op,
        dtypes=HIST_DTYPES,
    )
    bench.run()
