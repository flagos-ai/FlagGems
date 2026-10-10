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

from . import base, consts

PAD1D_SHAPES = [(2, 3, 7), (4, 16, 64), (8, 32, 256), (32, 256)]
PAD1D_PADDINGS = [(0, 0), (1, 2), (3, 1)]

PAD2D_SHAPES = [(2, 3, 8, 8), (3, 16, 32), (1, 64, 256, 256), (4, 64, 256, 512)]
PAD2D_PADDINGS = [(0, 0, 0, 0), (1, 2, 3, 4), (0, 2, 3, 4)]


class _Pad1dBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = PAD1D_SHAPES
        self.shape_desc = "shape, padding"


class _Pad2dBackwardBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = PAD2D_SHAPES
        self.shape_desc = "shape, padding"


def _pad1d_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    for padding in PAD1D_PADDINGS:
        pl, pr = padding
        w_out = shape[-1] + pl + pr
        out_shape = (*shape[:-1], w_out)
        out = torch.empty(out_shape, dtype=dtype, device=device)
        yield inp, {"padding": padding, "out": out}


def _pad2d_backward_input_fn(shape, dtype, device):
    for padding in PAD2D_PADDINGS:
        pl, pr, pt, pb = padding
        if len(shape) == 4:
            n, c, h, w = shape
            padded_shape = (n, c, h + pt + pb, w + pl + pr)
        else:
            c, h, w = shape
            padded_shape = (c, h + pt + pb, w + pl + pr)
        x = torch.randn(shape, dtype=dtype, device=device)
        grad_output = torch.randn(padded_shape, dtype=dtype, device=device)
        grad_input = torch.empty(shape, dtype=dtype, device=device)
        yield grad_output, x, {"padding": padding, "grad_input": grad_input}


@pytest.mark.replication_pad1d_out
def test_benchmark_replication_pad1d_out():
    bench = _Pad1dBenchmark(
        op_name="replication_pad1d_out",
        torch_op=torch.ops.aten.replication_pad1d.out,
        gems_op=flag_gems.replication_pad1d_out,
        input_fn=_pad1d_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.replication_pad2d_backward_grad_input
def test_benchmark_replication_pad2d_backward_grad_input():
    bench = _Pad2dBackwardBenchmark(
        op_name="replication_pad2d_backward_grad_input",
        torch_op=torch.ops.aten.replication_pad2d_backward.grad_input,
        gems_op=flag_gems.replication_pad2d_backward_grad_input,
        input_fn=_pad2d_backward_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
