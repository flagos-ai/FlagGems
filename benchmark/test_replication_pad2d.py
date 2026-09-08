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

from . import base, consts


def _input_fn(config, dtype, device):
    shape, padding = config
    x = torch.randn(shape, dtype=dtype, device=device)
    yield x, list(padding)


def _input_fn_out(config, dtype, device):
    shape, padding = config
    x = torch.randn(shape, dtype=dtype, device=device)
    pad_left, pad_right, pad_top, pad_bottom = padding
    H_out = shape[-2] + pad_top + pad_bottom
    W_out = shape[-1] + pad_left + pad_right
    out_shape = (*shape[:-2], H_out, W_out)
    out = torch.empty(out_shape, dtype=dtype, device=device)
    yield x, list(padding), {"out": out}


class ReplicationPad2dBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        # Representative 4D shapes covering small (8x8) to ResNet-like (56x56)
        # spatial sizes with moderate uniform padding.
        self.shapes = [
            ((2, 3, 8, 8), (1, 1, 2, 2)),
            ((4, 8, 64, 64), (2, 2, 2, 2)),
            ((16, 32, 56, 56), (2, 2, 2, 2)),
            ((32, 64, 28, 28), (3, 3, 3, 3)),
            ((64, 128, 14, 14), (1, 2, 3, 4)),
        ]

    def set_more_shapes(self):
        return None

    def get_input_iter(self, cur_dtype):
        for config in self.shapes:
            yield from _input_fn(config, cur_dtype, self.device)


class ReplicationPad2dOutBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        # Same shapes as forward so the _out variant can be compared fairly.
        self.shapes = [
            ((2, 3, 8, 8), (1, 1, 2, 2)),
            ((4, 8, 64, 64), (2, 2, 2, 2)),
            ((16, 32, 56, 56), (2, 2, 2, 2)),
            ((32, 64, 28, 28), (3, 3, 3, 3)),
            ((64, 128, 14, 14), (1, 2, 3, 4)),
        ]

    def set_more_shapes(self):
        return None

    def get_input_iter(self, cur_dtype):
        for config in self.shapes:
            yield from _input_fn_out(config, cur_dtype, self.device)


@pytest.mark.replication_pad2d
def test_replication_pad2d():
    bench = ReplicationPad2dBenchmark(
        op_name="replication_pad2d",
        torch_op=torch.ops.aten.replication_pad2d,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.replication_pad2d_out
def test_replication_pad2d_out():
    bench = ReplicationPad2dOutBenchmark(
        op_name="replication_pad2d_out",
        torch_op=torch.ops.aten.replication_pad2d.out,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
