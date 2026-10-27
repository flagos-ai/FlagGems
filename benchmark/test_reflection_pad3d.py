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


class ReflectionPad3dBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        # (shape, padding) pairs covering various volume sizes and padding configs
        self.shapes = [
            ((2, 4, 8, 16, 16), (1, 1, 1, 1, 1, 1)),
            ((1, 3, 16, 32, 32), (2, 3, 2, 3, 1, 1)),
            ((2, 4, 16, 32, 64), (1, 1, 2, 2, 2, 2)),
            ((1, 1, 32, 64, 128), (0, 4, 0, 4, 0, 4)),
        ]

    def set_more_shapes(self):
        return None

    def get_input_iter(self, dtype):
        for config in self.shapes:
            yield from _input_fn(config, dtype, self.device)


@pytest.mark.reflection_pad3d
def test_reflection_pad3d():
    bench = ReflectionPad3dBenchmark(
        op_name="reflection_pad3d",
        torch_op=torch.ops.aten.reflection_pad3d,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()


def _input_fn_out(config, dtype, device):
    shape, padding = config
    x = torch.randn(shape, dtype=dtype, device=device)
    pad_l, pad_r, pad_t, pad_b, pad_f, pad_ba = padding
    D_out = x.shape[-3] + pad_f + pad_ba
    H_out = x.shape[-2] + pad_t + pad_b
    W_out = x.shape[-1] + pad_l + pad_r
    out_shape = (*x.shape[:-3], D_out, H_out, W_out)
    out = torch.empty(out_shape, dtype=dtype, device=device)
    yield x, list(padding), {"out": out}


class ReflectionPad3dOutBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        # (shape, padding) pairs covering various volume sizes and padding configs
        self.shapes = [
            ((2, 4, 8, 16, 16), (1, 1, 1, 1, 1, 1)),
            ((1, 3, 16, 32, 32), (2, 3, 2, 3, 1, 1)),
            ((2, 4, 16, 32, 64), (1, 1, 2, 2, 2, 2)),
            ((1, 1, 32, 64, 128), (0, 4, 0, 4, 0, 4)),
        ]

    def set_more_shapes(self):
        return None

    def get_input_iter(self, dtype):
        for config in self.shapes:
            yield from _input_fn_out(config, dtype, self.device)


@pytest.mark.reflection_pad3d_out
def test_reflection_pad3d_out():
    bench = ReflectionPad3dOutBenchmark(
        op_name="reflection_pad3d_out",
        torch_op=torch.ops.aten.reflection_pad3d.out,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
