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

import os

import pytest
import torch
import yaml

from . import base, consts

SHAPE_FILE = os.path.join(os.path.dirname(__file__), "core_shapes.yaml")

with open(SHAPE_FILE, "r") as f:
    _yaml_config = yaml.safe_load(f)

PIXEL_UNSHUFFLE_SHAPES = [
    (tuple(s[:-1]), s[-1]) for s in _yaml_config["pixel_unshuffle"]["shapes"]
]


def _input_fn(config, dtype, device):
    shape, downscale_factor = config
    x = torch.randn(shape, dtype=dtype, device=device)
    yield x, downscale_factor


class PixelUnshuffleBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = PIXEL_UNSHUFFLE_SHAPES

    def set_more_shapes(self):
        return []

    def get_input_iter(self, cur_dtype):
        for config in self.shapes:
            yield from _input_fn(config, cur_dtype, self.device)


@pytest.mark.pixel_unshuffle
def test_pixel_unshuffle():
    bench = PixelUnshuffleBenchmark(
        op_name="pixel_unshuffle",
        torch_op=torch.ops.aten.pixel_unshuffle,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()


def _input_fn_out(config, dtype, device):
    shape, downscale_factor = config
    x = torch.randn(shape, dtype=dtype, device=device)
    N, C, H, W = shape
    r = downscale_factor
    out = torch.empty((N, C * r * r, H // r, W // r), dtype=dtype, device=device)
    yield x, downscale_factor, {"out": out}


class PixelUnshuffleOutBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = PIXEL_UNSHUFFLE_SHAPES

    def set_more_shapes(self):
        return []

    def get_input_iter(self, cur_dtype):
        for config in self.shapes:
            yield from _input_fn_out(config, cur_dtype, self.device)


@pytest.mark.pixel_unshuffle_out
def test_pixel_unshuffle_out():
    bench = PixelUnshuffleOutBenchmark(
        op_name="pixel_unshuffle_out",
        torch_op=torch.ops.aten.pixel_unshuffle.out,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
