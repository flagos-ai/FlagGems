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

PIXEL_SHUFFLE_SHAPES = [
    (tuple(s[:-1]), s[-1]) for s in _yaml_config["pixel_shuffle"]["shapes"]
]


def _input_fn(config, dtype, device):
    shape, upscale_factor = config
    x = torch.randn(shape, dtype=dtype, device=device)
    yield x, upscale_factor


class PixelShuffleBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = PIXEL_SHUFFLE_SHAPES

    def set_more_shapes(self):
        return []

    def get_input_iter(self, cur_dtype):
        for config in self.shapes:
            yield from _input_fn(config, cur_dtype, self.device)


@pytest.mark.pixel_shuffle
def test_pixel_shuffle():
    bench = PixelShuffleBenchmark(
        op_name="pixel_shuffle",
        torch_op=torch.ops.aten.pixel_shuffle,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()


def _input_fn_out(config, dtype, device):
    shape, upscale_factor = config
    x = torch.randn(shape, dtype=dtype, device=device)
    N, C, H, W = shape
    r = upscale_factor
    out = torch.empty((N, C // (r * r), H * r, W * r), dtype=dtype, device=device)
    yield x, upscale_factor, {"out": out}


class PixelShuffleOutBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = PIXEL_SHUFFLE_SHAPES

    def set_more_shapes(self):
        return []

    def get_input_iter(self, cur_dtype):
        for config in self.shapes:
            yield from _input_fn_out(config, cur_dtype, self.device)


@pytest.mark.pixel_shuffle_out
def test_pixel_shuffle_out():
    bench = PixelShuffleOutBenchmark(
        op_name="pixel_shuffle_out",
        torch_op=torch.ops.aten.pixel_shuffle.out,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
