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

from typing import Generator

import pytest
import torch

from . import base, consts, utils


class AvgPool1dBenchmark(base.GenericBenchmark):
    def get_input_iter(self, dtype) -> Generator:
        shapes_3d = [
            (4, 3, 224),
            (16, 64, 128),
            (32, 128, 64),
            (64, 256, 32),
            (128, 512, 16),
        ]

        for shape in shapes_3d:
            yield from self.input_fn(shape, dtype, self.device)


def avg_pool1d_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)

    # Common case
    yield inp, {
        "kernel_size": [3],
        "stride": [2],
        "padding": [1],
        "ceil_mode": False,
        "count_include_pad": True,
    }

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        # With count_include_pad=False
        yield inp, {
            "kernel_size": [3],
            "stride": [2],
            "padding": [1],
            "ceil_mode": False,
            "count_include_pad": False,
        }

        # With ceil_mode
        yield inp, {
            "kernel_size": [3],
            "stride": [2],
            "padding": [1],
            "ceil_mode": True,
            "count_include_pad": True,
        }


@pytest.mark.avg_pool1d
def test_avg_pool1d():
    bench = AvgPool1dBenchmark(
        input_fn=avg_pool1d_input_fn,
        op_name="avg_pool1d",
        torch_op=torch.ops.aten.avg_pool1d,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
