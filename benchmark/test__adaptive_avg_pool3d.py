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

from . import base, consts


def adaptive_avg_pool3d_input_fn(shape, dtype, device):
    inp = base.generate_tensor_input(shape, dtype, device)
    # output_size must be (D, H, W).
    yield inp, (7, 7, 7)
    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        yield inp, (1, 1, 1)
        yield inp, (2, 2, 2)
        yield inp, (4, 4, 4)


class AdaptiveAvgPool3dBenchmark(base.GenericBenchmark):
    def get_input_iter(self, cur_dtype) -> Generator:
        shapes_5d = [
            (2, 3, 8, 8, 8),  # Small input
            (4, 16, 16, 16, 16),  # Medium input
            (2, 64, 32, 32, 32),  # Medium input
            (1, 3, 64, 64, 64),  # Typical video clip
            (4, 128, 16, 16, 16),  # Typical intermediate feature map
        ]

        for shape in shapes_5d:
            yield from self.input_fn(shape, cur_dtype, self.device)


# The yaml ids are ``adaptive_avg_pool3d`` / ``adaptive_avg_pool3d_out``, but the
# aten name starts with an underscore and the stripped name collides with the
# separate ``adaptive_avg_pool3d`` op, so the #6359 convention maps the
# implementation file to ``underscore_adaptive_avg_pool3d``. CI selects tests by
# that derived marker (tools/test-op.sh), while the rule-check marker gate uses
# the yaml id -- so declare both names (cf. tests/test__sobol_engine_scramble_.py).
@pytest.mark.adaptive_avg_pool3d
@pytest.mark.underscore_adaptive_avg_pool3d
def test_perf_adaptive_avg_pool3d():
    bench = AdaptiveAvgPool3dBenchmark(
        input_fn=adaptive_avg_pool3d_input_fn,
        op_name="adaptive_avg_pool3d",
        torch_op=torch.ops.aten._adaptive_avg_pool3d,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def adaptive_avg_pool3d_out_input_fn(shape, dtype, device):
    inp = base.generate_tensor_input(shape, dtype, device)

    def _run(output_size):
        out = torch.empty(
            (shape[0], shape[1], *output_size), dtype=dtype, device=device
        )
        return inp, output_size, {"out": out}

    yield _run((7, 7, 7))
    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        yield _run((1, 1, 1))
        yield _run((2, 2, 2))
        yield _run((4, 4, 4))


class AdaptiveAvgPool3dOutBenchmark(base.GenericBenchmark):
    def get_input_iter(self, cur_dtype) -> Generator:
        shapes_5d = [
            (2, 3, 8, 8, 8),  # Small input
            (4, 16, 16, 16, 16),  # Medium input
            (2, 64, 32, 32, 32),  # Medium input
            (1, 3, 64, 64, 64),  # Typical video clip
            (4, 128, 16, 16, 16),  # Typical intermediate feature map
        ]

        for shape in shapes_5d:
            yield from self.input_fn(shape, cur_dtype, self.device)


@pytest.mark.adaptive_avg_pool3d_out
@pytest.mark.underscore_adaptive_avg_pool3d_out
def test_perf_adaptive_avg_pool3d_out():
    bench = AdaptiveAvgPool3dOutBenchmark(
        input_fn=adaptive_avg_pool3d_out_input_fn,
        op_name="adaptive_avg_pool3d_out",
        torch_op=torch.ops.aten._adaptive_avg_pool3d.out,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
