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

import flag_gems

from . import base, consts, utils


def max_pool3d_with_indices_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, {
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "dilation": 1,
        "ceil_mode": False,
    }
    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        # Non-cubic kernel/stride/padding
        if shape[-3] > 5 and shape[-2] > 5 and shape[-1] > 5:
            yield inp, {
                "kernel_size": (2, 3, 3),
                "stride": (1, 2, 2),
                "padding": (0, 1, 1),
                "dilation": 1,
                "ceil_mode": False,
            }
        # With ceil_mode
        yield inp, {
            "kernel_size": 3,
            "stride": 2,
            "padding": 1,
            "dilation": 1,
            "ceil_mode": True,
        }


class MaxPool3DWithIndicesBenchmark(base.GenericBenchmark):
    def get_input_iter(self, dtype) -> Generator:
        # Representative 5-D (N, C, D, H, W) tensors covering typical 3D-CNN
        # feature-map sizes from shallow/large to deep/small.
        shapes_5d = [
            (4, 3, 16, 56, 56),
            (8, 64, 8, 28, 28),
            (16, 128, 4, 14, 14),
            (32, 256, 2, 7, 7),
        ]

        for shape in shapes_5d:
            yield from self.input_fn(shape, dtype, self.device)


@pytest.mark.max_pool3d_with_indices
def test_max_pool3d_with_indices():
    bench = MaxPool3DWithIndicesBenchmark(
        input_fn=max_pool3d_with_indices_input_fn,
        op_name="max_pool3d_with_indices",
        torch_op=lambda inp, **kwargs: torch.nn.functional.max_pool3d(
            inp, return_indices=True, **kwargs
        ),
        gems_op=flag_gems.max_pool3d_with_indices,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
