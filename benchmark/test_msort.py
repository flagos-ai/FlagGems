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

from . import base, consts, utils


class MsortBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (8, 1024),
            (32, 4096),
            (128, 8192),
            (512, 4096),
            (32, 131072),
        ]
        if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            self.shapes.extend(self.set_more_shapes())

    def set_more_shapes(self):
        return [(8, 262144), (128, 65536), (512, 16384), (32, 524288)]


def msort_input_fn(shape, dtype, device):
    yield utils.generate_tensor_input(shape, dtype, device),


def msort_out_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, {"out": torch.empty_like(inp)}


@pytest.mark.msort
def test_msort():
    MsortBenchmark(
        op_name="msort",
        torch_op=torch.msort,
        input_fn=msort_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    ).run()


@pytest.mark.msort_out
def test_msort_out():
    MsortBenchmark(
        op_name="msort_out",
        torch_op=torch.msort,
        input_fn=msort_out_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    ).run()
