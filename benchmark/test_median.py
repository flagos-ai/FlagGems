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


class MedianNoDimBenchmark(base.Benchmark):
    DEFAULT_SHAPE_FILES = "benchmark/core_shapes.yaml"
    DEFAULT_SHAPE_DESC = "input shape"

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            yield (utils.generate_tensor_input(shape, cur_dtype, self.device),)


class MedianReductionBenchmark(base.Benchmark):
    DEFAULT_SHAPE_FILES = "benchmark/core_shapes.yaml"
    DEFAULT_SHAPE_DESC = "input shape or [input shape, dim, keepdim]"

    def get_input_iter(self, cur_dtype) -> Generator:
        for case_id, shape_spec in enumerate(self.shapes):
            if shape_spec and isinstance(shape_spec[0], (list, tuple)):
                shape = tuple(shape_spec[0])
                dim = int(shape_spec[1])
                keepdim = bool(shape_spec[2]) if len(shape_spec) > 2 else False
            else:
                shape = shape_spec
                keepdim = case_id % 3 == 0
                if len(shape) == 1:
                    dim = 0
                elif case_id % 2 == 0:
                    dim = len(shape) - 1
                else:
                    dim = 0
            inp = utils.generate_tensor_input(shape, cur_dtype, self.device)
            yield inp, dim, {"keepdim": keepdim}


@pytest.mark.median
def test_median():
    bench = MedianNoDimBenchmark(
        op_name="median",
        torch_op=torch.median,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
    )
    bench.run()


@pytest.mark.median
def test_median_dim():
    bench = MedianReductionBenchmark(
        op_name="median_dim",
        torch_op=torch.median,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
    )
    bench.run()
