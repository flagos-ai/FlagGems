# Copyright 2026, The FlagOS Contributors.
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


class NestedTensorFromTensorListBenchmark(base.Benchmark):
    """
    Benchmark for _nested_tensor_from_tensor_list operator.
    """

    def set_shapes(self, shape_file_path=None):
        # Component-shape lists covering small/medium/large nested tensor cases.
        # Every component shares the trailing dimension so they form a valid
        # nested tensor.
        self.shapes = [
            [(2048, 4096), (2048, 4096), (2048, 4096)],
            [(64, 512, 512), (32, 512, 512), (96, 512, 512)],
            [(1024, 1024), (2048, 1024), (512, 1024), (1536, 1024)],
        ]

    def get_input_iter(self, cur_dtype):
        for shapes in self.shapes:
            tensor_list = [
                torch.randn(shape, dtype=cur_dtype, device=self.device)
                for shape in shapes
            ]
            yield (tensor_list,)

    def get_tflops(self, op, *args, **kwargs):
        return 0.0


@pytest.mark.nested_tensor_from_tensor_list
@pytest.mark.parametrize(
    "dtype",
    consts.FLOAT_DTYPES,
)
def test_nested_tensor_from_tensor_list(dtype):
    bench = NestedTensorFromTensorListBenchmark(
        op_name="nested_tensor_from_tensor_list",
        torch_op=flag_gems._nested_tensor_from_tensor_list,
        dtypes=[dtype],
    )
    bench.run()
