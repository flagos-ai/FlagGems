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

import math

import pytest
import torch

from . import base, consts

# Select every overload and contract case with the operator marker.


# The default overload follows the master workload grid. The independent
# out-buffer cases keep their existing capped YAML shape set.
DETACH_COPY_SHAPES = [(1024,), (65536,), (1048576,), (1024, 1024)]
_MAX_ELEMENTS = 2**26


class DetachCopyBenchmark(base.UnaryPointwiseBenchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS

    def set_shapes(self, shape_file_path=None):
        self.shapes = DETACH_COPY_SHAPES


class DetachCopyOutBenchmark(base.UnaryPointwiseOutBenchmark):
    MAX_ELEMENTS = _MAX_ELEMENTS

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        self.shapes = [
            shape for shape in self.shapes if math.prod(shape) <= self.MAX_ELEMENTS
        ]


@pytest.mark.detach_copy
def test_detach_copy():
    bench = DetachCopyBenchmark(
        op_name="detach_copy",
        torch_op=torch.ops.aten.detach_copy,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES + consts.BOOL_DTYPES,
    )
    bench.run()


@pytest.mark.detach_copy_out
def test_detach_copy_out():
    bench = DetachCopyOutBenchmark(
        op_name="detach_copy",
        torch_op=torch.ops.aten.detach_copy.out,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
