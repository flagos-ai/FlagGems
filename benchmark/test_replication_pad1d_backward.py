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

# (batch, width) pairs covering small to medium tensors for pad1d backward
REPLICATION_PAD1D_BACKWARD_SHAPES = [
    (2, 3),
    (4, 8),
    (8, 16),
    (1, 32),
    (4, 64),
    (8, 128),
]


class ReplicationPad1dBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = REPLICATION_PAD1D_BACKWARD_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            yield from self._inputs_for_shape(shape, cur_dtype)

    def _inputs_for_shape(self, shape, cur_dtype):
        if len(shape) == 2:
            B, W = shape
        else:
            B, W = 1, shape[0]
        padding = (1, 2)
        x = torch.randn(B, W, dtype=cur_dtype, device=self.device)
        W_out = W + sum(padding)
        grad = torch.ones(B, W_out, dtype=cur_dtype, device=self.device)
        yield grad, x, padding

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": list(shape)},
                    params={"padding": [1, 2]},
                    builder_args=(shape,),
                ),
            )

    def build_inputs(self, case):
        return next(
            self._inputs_for_shape(case.builder_args[0].builder_args[0], case.dtype)
        )


@pytest.mark.replication_pad1d_backward
def test_replication_pad1d_backward():
    bench = ReplicationPad1dBackwardBenchmark(
        op_name="replication_pad1d_backward",
        torch_op=torch.ops.aten.replication_pad1d_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
