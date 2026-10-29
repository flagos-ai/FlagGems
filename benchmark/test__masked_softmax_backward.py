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


class MaskedSoftmaxBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        # Masked softmax is an attention operation, so the softmax (last)
        # reduction dimension stays bounded (sequence length) in practice; the
        # kernel loads a full reduction row into one Triton block. Use
        # representative 2-D/3-D shapes and avoid the generic 1-D 1G-element
        # default whose single row would exceed the per-block tensor limit.
        self.shapes = [
            (64, 64),
            (4096, 4096),
            (64, 512, 512),
            (8, 32, 2048),
        ]

    def set_more_shapes(self):
        return None

    def get_case_iter(self, dtype) -> Generator:
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": list(shape)},
                    params={"dim": -1},
                    builder_args=(shape,),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        (shape,) = plan.builder_args
        dtype = case.dtype
        inp = torch.randn(shape, dtype=dtype, device=self.device)
        mask = torch.rand(shape, device=self.device) < 0.3
        # dim=-1 reduction; build a consistent forward output.
        output = torch.ops.aten._masked_softmax(inp, mask, -1, 2)
        output = torch.where(mask, torch.zeros_like(output), output)
        grad_output = torch.randn(shape, dtype=dtype, device=self.device)
        return grad_output, output, mask, -1


@pytest.mark.underscore_masked_softmax_backward
def test__masked_softmax_backward():
    bench = MaskedSoftmaxBackwardBenchmark(
        op_name="_masked_softmax_backward",
        torch_op=torch.ops.aten._masked_softmax_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
