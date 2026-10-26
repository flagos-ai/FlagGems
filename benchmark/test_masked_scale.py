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

import flag_gems

from . import base, utils


class MaskedScaleBenchmark(base.Benchmark):
    def set_more_shapes(self):
        special_shapes_2d = [(1024, 2**i) for i in range(0, 20, 4)]
        shapes_3d = [(64, 64, 2**i) for i in range(0, 20, 4)]
        return special_shapes_2d + shapes_3d

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            dims = tuple(shape)
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": list(dims), "mask": list(dims)},
                    params={"scale": 2.0},
                    builder_args=(dims,),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        shape = plan.builder_args[0]
        torch.manual_seed(0)
        inp = utils.generate_tensor_input(shape, case.dtype, self.device)
        if flag_gems.vendor_name == "cambricon":
            mask = torch.randint(0, 2, shape, dtype=torch.uint8, device="cpu").to(
                self.device
            )
        else:
            mask = torch.randint(0, 2, shape, dtype=torch.uint8, device=self.device)
        return inp, mask, plan.params["scale"]

    def get_tflops(self, op, *args, **kwargs):
        return args[0].numel()


@pytest.mark.masked_scale
@pytest.mark.underscore_masked_scale
def test_masked_scale():
    bench = MaskedScaleBenchmark(
        op_name="_masked_scale",
        torch_op=torch.ops.aten._masked_scale,
        dtypes=[torch.float32],
    )
    bench.run()
