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


class VdotBenchmark(base.BlasBenchmark):
    def set_more_shapes(self):
        return []

    def get_input_iter(self, dtype) -> Generator:
        for shape in self.shapes:
            m = shape[0]
            yield from self.input_fn(m, dtype, self.device)


@pytest.mark.vdot
def test_vdot():
    def vdot_input_fn(m, cur_dtype, device):
        inp1 = torch.randn([m], dtype=cur_dtype, device=device)
        inp2 = torch.randn([m], dtype=cur_dtype, device=device)
        yield inp1, inp2

    bench = VdotBenchmark(
        input_fn=vdot_input_fn,
        op_name="vdot",
        torch_op=torch.Tensor.vdot,
        dtypes=consts.COMPLEX_DTYPES + consts.FLOAT_DTYPES,
    )
    bench.run()
