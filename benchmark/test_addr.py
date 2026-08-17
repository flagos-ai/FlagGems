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


class AddrBenchmark(base.BlasBenchmark):
    def set_more_shapes(self):
        return []

    def get_input_iter(self, dtype) -> Generator:
        for shape in self.shapes:
            m, n = shape[0], shape[1]
            yield from self.input_fn(m, n, dtype, self.device)


def _input_fn(m, n, cur_dtype, device):
    inp1 = torch.randn([m, n], dtype=cur_dtype, device=device)
    inp2 = torch.randn([m], dtype=cur_dtype, device=device)
    inp3 = torch.randn([n], dtype=cur_dtype, device=device)
    yield inp1, inp2, inp3, {"alpha": 0.5, "beta": 0.5}


@pytest.mark.addr
def test_addr():
    bench = AddrBenchmark(
        op_name="addr",
        input_fn=_input_fn,
        torch_op=torch.Tensor.addr,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
