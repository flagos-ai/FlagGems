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


class OuterBenchmark(base.GenericBenchmark2DOnly):
    def set_more_shapes(self):
        return []

    def get_input_iter(self, dtype) -> Generator:
        for m, n in self.shapes:
            yield from self.input_fn(m, n, dtype, self.device)


def _input_fn(m, n, cur_dtype, device):
    inp1 = torch.randn([m], dtype=cur_dtype, device=device)
    inp2 = torch.randn([n], dtype=cur_dtype, device=device)
    yield inp1, inp2


@pytest.mark.outer
def test_outer():
    bench = OuterBenchmark(
        op_name="outer",
        input_fn=_input_fn,
        torch_op=torch.Tensor.outer,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
