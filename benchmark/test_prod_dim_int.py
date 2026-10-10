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

from . import base, consts


class ProdDimIntBenchmark(base.UnaryReductionBenchmark):
    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield (inp,)

            if inp.ndim >= 2:
                yield (inp, 0)
                yield (inp, -1)

            if inp.ndim >= 3:
                yield (inp, 1)


@pytest.mark.prod_dim_int
def test_prod_dim_int():
    bench = ProdDimIntBenchmark(
        op_name="prod_dim_int", torch_op=torch.prod, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()
