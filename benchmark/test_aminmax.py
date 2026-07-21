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

from . import base, consts, utils


def aminmax_input_fn(shape, cur_dtype, device):
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    # Test dim=None (whole tensor reduction)
    yield inp,
    # Test dim=-1 (last dimension)
    yield inp, {"dim": -1}
    # Test dim=0 (first dimension)
    if len(shape) > 1:
        yield inp, {"dim": 0}


class AminmaxBenchmark(base.UnaryReductionBenchmark):
    def get_input_iter(self, dtype):
        for shape in self.shapes:
            yield from aminmax_input_fn(shape, dtype, self.device)


@pytest.mark.aminmax
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_aminmax():
    bench = AminmaxBenchmark(
        op_name="aminmax",
        torch_op=torch.aminmax,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
