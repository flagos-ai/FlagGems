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

from . import base, utils


def _binary_input_fn(shape, dtype, device):
    inp1 = utils.generate_tensor_input(shape, dtype, device)
    inp2 = utils.generate_tensor_input(shape, dtype, device)
    yield inp1, inp2


# Note: tl.math.div_rz only supports float32, so we only benchmark float32
@pytest.mark.trunc_divide
def test_trunc_divide():
    bench = base.GenericBenchmark(
        op_name="trunc_divide",
        input_fn=_binary_input_fn,
        torch_op=lambda a, b: torch.div(a, b, rounding_mode="trunc"),
        dtypes=[torch.float32],
    )
    bench.run()


@pytest.mark.trunc_divide_
def test_trunc_divide_inplace():
    bench = base.GenericBenchmark(
        op_name="trunc_divide_",
        input_fn=_binary_input_fn,
        torch_op=lambda a, b: a.div_(b, rounding_mode="trunc"),
        dtypes=[torch.float32],
        is_inplace=True,
    )
    bench.run()
