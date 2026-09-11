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


@pytest.mark.infinitely_differentiable_gelu_backward
def test_infinitely_differentiable_gelu_backward():
    def input_fn(shape, dtype, device):
        grad = torch.randn(shape, dtype=dtype, device=device)
        self_input = torch.randn(shape, dtype=dtype, device=device)
        yield grad, self_input

    bench = base.GenericBenchmark(
        input_fn=input_fn,
        op_name="infinitely_differentiable_gelu_backward",
        torch_op=torch.ops.aten.infinitely_differentiable_gelu_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
