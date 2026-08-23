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


def new_ones_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    yield {"tensor": inp, "size": shape},


@pytest.mark.new_ones
def test_new_ones():
    bench = base.GenericBenchmark(
        op_name="new_ones",
        torch_op=lambda tensor, size: tensor.new_ones(size),
        input_fn=new_ones_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
