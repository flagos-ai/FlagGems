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


def hash_tensor_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    # Reduce over the last dim: the contiguous axis, which is the common
    # row-hash use case and keeps the reduction size tied to the shape.
    yield inp, [-1]


@pytest.mark.hash_tensor
def test_hash_tensor():
    bench = base.GenericBenchmark2DOnly(
        input_fn=hash_tensor_input_fn,
        op_name="hash_tensor",
        torch_op=torch.ops.aten.hash_tensor,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
