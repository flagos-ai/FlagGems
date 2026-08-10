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


@pytest.mark.is_non_overlapping_and_dense
def test_is_non_overlapping_and_dense():
    def is_non_overlapping_and_dense_input_fn(shape, dtype, device):
        # The operator only inspects sizes and strides, so the tensor is left
        # uninitialised and no values need to be prepared.
        yield (torch.empty(shape, dtype=dtype, device=device),)

    bench = base.GenericBenchmark(
        input_fn=is_non_overlapping_and_dense_input_fn,
        op_name="is_non_overlapping_and_dense",
        torch_op=torch.ops.aten.is_non_overlapping_and_dense,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
