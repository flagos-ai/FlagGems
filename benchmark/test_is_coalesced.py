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

import math

import pytest
import torch

from . import base, consts


@pytest.mark.is_coalesced
def test_is_coalesced():
    def is_coalesced_input_fn(shape, dtype, device):
        # is_coalesced only reads the coalesced flag of a sparse COO tensor, so the
        # number of stored elements is kept small and independent of the dense
        # shape: allocating one value per dense element would exhaust memory on the
        # largest benchmark shapes without changing what is measured.
        nnz = min(1024, math.prod(shape))
        indices = torch.stack(
            [torch.randint(0, dim, (nnz,), device=device) for dim in shape]
        )
        values = torch.randn(nnz, dtype=dtype, device=device)
        yield (torch.sparse_coo_tensor(indices, values, shape),)

    bench = base.GenericBenchmark(
        input_fn=is_coalesced_input_fn,
        op_name="is_coalesced",
        torch_op=torch.ops.aten.is_coalesced,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
