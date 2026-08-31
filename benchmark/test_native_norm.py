# Copyright 2026, The FlagOS Contributors.
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

from . import base


@pytest.mark.native_norm
def test_native_norm():
    def native_norm_input_fn(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=device)
        # p=2 for L2 norm
        yield inp, 2

    def torch_native_norm(inp, p):
        # torch.native_norm doesn't have dense CUDA impl, use vector_norm reference
        return torch.linalg.vector_norm(inp.flatten(), p)

    # dtypes handled by GenericBenchmark2DOnly.DEFAULT_DTYPES = consts.FLOAT_DTYPES
    bench = base.GenericBenchmark2DOnly(
        op_name="native_norm",
        input_fn=native_norm_input_fn,
        torch_op=torch_native_norm,
    )
    bench.run()
