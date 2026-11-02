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

import flag_gems

from . import base, consts


def native_norm_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    # p=2 for L2 norm
    yield inp, 2


def torch_native_norm(inp, p):
    # aten::native_norm has no CUDA implementation, so the baseline is the
    # equivalent full-tensor vector_norm.
    return torch.linalg.vector_norm(inp.flatten(), p)


@pytest.mark.native_norm
def test_native_norm():
    # The gems side is passed explicitly so the measured latency is the
    # operator itself rather than the GEMS dispatch layer around it; the
    # dispatch path otherwise adds a flat ~20us per call, which is larger
    # than the kernel for every small shape.
    bench = base.GenericBenchmark2DOnly(
        op_name="native_norm",
        input_fn=native_norm_input_fn,
        torch_op=torch_native_norm,
        gems_op=flag_gems.native_norm,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
