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

from . import base


def _input_fn(shape, dtype, device):
    # Generate quantized tensor for benchmarking
    fp_tensor = torch.randn(shape, device="cpu")
    q_tensor = torch.quantize_per_tensor(
        fp_tensor, scale=0.1, zero_point=0, dtype=torch.qint8
    ).to(device)
    yield q_tensor,


@pytest.mark.dequantize
def test_dequantize():
    # Dequantize operates on quantized tensor input;
    # no FLOAT_DTYPES parametrization needed (input always qint8, output always float32).
    bench = base.GenericBenchmarkExcluse1D(
        op_name="dequantize",
        input_fn=_input_fn,
        dtypes=[torch.qint8],
        torch_op=torch.dequantize,
    )
    bench.run()
