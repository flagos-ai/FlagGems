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

from . import base, consts

# Quantization range for the per-tensor affine fake-quant benchmark.
QUANT_MIN = 0
QUANT_MAX = 255


def fake_quantize_learnable_per_tensor_affine_input_fn(shape, cur_dtype, device):
    inp = torch.randn(shape, dtype=cur_dtype, device=device) * 5
    scale = torch.tensor([0.1], dtype=torch.float32, device=device)
    zero_point = torch.tensor([127], dtype=torch.int32, device=device)
    yield inp, scale, zero_point, QUANT_MIN, QUANT_MAX


@pytest.mark.fake_quantize_learnable_per_tensor_affine
def test_fake_quantize_learnable_per_tensor_affine():
    bench = base.GenericBenchmark(
        op_name="fake_quantize_learnable_per_tensor_affine",
        torch_op=torch.ops.aten._fake_quantize_learnable_per_tensor_affine,
        input_fn=fake_quantize_learnable_per_tensor_affine_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
