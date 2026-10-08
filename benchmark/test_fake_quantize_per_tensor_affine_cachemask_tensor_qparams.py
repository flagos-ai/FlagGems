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

# int8 symmetric quantization range used by the fake-quant benchmarks below.
QUANT_MIN = -128
QUANT_MAX = 127


def _fake_quant_input_fn(shape, cur_dtype, device):
    inp = torch.randn(shape, dtype=cur_dtype, device=device)
    scale = torch.tensor(0.1, dtype=torch.float32, device=device)
    zero_point = torch.tensor(0, dtype=torch.int32, device=device)
    fake_quant_enabled = torch.tensor(1, dtype=torch.int64, device=device)
    yield (
        inp,
        scale,
        zero_point,
        fake_quant_enabled,
        QUANT_MIN,
        QUANT_MAX,
    )


@pytest.mark.fake_quantize_per_tensor_affine_cachemask_tensor_qparams
def test_fake_quantize_per_tensor_affine_cachemask_tensor_qparams():
    bench = base.GenericBenchmark(
        op_name="fake_quantize_per_tensor_affine_cachemask_tensor_qparams",
        torch_op=torch.ops.aten._fake_quantize_per_tensor_affine_cachemask_tensor_qparams,
        dtypes=consts.FLOAT_DTYPES,
        input_fn=_fake_quant_input_fn,
    )
    bench.run()
