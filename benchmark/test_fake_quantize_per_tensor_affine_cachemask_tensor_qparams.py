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
from .generated_operator_utils import OperatorBenchmark

QUANT_MIN = -128
QUANT_MAX = 127


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"input": list(dims), "scale": [], "zero_point": [], "enabled": []},
        params={"quant_min": QUANT_MIN, "quant_max": QUANT_MAX},
        builder_args=(dims,),
    )


def _build_inputs_fn(plan, dtype, device):
    if base.vendor_name == "hygon" and dtype == torch.bfloat16:
        pytest.skip("Hygon native fake-quant benchmark reference lacks bfloat16")
    (shape,) = plan.builder_args
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=device)
    scale = torch.tensor(0.1, dtype=torch.float32, device=device)
    zero_point = torch.tensor(0, dtype=torch.int32, device=device)
    enabled = torch.tensor(1, dtype=torch.int64, device=device)
    return inp, scale, zero_point, enabled, QUANT_MIN, QUANT_MAX


@pytest.mark.fake_quantize_per_tensor_affine_cachemask_tensor_qparams
@pytest.mark.underscore_fake_quantize_per_tensor_affine_cachemask_tensor_qparams
def test_fake_quantize_per_tensor_affine_cachemask_tensor_qparams():
    bench = OperatorBenchmark(
        op_name="_fake_quantize_per_tensor_affine_cachemask_tensor_qparams",
        torch_op=torch.ops.aten._fake_quantize_per_tensor_affine_cachemask_tensor_qparams,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
