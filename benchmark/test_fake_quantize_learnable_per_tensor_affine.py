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

# Quantization range for the per-tensor affine fake-quant benchmark.
QUANT_MIN = 0
QUANT_MAX = 255


class FakeQuantizeBenchmark(base.GenericBenchmark):
    def build_inputs(self, case):
        try:
            return super().build_inputs(case)
        except pytest.skip.Exception:
            base.Config.skipped_case_ids.add(case.case_id)
            raise


def _case_fn(shape, cur_dtype):
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={"quant_min": QUANT_MIN, "quant_max": QUANT_MAX},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, cur_dtype, device):
    if flag_gems.vendor_name == "kunlunxin" and cur_dtype == torch.float16:
        pytest.skip(
            "Kunlunxin native Half fake-quant reference returns float32 instead of Half"
        )
    (shape,) = plan.builder_args
    inp = torch.randn(shape, dtype=cur_dtype, device=device) * 5
    scale = torch.tensor([0.1], dtype=torch.float32, device=device)
    zero_point = torch.tensor([127], dtype=torch.int32, device=device)
    return inp, scale, zero_point, QUANT_MIN, QUANT_MAX


@pytest.mark.fake_quantize_learnable_per_tensor_affine
@pytest.mark.underscore_fake_quantize_learnable_per_tensor_affine
def test_fake_quantize_learnable_per_tensor_affine():
    bench = FakeQuantizeBenchmark(
        op_name="_fake_quantize_learnable_per_tensor_affine",
        torch_op=torch.ops.aten._fake_quantize_learnable_per_tensor_affine,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
