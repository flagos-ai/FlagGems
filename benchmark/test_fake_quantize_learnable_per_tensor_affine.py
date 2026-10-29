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


def _case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"input": shape, "scale": (1,), "zero_point": (1,)},
        params={"quant_min": QUANT_MIN, "quant_max": QUANT_MAX},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    inp = torch.randn(shape, dtype=dtype, device=device) * 5
    scale = torch.tensor([0.1], dtype=torch.float32, device=device)
    zero_point = torch.tensor([127], dtype=torch.int32, device=device)
    return inp, scale, zero_point, QUANT_MIN, QUANT_MAX


class FakeQuantizeBenchmark(base.GenericBenchmark):
    def get_case_iter(self, dtype):
        for case in super().get_case_iter(dtype):
            if self._skip_native_half_case(case):
                base.Config.skipped_case_ids.add(case.case_id)
                print(
                    f"SKIPPED native benchmark case {case.case_id}: "
                    "Kunlunxin native Half fake-quant reference returns float32 instead of Half"
                )
                continue
            yield case

    def _skip_native_half_case(self, case):
        native_requested = (
            not getattr(base.Config, "list_cases", False)
            and not base.Config.skip_native
            and "latency_base" in self.to_bench_metrics
            and not getattr(base.Config, "preflight_only", False)
            and not getattr(base.Config, "profile_only", False)
        )
        return (
            base.vendor_name == "kunlunxin"
            and case.dtype == torch.float16
            and native_requested
        )

    def build_inputs(self, case):
        return super().build_inputs(case)


@pytest.mark.fake_quantize_learnable_per_tensor_affine
def test_fake_quantize_learnable_per_tensor_affine():
    bench = FakeQuantizeBenchmark(
        op_name="fake_quantize_learnable_per_tensor_affine",
        torch_op=torch.ops.aten._fake_quantize_learnable_per_tensor_affine,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
