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

from typing import Generator

import pytest
import torch

from . import base


class AmpForeachNonFiniteCheckAndUnscaleBenchmark(base.Benchmark):
    """
    Benchmark class for _amp_foreach_non_finite_check_and_unscale_ operation.
    This operation takes a list of tensors, a found_inf tensor, and an inv_scale
    scalar tensor, and checks for non-finite values while unscaling.
    """

    # Common sizes for gradient tensors in training scenarios covering small to medium workloads
    DEFAULT_SHAPES = [
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
    ]
    # shapes describe the first tensor; a second tensor at half size is generated
    DEFAULT_SHAPE_DESC = "M, N"

    def set_more_shapes(self):
        # larger shapes for comprehensive benchmark level
        more_shapes_2d = [(1024, 2**i) for i in range(2, 14, 4)]
        more_shapes_3d = [(64, 2**i, 64) for i in range(2, 10, 4)]
        return more_shapes_2d + more_shapes_3d

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            second_shape = (max(1, shape[0] // 2),) + shape[1:]
            case = self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={
                        "tensors": [shape, second_shape],
                        "found_inf": (),
                        "inv_scale": (),
                    },
                    params={},
                    builder_args=(shape, second_shape),
                ),
            )
            if self._skip_native_half_case(case):
                base.Config.skipped_case_ids.add(case.case_id)
                print(
                    f"SKIPPED native benchmark case {case.case_id}: "
                    "Kunlunxin native Half AMP unscale returns incorrect values"
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
        shape, second_shape = case.builder_args[0].builder_args
        tensors = [
            torch.randn(shape, device=self.device, dtype=case.dtype),
            torch.randn(second_shape, device=self.device, dtype=case.dtype),
        ]
        inv_scale = torch.tensor(2.0, device=self.device, dtype=torch.float32)
        found_inf = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        return tensors, found_inf, inv_scale

    def get_input_iter(self, dtype) -> Generator:
        for case in self.get_case_iter(dtype):
            yield self.build_inputs(case)


@pytest.mark.amp_foreach_non_finite_check_and_unscale_
def test_amp_foreach_non_finite_check_and_unscale_():
    bench = AmpForeachNonFiniteCheckAndUnscaleBenchmark(
        op_name="amp_foreach_non_finite_check_and_unscale_",
        torch_op=torch._amp_foreach_non_finite_check_and_unscale_,
        # bfloat16 is not supported by the CUDA kernel for this operator
        dtypes=[torch.float16, torch.float32],
    )
    bench.run()
