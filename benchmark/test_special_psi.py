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

import flag_gems

from . import base, utils

# consts.FLOAT_DTYPES leaves bfloat16 ungated and has no float64, so gate both
# on the backend's static capabilities.
_BENCH_DTYPES = [torch.float16, torch.float32]
if flag_gems.runtime.device.support_bf16:
    _BENCH_DTYPES.append(torch.bfloat16)
if flag_gems.runtime.device.support_fp64:
    _BENCH_DTYPES.append(torch.float64)


class SpecialPsiBenchmark(base.UnaryPointwiseBenchmark):
    """UnaryPointwiseBenchmark with an explicit float64 input build.

    benchmark.utils.generate_tensor_input has no float64 branch, so that dtype
    is built here directly; every other dtype keeps the shared builder. The
    metadata-only get_case_iter is inherited unchanged, so listing, timing and
    replay all use the same descriptors and shapes.
    """

    def build_inputs(self, case):
        shape = case.builder_args[0].builder_args[0]
        if case.dtype == torch.float64:
            return (torch.randn(shape, dtype=torch.float64, device=self.device),)
        return (utils.generate_tensor_input(shape, case.dtype, self.device),)


@pytest.mark.special_psi
def test_special_psi():
    bench = SpecialPsiBenchmark(
        op_name="special_psi",
        torch_op=torch.ops.aten.special_psi,
        dtypes=_BENCH_DTYPES,
        gems_op=getattr(flag_gems, "special_psi", None),
    )
    bench.run()
