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

from . import base


_TEST_DTYPES = (
    [torch.float32, torch.float64]
    if flag_gems.vendor_name == "nvidia"
    else [torch.float32]
)


class LuWithInfoBenchmark(base.Benchmark):
    DEFAULT_SHAPES = [
        (64, 64),
        (256, 256),
        (512, 512),
        (1024, 1024),
        (8, 128, 128),
    ]

    def set_shapes(self, shape_file_path=None):
        self.shapes = self.DEFAULT_SHAPES
        self.shape_desc = "input shape"

    def supports_cases(self):
        return True

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"self": list(shape)},
                    params={"pivot": True, "check_errors": False},
                    builder_args=(tuple(shape),),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        (shape,) = plan.builder_args
        torch.manual_seed(0)
        inp = torch.randn(shape, dtype=case.dtype, device=self.device)
        return inp, {"pivot": True, "check_errors": False}


@pytest.mark.lu_with_info
@pytest.mark.underscore_lu_with_info
def test_lu_with_info():
    LuWithInfoBenchmark(
        op_name="_lu_with_info",
        torch_op=torch._lu_with_info,
        dtypes=_TEST_DTYPES,
    ).run()
