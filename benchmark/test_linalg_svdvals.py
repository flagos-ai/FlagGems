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

import os

import pytest
import torch

import flag_gems

from . import base
from .generated_operator_utils import OperatorBenchmark
from .test_linalg_cond import _MusaLinalgStaticBaseline

SHAPES = [
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
    (256, 256),
    (512, 512),
]
MASTER_EXTRA_SHAPES = [
    (1024, 1024),
    (4096, 4096),
    (1024, 65536),
]


class OperatorCaseBenchmark(_MusaLinalgStaticBaseline, OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = SHAPES + MASTER_EXTRA_SHAPES
        self.shape_desc = "(*B), M, N"

    def build_inputs(self, case):
        if (
            flag_gems.vendor_name == "mthreads"
            and not (base.Config.preflight_only or base.Config.profile_only)
            and case.builder_args[0].builder_args[0] in MASTER_EXTRA_SHAPES
        ):
            base.Config.skipped_case_ids.add(case.case_id)
            pytest.skip("The historical MUSA SVD values baseline does not cover this shape")
        return super().build_inputs(case)


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"input": list(dims)},
        params={},
        builder_args=(dims,),
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=device)
    return (inp,)


@pytest.mark.linalg_svdvals
def test_linalg_svdvals():
    if (
        flag_gems.vendor_name == "mthreads"
        and not (
            base.Config.query
            or base.Config.list_cases
            or base.Config.preflight_only
            or base.Config.profile_only
        )
        and not os.environ.get("FLAGGEMS_MUSA_LINALG_H800_BASELINE")
    ):
        pytest.skip("MUSA native SVD values require unavailable LAPACK support")
    OperatorCaseBenchmark(
        op_name="linalg_svdvals",
        torch_op=torch.linalg.svdvals,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.float32],
    ).run()
