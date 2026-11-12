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

from . import base, consts, utils
from .generated_operator_utils import OperatorBenchmark


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"input": list(dims)},
        params={"dim": [-1], "keepdim": False, "mode": 0},
        builder_args=(dims,),
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    torch.manual_seed(0)
    inp = utils.generate_tensor_input(shape, dtype, device)
    return inp, [-1], {"keepdim": False, "mode": 0}


class HashTensorBenchmark(base.GenericBenchmark2DOnly, OperatorBenchmark):
    """Keep the reduction shape policy while exposing replayable cases."""

    def supports_cases(self):
        return True

    def set_shapes(self, shape_file_path=None):
        base.Benchmark.set_shapes(self, shape_file_path)

    get_case_iter = OperatorBenchmark.get_case_iter
    def _run_cases(self, case_ids):
        skipped_before = set(base.Config.skipped_case_ids)
        results = super()._run_cases(case_ids)
        if not results and base.Config.skipped_case_ids - skipped_before:
            pytest.skip("Iluvatar ATen hash_tensor returns uint32; benchmark requires uint64")
        return results

    def build_inputs(self, case):
        if flag_gems.vendor_name == "iluvatar":
            probe = torch.ones((1, 2), dtype=case.dtype, device=self.device)
            result = self.torch_op(probe, [-1], keepdim=False, mode=0)
            if result.dtype == torch.uint32:
                base.Config.skipped_case_ids.add(case.case_id)
                pytest.skip(
                    "Iluvatar ATen hash_tensor returns uint32; benchmark requires uint64"
                )
        return OperatorBenchmark.build_inputs(self, case)


@pytest.mark.hash_tensor
def test_hash_tensor():
    bench = HashTensorBenchmark(
        op_name="hash_tensor",
        torch_op=torch.ops.aten.hash_tensor,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
