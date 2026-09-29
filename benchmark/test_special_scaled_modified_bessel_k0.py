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

"""Benchmark for aten::special_scaled_modified_bessel_k0.

torch_op is the ATen reference (the perf baseline) and gems_op is the FlagGems
candidate; both are called as op(input), so candidate and reference share the
same call semantics. Case listing is metadata-only, and the same case plans,
dtype list and input builders serve --list-cases, --case-id replay and normal
execution.

core_shapes.yaml has no entry for this operator and the base-class fallback is
the default shape list, which contains a 1e9-element tensor. The shapes below
instead follow the core reference level of comparable unary pointwise operators
([4096, 4096], [64, 512, 512], [1024, 65536]) and add the spec's higher-rank
sizes plus two small shapes that measure the launch-bound regime.
OperatorBenchmark keeps --shape_file precedence: an explicit entry for this
operator in a shape file still overrides these defaults.
"""

import pytest
import torch

import flag_gems

from . import base, utils
from .generated_operator_utils import OperatorBenchmark

_SHAPES = [
    (1024,),
    (65536,),
    (64, 512, 512),
    (1024, 1024),
    (4096, 4096),
    (1024, 65536),
    (20, 320, 15),
    (16, 128, 64, 60),
    (16, 7, 57, 32, 29),
]

# The native operator computes float32 and rejects the half/bf16 types, and
# float64 is a static backend capability; one list serves listing and execution.
_DTYPES = [torch.float32]
if flag_gems.runtime.device.support_fp64:
    _DTYPES.append(torch.float64)


def _case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    if dtype == torch.float64:
        # benchmark.utils.generate_tensor_input has no float64 branch and
        # returns None for it, so the same randn input is built directly.
        return torch.randn(shape, dtype=dtype, device=device), {}
    return utils.generate_tensor_input(shape, dtype, device), {}


class SpecialScaledModifiedBesselK0Benchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        # With default_shapes this assigns self.shapes directly and never calls
        # base.set_shapes, so no set_more_shapes merge applies: the core and
        # comprehensive levels list exactly the same cases.
        super().set_shapes(shape_file_path, default_shapes=_SHAPES)


@pytest.mark.special_scaled_modified_bessel_k0
def test_special_scaled_modified_bessel_k0():
    bench = SpecialScaledModifiedBesselK0Benchmark(
        op_name="special_scaled_modified_bessel_k0",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.special_scaled_modified_bessel_k0,
        gems_op=getattr(flag_gems, "special_scaled_modified_bessel_k0", None),
        dtypes=_DTYPES,
    )
    bench.run()
