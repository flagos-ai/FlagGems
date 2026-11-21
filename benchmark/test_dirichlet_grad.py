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
from .generated_operator_utils import OperatorBenchmark


_DTYPES = [torch.float32, torch.float64]
_SHAPES = [(8,), (1024, 8), (4096, 32), (64, 1024, 16)]


class DirichletGradBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(_SHAPES)


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"x": list(dims), "alpha": list(dims), "total": list(dims)},
        params={"domain": "simplex and positive concentration"},
        builder_args=(dims,),
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    torch.manual_seed(0)
    alpha = torch.rand(shape, dtype=dtype, device=device) * 9.5 + 0.5
    total = alpha.sum(-1, keepdim=True).expand_as(alpha).contiguous()
    gamma_shape = alpha if dtype == torch.float64 else alpha.float()
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float64:
        gamma_shape = gamma_shape.cpu()
    samples = torch.distributions.Gamma(
        gamma_shape, torch.ones_like(gamma_shape)
    ).sample().to(device=alpha.device, dtype=dtype)
    x = samples / samples.sum(-1, keepdim=True)
    return x, alpha, total


@pytest.mark.dirichlet_grad
def test_dirichlet_grad():
    bench = DirichletGradBenchmark(
        op_name="dirichlet_grad",
        torch_op=torch.ops.aten._dirichlet_grad,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=_DTYPES,
    )
    bench.run()
