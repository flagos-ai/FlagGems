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

from . import base

# `_dirichlet_grad` is only implemented for the AT_DISPATCH_FLOATING_TYPES path
# (float32 / float64) in the ATen CUDA dispatcher, so the benchmark only
# covers those two dtypes.
# Only float32/float64 are supported by the ATen CUDA dispatcher; matching that.
DIRICHLET_GRAD_BENCH_DTYPES = [torch.float32, torch.float64]

# Simplex-shaped tensors: the last dimension is the number of categories.
# These shapes keep the benchmark fast while spanning 1-D, 2-D and 3-D cases
# with category counts that exercise the small / large / mid / rational
# sub-branches of the kernel.
DIRICHLET_GRAD_BENCH_SHAPES = [
    (8,),
    (1024, 8),
    (4096, 32),
    (64, 1024, 16),
]


def _dirichlet_input_fn(shape, dtype, device):
    """Build a coherent (x, alpha, total) triple for `_dirichlet_grad`.

    ``alpha`` is uniform in ``[0.5, 10]``; ``x`` is a Gamma(shape=alpha)-draw
    normalized to the simplex; ``total`` is the concentration sum broadcast
    back to ``alpha``'s shape -- exactly the signature the ATen op expects.
    """
    alpha = torch.rand(*shape, dtype=dtype, device=device) * 9.5 + 0.5
    total = alpha.sum(-1, keepdim=True).expand_as(alpha).contiguous()
    gamma_shape = alpha if dtype == torch.float64 else alpha.float()
    gamma = torch.distributions.Gamma(gamma_shape, torch.ones_like(gamma_shape))
    x = gamma.sample().to(dtype)
    x = x / x.sum(-1, keepdim=True)
    yield x, alpha, total


class DirichletGradBenchmark(base.GenericBenchmark):
    """`GenericBenchmark` restricted to the simplex shapes above.

    The base class pulls shapes from ``DEFAULT_SHAPES`` (which include a 1B
    element 1-D tensor and a 1024^3 3-D tensor); for a Dirichlet gradient
    those are both degenerate (a 1-category simplex) and far too large, so
    we pin the shape list to a handful of realistic simplex tensors.
    """

    def set_shapes(self, shape_file_path=None):
        self.shapes = list(DIRICHLET_GRAD_BENCH_SHAPES)


@pytest.mark.dirichlet_grad
def test_dirichlet_grad():
    bench = DirichletGradBenchmark(
        op_name="dirichlet_grad",
        input_fn=_dirichlet_input_fn,
        torch_op=torch.ops.aten._dirichlet_grad,
        dtypes=DIRICHLET_GRAD_BENCH_DTYPES,
    )
    bench.run()
