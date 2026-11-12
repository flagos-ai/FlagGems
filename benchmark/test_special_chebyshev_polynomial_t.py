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


class ChebyshevPolynomialTBenchmark(base.GenericBenchmark):
    """Custom benchmark for special_chebyshev_polynomial_t (binary op: x, n)."""

    def set_shapes(self, shape_file_path=None):
        # SPECIAL_SHAPES equivalent; special.* ops use a modest subset of shapes
        self.shapes = [
            (1024, 1024),
            (20, 320, 15),
            (16, 128, 64, 1280),
            (2, 19, 7),
        ]

    def set_more_shapes(self):
        return [(1024, 2**i) for i in range(0, 20, 4)] + [
            (64, 64, 2**i) for i in range(0, 20, 4)
        ]


    def get_tflops(self, op, *args, **kwargs):
        shape = list(args[0].shape)
        return torch.tensor(shape).prod().item()


class ChebyshevPolynomialTOutBenchmark(ChebyshevPolynomialTBenchmark):
    """Benchmark for special_chebyshev_polynomial_t_out."""

    pass


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"x": list(dims), "n": []},
        params={"degree_kind": "tensor", "degree": 3},
        builder_args=(dims,),
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    x = base.generate_tensor_input(shape, dtype, device) * 2 - 1
    degree = torch.tensor(3, dtype=torch.int64, device=device)
    return x, degree


def _build_out_inputs_fn(plan, dtype, device):
    x, degree = _build_inputs_fn(plan, dtype, device)
    return x, degree, {"out": torch.empty_like(x)}


@pytest.mark.special_chebyshev_polynomial_t
def test_special_chebyshev_polynomial_t():
    bench = ChebyshevPolynomialTBenchmark(
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        op_name="special_chebyshev_polynomial_t",
        torch_op=torch.special.chebyshev_polynomial_t,
        # special.* operators only support float32
        dtypes=[torch.float32],
    )
    bench.run()


@pytest.mark.special_chebyshev_polynomial_t_out
def test_special_chebyshev_polynomial_t_out():
    bench = ChebyshevPolynomialTOutBenchmark(
        case_fn=_case_fn,
        build_inputs_fn=_build_out_inputs_fn,
        op_name="special_chebyshev_polynomial_t_out",
        torch_op=torch.special.chebyshev_polynomial_t,
        dtypes=[torch.float32],
    )
    bench.run()
