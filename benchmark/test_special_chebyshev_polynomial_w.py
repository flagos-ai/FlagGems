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


class ChebyshevPolynomialWBenchmark(base.Benchmark):
    """Custom benchmark for special_chebyshev_polynomial_w (binary op: x, n)."""

    def set_shapes(self, shape_file_path=None):
        # SPECIAL_SHAPES equivalent; special.* ops use modest subset of shapes
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

    def get_input_iter(self, cur_dtype) -> Generator:
        # Constant small degree for benchmark stability
        n_val = 3
        for shape in self.shapes:
            x = base.generate_tensor_input(shape, cur_dtype, self.device)
            # x in [-1, 1] domain
            x = x * 2 - 1
            n = torch.tensor(n_val, dtype=torch.int64, device=self.device)
            yield x, n

    def get_tflops(self, op, *args, **kwargs):
        shape = list(args[0].shape)
        return torch.tensor(shape).prod().item()


class ChebyshevPolynomialWOutBenchmark(ChebyshevPolynomialWBenchmark):
    """Benchmark for special_chebyshev_polynomial_w_out."""

    def get_input_iter(self, cur_dtype) -> Generator:
        n_val = 3
        for shape in self.shapes:
            x = base.generate_tensor_input(shape, cur_dtype, self.device)
            x = x * 2 - 1
            n = torch.tensor(n_val, dtype=torch.int64, device=self.device)
            out = torch.empty_like(x)
            yield x, n, {"out": out}


@pytest.mark.special_chebyshev_polynomial_w
def test_special_chebyshev_polynomial_w():
    bench = ChebyshevPolynomialWBenchmark(
        op_name="special_chebyshev_polynomial_w",
        torch_op=torch.special.chebyshev_polynomial_w,
        # special.* operators only support float32
        dtypes=[torch.float32],
    )
    bench.run()


@pytest.mark.special_chebyshev_polynomial_w_out
def test_special_chebyshev_polynomial_w_out():
    bench = ChebyshevPolynomialWOutBenchmark(
        op_name="special_chebyshev_polynomial_w_out",
        torch_op=torch.special.chebyshev_polynomial_w,
        dtypes=[torch.float32],
    )
    bench.run()
