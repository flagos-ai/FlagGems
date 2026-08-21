# Copyright 2026, The FlagOS Contributors.
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
import math

import pytest
import torch

import flag_gems

from . import base

# fp64 is not supported on every platform (e.g. ascend, iluvatar).
_GAMMAINCC_DTYPES = [
    torch.float32,
]
if flag_gems.runtime.device.support_fp64:
    _GAMMAINCC_DTYPES.append(torch.float64)


class GammainccBenchmark(base.GenericBenchmark):
    """GenericBenchmark with domain-valid inputs.

    Float64 is benchmarked on a representative subset of shapes (up to
    MAX_FLOAT64_ELEMENTS elements) to keep the runtime bounded: torch's
    float64 baseline is very slow on large tensors.
    """

    MAX_FLOAT64_ELEMENTS = 2**24

    def get_input_iter(self, dtype):
        shapes = self.shapes
        if dtype == torch.float64:
            shapes = [
                shape
                for shape in shapes
                if math.prod(shape) <= self.MAX_FLOAT64_ELEMENTS
            ]
        for shape in shapes:
            yield from self.input_fn(shape, dtype, self.device)


def _gammaincc_input(shape, dtype, device):
    # Q(a, x) is only defined for a > 0 and x >= 0; the default randn generator
    # would push torch's reference kernel into a non-converging path.
    a = torch.rand(shape, dtype=dtype, device=device) * 10 + 0.1
    x = torch.rand(shape, dtype=dtype, device=device) * 10 + 0.1
    yield a, x


def _gammaincc_input_out(shape, dtype, device):
    a = torch.rand(shape, dtype=dtype, device=device) * 10 + 0.1
    x = torch.rand(shape, dtype=dtype, device=device) * 10 + 0.1
    out = torch.empty_like(a)
    yield a, x, {"out": out}


@pytest.mark.special_gammaincc
def test_special_gammaincc():
    bench = GammainccBenchmark(
        op_name="special_gammaincc",
        torch_op=torch.ops.aten.special_gammaincc,
        gems_op=flag_gems.special_gammaincc,
        input_fn=_gammaincc_input,
        dtypes=_GAMMAINCC_DTYPES,
    )
    bench.run()


@pytest.mark.special_gammaincc_out
def test_special_gammaincc_out():
    bench = GammainccBenchmark(
        op_name="special_gammaincc_out",
        torch_op=torch.ops.aten.special_gammaincc.out,
        gems_op=flag_gems.special_gammaincc_out,
        input_fn=_gammaincc_input_out,
        dtypes=_GAMMAINCC_DTYPES,
    )
    bench.run()
