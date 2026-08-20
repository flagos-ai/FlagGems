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
_IGAMMAC_DTYPES = [
    torch.float32,
]
if flag_gems.runtime.device.support_fp64:
    _IGAMMAC_DTYPES.append(torch.float64)


class IgammacBenchmark(base.GenericBenchmark):
    """GenericBenchmark with domain-valid inputs.

    Float64 is benchmarked on a representative subset of shapes (up to
    MAX_FLOAT64_ELEMENTS elements) to keep the runtime bounded: torch's
    float64 igammac baseline is very slow on large tensors.
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


def _igammac_input(shape, dtype, device):
    # igammac(a, x) is only defined for a > 0 and x >= 0; the default randn
    # generator would push torch's reference kernel into a non-converging path.
    a = torch.rand(shape, dtype=dtype, device=device) * 10 + 0.1
    x = torch.rand(shape, dtype=dtype, device=device) * 10 + 0.1
    yield a, x


def _igammac_input_out(shape, dtype, device):
    a = torch.rand(shape, dtype=dtype, device=device) * 10 + 0.1
    x = torch.rand(shape, dtype=dtype, device=device) * 10 + 0.1
    out = torch.empty_like(a)
    yield a, x, {"out": out}


@pytest.mark.igammac
def test_igammac():
    bench = IgammacBenchmark(
        op_name="igammac",
        torch_op=torch.special.gammaincc,
        gems_op=flag_gems.igammac,
        input_fn=_igammac_input,
        dtypes=_IGAMMAC_DTYPES,
    )
    bench.run()


@pytest.mark.igammac_out
def test_igammac_out():
    bench = IgammacBenchmark(
        op_name="igammac_out",
        input_fn=_igammac_input_out,
        torch_op=torch.ops.aten.special_gammaincc.out,
        gems_op=flag_gems.igammac_out,
        dtypes=_IGAMMAC_DTYPES,
    )
    bench.run()
