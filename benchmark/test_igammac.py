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
import pytest
import torch

from . import base


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
    bench = base.GenericBenchmark(
        op_name="igammac",
        torch_op=torch.special.gammaincc,
        input_fn=_igammac_input,
        dtypes=[torch.float32],
    )
    bench.run()


@pytest.mark.igammac_out
def test_igammac_out():
    bench = base.GenericBenchmark(
        op_name="igammac_out",
        input_fn=_igammac_input_out,
        torch_op=torch.ops.aten.special_gammaincc.out,
        dtypes=[torch.float32],
    )
    bench.run()
