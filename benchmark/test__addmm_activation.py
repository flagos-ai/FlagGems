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
from _pytest.mark.structures import Mark, MarkDecorator

from . import base, consts

# ``_addmm_activation`` starts with an underscore, and ``pytest.mark`` refuses to
# generate a marker via attribute access for such names. Register the markers
# directly on the MarkGenerator so ``@pytest.mark._addmm_activation`` and
# ``-m _addmm_activation`` both work.
for _mark_name in ("_addmm_activation", "_addmm_activation_out"):
    setattr(
        pytest.mark,
        _mark_name,
        MarkDecorator(Mark(_mark_name, (), {}, _ispytest=True), _ispytest=True),
    )


def _input_fn(b, m, n, k, dtype, device, b_column_major):
    inp1 = torch.randn([m, k], dtype=dtype, device=device)
    bias = torch.randn([m, n], dtype=dtype, device=device)
    if b_column_major:
        inp2 = torch.randn([n, k], dtype=dtype, device=device)
        yield bias, inp1, inp2.t(),
    else:
        inp2 = torch.randn([k, n], dtype=dtype, device=device)
        yield bias, inp1, inp2,


@pytest.mark._addmm_activation
def test__addmm_activation(monkeypatch):
    bench = base.BlasBenchmark(
        op_name="_addmm_activation",
        input_fn=_input_fn,
        torch_op=torch._addmm_activation,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()


def _input_fn_out(b, m, n, k, dtype, device, b_column_major):
    inp1 = torch.randn([m, k], dtype=dtype, device=device)
    bias = torch.randn([m, n], dtype=dtype, device=device)
    out = torch.empty([m, n], dtype=dtype, device=device)
    if b_column_major:
        inp2 = torch.randn([n, k], dtype=dtype, device=device)
        yield bias, inp1, inp2.t(), {"out": out}
    else:
        inp2 = torch.randn([k, n], dtype=dtype, device=device)
        yield bias, inp1, inp2, {"out": out}


@pytest.mark._addmm_activation_out
def test__addmm_activation_out(monkeypatch):
    bench = base.BlasBenchmark(
        op_name="_addmm_activation_out",
        input_fn=_input_fn_out,
        torch_op=torch._addmm_activation,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
