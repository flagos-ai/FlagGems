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

from . import base, consts


def _input_fn(b, m, n, k, dtype, device, b_column_major):
    inp1 = torch.randn([b, m, k], dtype=dtype, device=device)
    if b_column_major:
        inp2 = torch.randn([b, n, k], dtype=dtype, device=device)
        yield inp1, inp2.transpose(1, 2)
    else:
        inp2 = torch.randn([b, k, n], dtype=dtype, device=device)
        yield inp1, inp2


@pytest.mark.bmm
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_bmm(monkeypatch):
    bench = base.BlasBenchmark(
        op_name="bmm",
        input_fn=_input_fn,
        torch_op=torch.bmm,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()


def _input_fn_out(b, m, n, k, dtype, device, b_column_major):
    inp1 = torch.randn([b, m, k], dtype=dtype, device=device)
    if b_column_major:
        inp2 = torch.randn([b, n, k], dtype=dtype, device=device)
        inp2 = inp2.transpose(1, 2)
    else:
        inp2 = torch.randn([b, k, n], dtype=dtype, device=device)
    out = torch.empty([b, m, n], dtype=dtype, device=device)
    yield inp1, inp2, {"out": out}


@pytest.mark.bmm_out
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_bmm_out(monkeypatch):
    bench = base.BlasBenchmark(
        op_name="bmm_out",
        input_fn=_input_fn_out,
        torch_op=torch.bmm,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
