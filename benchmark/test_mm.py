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


def mm_input_fn(b, m, n, k, cur_dtype, device, b_column_major):
    inp1 = torch.randn([m, k], dtype=cur_dtype, device=device)
    if b_column_major:
        inp2 = torch.randn([n, k], dtype=cur_dtype, device=device)
        yield inp1, inp2.t()
    else:
        inp2 = torch.randn([k, n], dtype=cur_dtype, device=device)
        yield inp1, inp2


@pytest.mark.mm
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_mm():
    bench = base.BlasBenchmark(
        op_name="mm",
        input_fn=mm_input_fn,
        torch_op=torch.Tensor.mm,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()


class MmSelfTransposeBenchmark(base.GenericBenchmark2DOnly):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["tflops"]

    def set_more_shapes(self):
        return []

    def get_tflops(self, op, *args, **kwargs):
        m, k = args[0].shape
        return 2 * m * m * k


def _input_fn(shape, cur_dtype, device):
    m, k = shape
    inp = torch.randn([k, m], dtype=cur_dtype, device=device).t()

    yield inp,


def torch_mm_self_transpose(inp):
    return torch.mm(inp, inp.t())


@pytest.mark.mm
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_mm_self_transpose_benchmark():
    bench = MmSelfTransposeBenchmark(
        op_name="mm_self_transpose",
        input_fn=_input_fn,
        torch_op=torch_mm_self_transpose,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()


def mm_out_input_fn(b, m, n, k, cur_dtype, device, b_column_major):
    inp1 = torch.randn([m, k], dtype=cur_dtype, device=device)
    if b_column_major:
        inp2 = torch.randn([n, k], dtype=cur_dtype, device=device)
        out = torch.empty([m, n], dtype=cur_dtype, device=device)
        yield inp1, inp2.t(), {"out": out}
    else:
        inp2 = torch.randn([k, n], dtype=cur_dtype, device=device)
        out = torch.empty([m, n], dtype=cur_dtype, device=device)
        yield inp1, inp2, {"out": out}


@pytest.mark.mm_out
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_mm_out():
    bench = base.BlasBenchmark(
        op_name="mm_out",
        input_fn=mm_out_input_fn,
        torch_op=torch.mm,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
