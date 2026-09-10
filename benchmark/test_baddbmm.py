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


class BaddbmmBenchmark(base.BlasBenchmark):
    def set_more_shapes(self):
        model_shapes_list = consts.model_shapes()

        skip_shapes = [
            (4, 8192, 128256, 4096),
            (4, 8192, 152064, 3584),
        ]

        filtered = []
        for shape in model_shapes_list:
            if shape not in skip_shapes:
                filtered.append(shape)

        return filtered

    def get_tflops(self, op, *args, **kwargs):
        # shape(b,m,k)(b,k,n)
        # total_flops = b * m * n * (2 * k + 1)
        total_flops = (
            args[1].shape[0]
            * args[1].shape[1]
            * args[2].shape[2]
            * (args[1].shape[2] * 2 + 1)
        )
        return total_flops


def _input_fn(b, m, n, k, dtype, device, b_column_major):
    inp1 = torch.randn([b, m, k], dtype=dtype, device=device, requires_grad=True)

    if b_column_major:
        inp2 = torch.randn([b, n, k], dtype=dtype, device=device, requires_grad=True)
        inp2 = inp2.transpose(1, 2).contiguous()
    else:
        inp2 = torch.randn([b, k, n], dtype=dtype, device=device, requires_grad=True)

    bias = torch.randn([b, m, n], dtype=dtype, device=device, requires_grad=True)

    yield bias, inp1, inp2


@pytest.mark.baddbmm
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_baddbmm():
    bench = BaddbmmBenchmark(
        op_name="baddbmm",
        input_fn=_input_fn,
        torch_op=torch.baddbmm,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()


def _input_fn_out(b, m, n, k, dtype, device, b_column_major):
    inp1 = torch.randn([b, m, k], dtype=dtype, device=device)

    if b_column_major:
        inp2 = torch.randn([b, n, k], dtype=dtype, device=device)
        inp2 = inp2.transpose(1, 2).contiguous()
    else:
        inp2 = torch.randn([b, k, n], dtype=dtype, device=device)

    bias = torch.randn([b, m, n], dtype=dtype, device=device)
    out = torch.empty([b, m, n], dtype=dtype, device=device)

    yield bias, inp1, inp2, {"out": out}


@pytest.mark.baddbmm_out
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_baddbmm_out():
    bench = BaddbmmBenchmark(
        op_name="baddbmm_out",
        input_fn=_input_fn_out,
        torch_op=torch.ops.aten.baddbmm.out,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
