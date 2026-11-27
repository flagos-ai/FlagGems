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

from . import base, consts


def _input_fn(b, m, n, k, dtype, device, b_column_major):
    inp1 = torch.randn([b, m, k], dtype=dtype, device=device, requires_grad=True)

    if b_column_major:
        inp2 = torch.randn([b, n, k], dtype=dtype, device=device, requires_grad=True)
        inp2 = inp2.transpose(1, 2).contiguous()
    else:
        inp2 = torch.randn([b, k, n], dtype=dtype, device=device, requires_grad=True)

    # addbmm has bias shape [m, n] (not batched like baddbmm)
    bias = torch.randn([m, n], dtype=dtype, device=device, requires_grad=True)

    yield bias, inp1, inp2


@pytest.mark.addbmm
@pytest.mark.skip_native(
    vendors=["enflame"],
    reason=(
        "torch_gcu has no PrivateUse1 kernel for aten::addbmm; the CPU fallback "
        "is about four orders of magnitude slower than the device path, so the "
        "native baseline cannot be timed"
    ),
)
def test_addbmm():
    bench = base.BlasBenchmark(
        op_name="addbmm",
        input_fn=_input_fn,
        torch_op=torch.addbmm,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def _backward_input_fn(b, m, n, k, dtype, device, b_column_major):
    for bias, inp1, inp2 in _input_fn(b, m, n, k, dtype, device, b_column_major):
        # Non-unit scales ensure the reference executes the same three scaled
        # gradients that the fused candidate produces.
        yield bias, inp1, inp2, {"beta": 0.75, "alpha": -0.5}


@pytest.mark.addbmm_backward
def test_addbmm_backward():
    bench = base.BlasBenchmark(
        op_name="addbmm",
        input_fn=_backward_input_fn,
        torch_op=torch.addbmm,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=True,
    )
    # Autograd fallthrough routes the registered torch.addbmm call through the
    # fused AddbmmFunction backward used by applications.
    bench.run()
