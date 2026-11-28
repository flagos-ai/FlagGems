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

import flag_gems

from . import base, consts


class AddmvBenchmark(base.GenericBenchmark2DOnly):
    def set_more_shapes(self):
        return []

    def get_input_iter(self, dtype) -> Generator:
        for m, n in self.shapes:
            yield from self.input_fn(m, n, dtype, self.device)


def _input_fn(m, n, cur_dtype, device):
    mat = torch.randn([m, n], dtype=cur_dtype, device=device)
    vec = torch.randn([n], dtype=cur_dtype, device=device)
    bias = torch.randn([m], dtype=cur_dtype, device=device)

    # torch.addmv(bias, mat, vec)
    yield bias, mat, vec


def _input_fn_out(m, n, cur_dtype, device):
    mat = torch.randn([m, n], dtype=cur_dtype, device=device)
    vec = torch.randn([n], dtype=cur_dtype, device=device)
    bias = torch.randn([m], dtype=cur_dtype, device=device)
    out = torch.empty([m], dtype=cur_dtype, device=device)
    yield bias, mat, vec, {"out": out}


@pytest.mark.addmv
def test_addmv():
    bench = AddmvBenchmark(
        op_name="addmv",
        input_fn=_input_fn,
        torch_op=torch.addmv,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.addmv_out
def test_addmv_out():
    bench = AddmvBenchmark(
        op_name="addmv_out",
        input_fn=_input_fn_out,
        torch_op=torch.addmv,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def _torch_addmv_backward(grad_output, mat, vec, beta=1.0, alpha=1.0):
    return (
        grad_output * beta,
        grad_output[:, None] * vec[None, :] * alpha,
        torch.mv(mat.t(), grad_output) * alpha,
    )


def _backward_input_fn(m, n, cur_dtype, device):
    grad_output = torch.randn([m], dtype=cur_dtype, device=device)
    mat = torch.randn([m, n], dtype=cur_dtype, device=device)
    vec = torch.randn([n], dtype=cur_dtype, device=device)
    yield grad_output, mat, vec, {"alpha": 0.75, "beta": -0.5}


@pytest.mark.addmv_backward
def test_addmv_backward():
    bench = AddmvBenchmark(
        op_name="addmv_backward",
        input_fn=_backward_input_fn,
        torch_op=_torch_addmv_backward,
        gems_op=flag_gems.addmv_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
