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


@pytest.mark.acosh
def test_acosh():
    bench = base.UnaryPointwiseBenchmark(
        op_name="acosh",
        torch_op=torch.acosh,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.acosh_
def test_acosh_():
    bench = base.UnaryPointwiseBenchmark(
        op_name="acosh_",
        torch_op=torch.acosh_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.acosh_backward
def test_acosh_backward():
    bench = base.UnaryPointwiseBenchmark(
        op_name="acosh",
        torch_op=torch.acosh,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=True,
    )
    # The Autograd fallthrough routes the default use_gems() path through
    # AcoshFunction, so this measures the fused registered backward kernel.
    bench.run()


def _acosh_backward_inplace_input_fn(shape, dtype, device):
    # Keep inputs inside acosh's real domain for a finite backward reference.
    yield torch.rand(shape, dtype=dtype, device=device) + 1.1,


@pytest.mark.acosh_backward_
def test_acosh_backward_():
    bench = base.GenericBenchmark(
        op_name="acosh_backward_",
        input_fn=_acosh_backward_inplace_input_fn,
        # PyTorch marks acosh_ backward as not implemented; out-of-place acosh
        # has the same mathematical gradient and is the native baseline here.
        torch_op=torch.acosh,
        gems_op=lambda x: flag_gems.acosh_(x.clone()),
        dtypes=consts.FLOAT_DTYPES,
        is_backward=True,
    )
    bench.run()
