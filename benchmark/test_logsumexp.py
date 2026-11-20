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

from . import base, consts, utils


def _input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, 1


@pytest.mark.logsumexp
def test_logsumexp():
    bench = base.GenericBenchmarkExcluse1D(
        input_fn=_input_fn,
        op_name="logsumexp",
        torch_op=torch.logsumexp,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.logsumexp_backward
def test_logsumexp_backward():
    # ``is_backward`` makes the timed callable
    # ``torch.autograd.grad(out, (inp,), grad_outputs=(dout,))`` on a graph built
    # once from the same ``(inp, dim=1)`` inputs, so the forward is not timed and
    # only the backward kernel is measured for both sides.
    bench = base.GenericBenchmarkExcluse1D(
        input_fn=_input_fn,
        op_name="logsumexp",
        torch_op=torch.logsumexp,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=True,
    )
    bench.set_gems(flag_gems.logsumexp)
    bench.run()
