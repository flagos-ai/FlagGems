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


def _input_fn(shape, dtype, device):
    inp1 = torch.randn(shape, dtype=dtype, device=device)
    inp2 = torch.randn(shape, dtype=dtype, device=device)
    target = (torch.randint(0, 2, shape, device=device, dtype=torch.int8) * 2 - 1).to(
        dtype
    )
    yield inp1, inp2, target, 0.5, 1


def _backward_input_fn(shape, dtype, device):
    inp1 = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    inp2 = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    target = (torch.randint(0, 2, shape, device=device, dtype=torch.int8) * 2 - 1).to(
        dtype
    )
    yield inp1, inp2, target, 0.5, 1


@pytest.mark.margin_ranking_loss
def test_margin_ranking_loss():
    bench = base.MarginRankingLossBenchmark(
        op_name="margin_ranking_loss",
        input_fn=_input_fn,
        torch_op=torch.ops.aten.margin_ranking_loss,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.margin_ranking_loss_backward
def test_margin_ranking_loss_backward():
    bench = base.MarginRankingLossBenchmark(
        op_name="margin_ranking_loss",
        input_fn=_backward_input_fn,
        torch_op=torch.ops.aten.margin_ranking_loss,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=True,
    )
    bench.run()
