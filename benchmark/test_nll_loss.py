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

from . import base, consts, utils


def nll_loss_input_fn(shape, cur_dtype, device):
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    target_shape = list(shape)
    del target_shape[1]
    target = torch.randint(0, shape[-1], target_shape, device=device)
    yield inp, target

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        weight = torch.randn(shape[1], dtype=cur_dtype, device=device)
        yield inp, target, {"weight": weight, "ignore_index": 1, "reduction": "none"}


@pytest.mark.nll_loss_forward
def test_nll_loss_forward():
    bench = base.GenericBenchmark2DOnly(
        op_name="nll_loss_forward",
        input_fn=nll_loss_input_fn,
        torch_op=torch.nn.functional.nll_loss,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.nll_loss_backward
def test_nll_loss_backward():
    bench = base.GenericBenchmark2DOnly(
        op_name="nll_loss_backward",
        input_fn=nll_loss_input_fn,
        torch_op=torch.nn.functional.nll_loss,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=True,
    )
    bench.run()
