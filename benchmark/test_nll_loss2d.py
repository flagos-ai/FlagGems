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

import math

import pytest
import torch

from . import base, consts, utils

# nll_loss backward needs input + grad_output + target; cap to avoid OOM.
MAX_ELEMENTS = 2**29


class NllLoss2dBenchmark(base.GenericBenchmark4DOnly):
    def init_user_config(self):
        super().init_user_config()
        self.shapes = [s for s in self.shapes if math.prod(s) <= MAX_ELEMENTS]


def nll_loss_input_fn(shape, cur_dtype, device):
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    target_shape = list(shape)
    del target_shape[1]
    target = torch.randint(0, shape[-1], target_shape, device=device)
    yield inp, target

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        weight = torch.randn(shape[1], dtype=cur_dtype, device=device)
        yield inp, target, {"weight": weight, "ignore_index": 1, "reduction": "none"}


@pytest.mark.nll_loss2d_forward
def test_nll_loss2d_forward():
    bench = NllLoss2dBenchmark(
        input_fn=nll_loss_input_fn,
        op_name="nll_loss2d_forward",
        torch_op=torch.nn.functional.nll_loss,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.nll_loss2d_backward
def test_nll_loss2d_backward():
    bench = NllLoss2dBenchmark(
        input_fn=nll_loss_input_fn,
        op_name="nll_loss2d_backward",
        torch_op=torch.nn.functional.nll_loss,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=True,
    )
    bench.run()
