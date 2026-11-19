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


def nll_loss_case_fn(shape, cur_dtype):
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={"variant": "default"},
        builder_args=(shape, "default"),
    )

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        yield base.BenchmarkCasePlan(
            shape={"input": list(shape)},
            params={"variant": "weighted"},
            builder_args=(shape, "weighted"),
        )


def nll_loss_build_inputs_fn(plan, cur_dtype, device):
    shape, variant = plan.builder_args
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    target_shape = list(shape)
    del target_shape[1]
    target = torch.randint(0, shape[-1], target_shape, device=device)
    if variant == "default":
        return inp, target
    weight = torch.randn(shape[1], dtype=cur_dtype, device=device)
    return inp, target, {"weight": weight, "ignore_index": 1, "reduction": "none"}


@pytest.mark.nll_loss2d_forward
def test_nll_loss2d_forward():
    bench = NllLoss2dBenchmark(
        case_fn=nll_loss_case_fn,
        build_inputs_fn=nll_loss_build_inputs_fn,
        op_name="nll_loss2d_forward",
        torch_op=torch.nn.functional.nll_loss,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def nll_loss_backward_build_inputs_fn(plan, dtype, device):
    shape, variant = plan.builder_args
    inp = utils.generate_tensor_input(shape, dtype, device)
    target_shape = (shape[0], *shape[2:])
    target = torch.randint(0, shape[1], target_shape, device=device)
    if variant == "weighted":
        weight = torch.randn(shape[1], dtype=dtype, device=device)
        valid_target = target[target != 1]
        total_weight = weight[valid_target].sum()
        grad = torch.randn(target_shape, dtype=dtype, device=device)
        return grad, inp, target, weight, 0, 1, total_weight
    grad = torch.randn((), dtype=dtype, device=device)
    # Construct the count in a wide dtype before converting, as the forward
    # reduction does. Direct half construction rejects counts above 65504.
    total_weight = torch.full(
        (), math.prod(target_shape), dtype=torch.float32, device=device
    ).to(dtype)
    return grad, inp, target, None, 1, -100, total_weight


@pytest.mark.nll_loss2d_backward
def test_nll_loss2d_backward():
    bench = NllLoss2dBenchmark(
        case_fn=nll_loss_case_fn,
        build_inputs_fn=nll_loss_backward_build_inputs_fn,
        op_name="nll_loss2d_backward",
        torch_op=torch.ops.aten.nll_loss2d_backward.default,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
