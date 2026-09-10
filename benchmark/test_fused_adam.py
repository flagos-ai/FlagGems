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

from . import base

# Shapes representing realistic optimizer parameter sizes:
# small embedding / attention weights / MLP weights
_FUSED_ADAM_SHAPES = [
    (256, 256),
    (512, 512),
    (1024, 256),
    (2048, 512),
    (4096, 256),
    (65536,),
]


class FusedAdamBenchmark(base.GenericBenchmark):
    # fused_adam uses 6 tensors per case (param, grad, 3 state, step)
    # so shapes are kept moderate to avoid OOM on CI GPUs
    DEFAULT_SHAPES = _FUSED_ADAM_SHAPES

    def set_shapes(self, shape_file=None):
        self.shapes = list(_FUSED_ADAM_SHAPES)


def fused_adam_input_fn(shape, dtype, device):
    param = torch.randn(shape, dtype=dtype, device=device)
    grad = torch.randn(shape, dtype=dtype, device=device)
    exp_avg = torch.zeros(shape, dtype=dtype, device=device)
    exp_avg_sq = torch.zeros(shape, dtype=dtype, device=device)
    max_exp_avg_sq = torch.zeros(shape, dtype=dtype, device=device)
    state_step = torch.tensor([1], dtype=torch.long, device=device)
    yield param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq, state_step


def torch_op(param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq, state_step):
    # Reference: compute manually using Adam formula
    lr = 0.001
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    step = state_step.item()

    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step

    # Update first moment estimate
    exp_avg = beta1 * exp_avg + (1 - beta1) * grad
    # Update second moment estimate
    exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad * grad
    # Compute bias-corrected estimates
    corrected_exp_avg = exp_avg / bias_correction1
    corrected_exp_avg_sq = exp_avg_sq / bias_correction2
    # Update parameters
    param = param - lr * corrected_exp_avg / (torch.sqrt(corrected_exp_avg_sq) + eps)
    return param


@pytest.mark.fused_adam
def test_fused_adam():
    def gems_op(param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq, state_step):
        return flag_gems._fused_adam(
            [param],
            [grad],
            [exp_avg],
            [exp_avg_sq],
            [max_exp_avg_sq],
            [state_step],
            lr=0.001,
            beta1=0.9,
            beta2=0.999,
            weight_decay=0.0,
            eps=1e-8,
            amsgrad=False,
            maximize=False,
        )

    bench = FusedAdamBenchmark(
        input_fn=fused_adam_input_fn,
        op_name="fused_adam",
        torch_op=torch_op,
        # _fused_adam only supports float32 for optimizer state precision
        dtypes=[torch.float32],
    )
    bench.set_gems(gems_op)
    bench.run()


@pytest.mark.fused_adam_
def test_fused_adam_():
    def gems_op(param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq, state_step):
        flag_gems._fused_adam_(
            [param],
            [grad],
            [exp_avg],
            [exp_avg_sq],
            [max_exp_avg_sq],
            [state_step],
            lr=0.001,
            beta1=0.9,
            beta2=0.999,
            weight_decay=0.0,
            eps=1e-8,
            amsgrad=False,
            maximize=False,
        )
        return param

    bench = FusedAdamBenchmark(
        input_fn=fused_adam_input_fn,
        op_name="fused_adam_",
        torch_op=torch_op,
        # _fused_adam only supports float32 for optimizer state precision
        dtypes=[torch.float32],
    )
    bench.set_gems(gems_op)
    bench.run()
