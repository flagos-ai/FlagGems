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

from . import base
from .generated_operator_utils import OperatorBenchmark

SHAPES = [(256, 256), (1024, 256), (2048, 512), (65536,)]


class FusedAdagradBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(SHAPES)
        self.shape_desc = "optimizer parameter shape"


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    for mode, lr_decay, weight_decay, maximize in (
        ("plain", 0.0, 0.0, False),
        ("options", 0.2, 0.01, True),
    ):
        yield base.BenchmarkCasePlan(
            shape={"param": list(dims)},
            params={
                "mode": mode,
                "lr_decay": lr_decay,
                "weight_decay": weight_decay,
                "maximize": maximize,
            },
            builder_args=(dims, lr_decay, weight_decay, maximize),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, lr_decay, weight_decay, maximize = plan.builder_args
    torch.manual_seed(0)
    params = [torch.randn(shape, dtype=dtype, device=device)]
    grads = [torch.randn(shape, dtype=dtype, device=device)]
    state_sums = [torch.rand(shape, dtype=dtype, device=device) + 0.25]
    state_steps = [torch.tensor([3.0], dtype=torch.float32, device=device)]
    kwargs = {
        "lr": 0.01,
        "lr_decay": lr_decay,
        "weight_decay": weight_decay,
        "eps": 1e-10,
        "maximize": maximize,
        "grad_scale": None,
        "found_inf": None,
    }
    return params, grads, state_sums, state_steps, kwargs


@pytest.mark.fused_adagrad_
@pytest.mark.underscore_fused_adagrad_
def test_fused_adagrad_():
    FusedAdagradBenchmark(
        op_name="_fused_adagrad_",
        torch_op=torch._fused_adagrad_,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.float32],
        fresh_inputs=True,
    ).run()
