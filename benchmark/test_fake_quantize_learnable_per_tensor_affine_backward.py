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
from .generated_operator_utils import OperatorBenchmark


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"grad": list(dims), "self": list(dims), "scale": [1], "zero_point": [1]},
        params={"quant_min": 0, "quant_max": 255, "grad_factor": 1.0},
        builder_args=(dims,),
    )


def _build_inputs_fn(plan, dtype, device):
    del dtype
    (shape,) = plan.builder_args
    torch.manual_seed(0)
    self_t = torch.randn(shape, dtype=torch.float32, device=device)
    grad = torch.randn(shape, dtype=torch.float32, device=device)
    scale = torch.tensor([0.1], dtype=torch.float32, device=device)
    zero_point = torch.tensor([128.0], dtype=torch.float32, device=device)
    return grad, self_t, scale, zero_point, 0, 255, 1.0


class FakeQuantizeLearnablePerTensorAffineBackwardBenchmark(OperatorBenchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:]

    def set_more_shapes(self):
        special_shapes_2d = [(1024, 2**i) for i in range(0, 20, 4)]
        special_shapes_3d = [(64, 64, 2**i) for i in range(0, 15, 4)]
        return special_shapes_2d + special_shapes_3d

    def get_gbps(self, args, latency):
        grad, self_t = args[0], args[1]
        io = grad.numel() * grad.element_size() * 2 + self_t.numel() * self_t.element_size()
        return io * 1e-9 / (latency * 1e-3)


@pytest.mark.fake_quantize_learnable_per_tensor_affine_backward
@pytest.mark.underscore_fake_quantize_learnable_per_tensor_affine_backward
def test_fake_quantize_learnable_per_tensor_affine_backward():
    bench = FakeQuantizeLearnablePerTensorAffineBackwardBenchmark(
        op_name="_fake_quantize_learnable_per_tensor_affine_backward",
        torch_op=torch.ops.aten._fake_quantize_learnable_per_tensor_affine_backward,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.float32],
    )
    bench.run()
