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
from .generated_operator_utils import OperatorBenchmark

SHAPES = [(4, 4), (16, 32), (32, 64), (64, 128), (1024, 1024)]
PARAMS = ((0.1, 0), (0.01, 10), (0.5, -20))


class DequantizeBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(SHAPES)
        self.shape_desc = "shape"


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    for scale, zero_point in PARAMS:
        yield base.BenchmarkCasePlan(
            shape={"self": list(dims)},
            params={"scale": scale, "zero_point": zero_point},
            builder_args=(dims, scale, zero_point),
        )


def _build_inputs_fn(plan, dtype, device):
    del dtype
    shape, scale, zero_point = plan.builder_args
    torch.manual_seed(0)
    host = torch.quantize_per_tensor(
        torch.randn(shape), scale=scale, zero_point=zero_point, dtype=torch.qint8
    )
    if flag_gems.vendor_name == "ascend" and str(device).startswith("npu"):
        target = torch._empty_affine_quantized(
            host.shape,
            scale=host.q_scale(),
            zero_point=host.q_zero_point(),
            dtype=host.dtype,
            device=device,
        )
        target.untyped_storage().copy_(host.untyped_storage())
        return (target,)
    return (host if device == "cpu" else host.to(device),)


@pytest.mark.dequantize
def test_dequantize():
    DequantizeBenchmark(
        op_name="dequantize",
        torch_op=torch.dequantize,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.qint8],
    ).run()
