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
from .generated_operator_utils import OperatorBenchmark


FP16_BF16_DTYPES = [dtype for dtype in consts.FLOAT_DTYPES if dtype != torch.float32]
MIXED_DTYPES_LINEAR_SHAPES = [
    (1, 4096, 4096, "int8", True, "silu"),
    (1, 4096, 11008, "int8", True, "silu"),
    (1, 11008, 4096, "int8", True, "none"),
    (4, 4096, 4096, "int8", True, "silu"),
    (4, 4096, 11008, "int4", True, "silu"),
    (16, 4096, 4096, "int8", False, "relu"),
    (16, 4096, 11008, "int4", True, "silu"),
    (32, 4096, 4096, "int8", True, "none"),
    (64, 4096, 11008, "int8", True, "silu"),
    (128, 4096, 4096, "int4", True, "silu"),
]


class MixedDtypesLinearBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = MIXED_DTYPES_LINEAR_SHAPES
        self.shape_desc = "M, K, N, mode, has_bias, activation"


def _case_fn(shape, dtype):
    del dtype
    m_size, k_size, n_size, mode, has_bias, activation = shape
    ncols = n_size if mode == "int8" else n_size // 2
    yield base.BenchmarkCasePlan(
        shape={
            "input": [m_size, k_size],
            "weight": [k_size, ncols],
            "scale": [n_size],
            "bias": [n_size] if has_bias else None,
        },
        params={
            "mode": mode,
            "has_bias": has_bias,
            "activation": activation,
        },
        builder_args=(m_size, k_size, n_size, mode, has_bias, activation),
    )


def _build_inputs_fn(plan, dtype, device):
    m_size, k_size, n_size, mode, has_bias, activation = plan.builder_args
    torch.manual_seed(0)
    input_tensor = torch.randn((m_size, k_size), dtype=dtype, device=device)
    ncols = n_size if mode == "int8" else n_size // 2
    weight = torch.randint(
        0, 256, (k_size, ncols), dtype=torch.uint8, device=device
    )
    scale = torch.randn((n_size,), dtype=dtype, device=device)
    bias = torch.randn((n_size,), dtype=dtype, device=device) if has_bias else None
    return input_tensor, weight, scale, {"bias": bias, "activation": activation}


def _reference(input_tensor, weight, scale, *, bias=None, activation=None):
    k_size, ncols = weight.shape
    n_size = scale.shape[0]
    if ncols == n_size:
        unpacked = weight.to(torch.int32) - 128
    else:
        low = (weight.to(torch.int32) & 0xF) - 8
        high = ((weight.to(torch.int32) >> 4) & 0xF) - 8
        unpacked = torch.stack([low, high], dim=-1).reshape(k_size, n_size)
    if input_tensor.dtype == torch.float16:
        accumulator = input_tensor.float() @ unpacked.float()
        accumulator = accumulator * scale.float()
    else:
        dequantized = (unpacked.float() * scale.float().unsqueeze(0)).to(
            input_tensor.dtype
        )
        accumulator = input_tensor.float() @ dequantized.float()
    if bias is not None:
        accumulator = accumulator + bias.float()
    if activation == "relu":
        accumulator = torch.relu(accumulator)
    elif activation == "silu":
        accumulator = accumulator * torch.sigmoid(accumulator)
    return accumulator.to(input_tensor.dtype)


def _gems(input_tensor, weight, scale, *, bias=None, activation=None):
    return flag_gems.mixed_dtypes_linear(
        input_tensor, weight, scale, bias=bias, activation=activation
    )


@pytest.mark.mixed_dtypes_linear
def test_mixed_dtypes_linear():
    MixedDtypesLinearBenchmark(
        op_name="mixed_dtypes_linear",
        torch_op=_reference,
        gems_op=_gems,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=FP16_BF16_DTYPES,
    ).run()
