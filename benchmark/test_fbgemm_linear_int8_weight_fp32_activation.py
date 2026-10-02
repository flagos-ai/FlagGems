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

import flag_gems

from . import base
from .generated_operator_utils import OperatorBenchmark

# aten::fbgemm_linear_int8_weight_fp32_activation is a host-side FBGEMM int8
# GEMM: float32 activations only, CPU operands, and the native torch packer (not
# FlagGems' private _wrapped_linear_prepack format). A geometry is (M, K, N):
# M activation rows, K reduction dim, N outputs.
FBGEMM_LINEAR_GEOMETRIES = [
    (64, 256, 64),
    (128, 512, 128),
    (256, 1024, 256),
    (512, 2048, 512),
    (1024, 1024, 1024),
]

_MAX_GENERIC_OUTPUTS = 256
_DEFAULT_REDUCTION_DIM = 1024


def _matrix_geometry(shape):
    """Map a requested shape onto a native-valid (activation, K, N) workload.

    FBGEMM_LINEAR_GEOMETRIES entries are native geometries already. The other
    entries come from the shared grid, which describes pointwise workloads where
    only the element count matters. Such a shape keeps its full element count
    (M * K == numel(shape)) and, at rank >= 2, its requested rank and dimension
    sizes: the last dim is the native reduction dim, so the native rank branches
    are preserved instead of being flattened into a single row count. Rank < 2
    shapes are factored with the greatest common divisor against the default
    reduction dim, so an arbitrary row count never drops a remainder.
    """
    shape = tuple(shape)
    if shape in FBGEMM_LINEAR_GEOMETRIES:
        m, k, n = shape
        return (m, k), k, n
    numel = math.prod(shape) if shape else 1
    if numel == 0:
        # A zero-element request has no reduction dim to keep; the native
        # kernel accepts M == 0 with any K > 0.
        return (
            (0, _DEFAULT_REDUCTION_DIM),
            _DEFAULT_REDUCTION_DIM,
            _MAX_GENERIC_OUTPUTS,
        )
    if len(shape) >= 2:
        k = shape[-1]
        return shape, k, min(k, _MAX_GENERIC_OUTPUTS)
    k = math.gcd(numel, _DEFAULT_REDUCTION_DIM)
    return (numel // k, k), k, min(k, _MAX_GENERIC_OUTPUTS)


def _case_fn(shape, dtype):
    del dtype
    activation, k, n = _matrix_geometry(shape)
    yield base.BenchmarkCasePlan(
        shape={"input": list(activation), "weight": [n, k]},
        params={"weight_scale": 0.01, "weight_zero_point": 0},
        builder_args=(activation, k, n),
    )


def _build_inputs_fn(plan, dtype, device):
    # unpack_to_args_kwargs forwards flat positional args, so the tuple below is
    # the native schema order:
    #   (input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias)
    # The operator is host-only, so every operand is built on CPU and the single
    # built set is shared by the reference and the candidate.
    del device
    activation, k, n = plan.builder_args
    inp = torch.randn(*activation, dtype=dtype)
    weight = torch.randint(-128, 128, (n, k), dtype=torch.int8)
    col_offsets = torch.sum(weight, dim=1, dtype=torch.int32)
    packed = torch.ops.aten.fbgemm_pack_quantized_matrix(weight)
    bias = torch.randn(n, dtype=torch.float32)
    return (
        inp,
        weight,
        packed,
        col_offsets,
        plan.params["weight_scale"],
        plan.params["weight_zero_point"],
        bias,
    )


class FbgemmLinearBenchmark(OperatorBenchmark):
    """Two-phase benchmark over the shared grid plus the native-valid extras."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        # Union the FBGEMM workloads onto the framework grid (or the caller's
        # --shape_file) instead of replacing the shared grid.
        self.shapes = list(
            dict.fromkeys([tuple(s) for s in self.shapes] + FBGEMM_LINEAR_GEOMETRIES)
        )


@pytest.mark.fbgemm_linear_int8_weight_fp32_activation
def test_fbgemm_linear_int8_weight_fp32_activation():
    bench = FbgemmLinearBenchmark(
        op_name="fbgemm_linear_int8_weight_fp32_activation",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.fbgemm_linear_int8_weight_fp32_activation,
        gems_op=getattr(flag_gems, "fbgemm_linear_int8_weight_fp32_activation", None),
        dtypes=[torch.float32],
    )
    bench.run()
