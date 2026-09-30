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

"""Benchmark for fbgemm_linear_int8_weight (host FBGEMM int8 linear).

The native operator is CPU-only and its scalar weight_scale / weight_zero_point
are Python numbers, so the builders allocate CPU operands and pack the int8
weight with the native quantize/pack pair. Case metadata stays JSON-compatible;
the packed buffer is produced only when inputs are actually built.
"""

import math

import pytest
import torch

import flag_gems

from . import base, utils
from .generated_operator_utils import OperatorBenchmark

OUTPUT_WIDTH = 8
_K_TARGET = 64

# Extra native-valid shapes kept alongside the shared grid: a non-power-of-two
# K and the spec's larger multi-rank shapes.
NATIVE_EXTRA_SHAPES = [
    (64, 33),
    (20, 320, 15),
    (16, 128, 64, 60),
    (16, 7, 57, 32, 29),
]


def _numel(shape):
    total = 1
    for dim in shape:
        total *= dim
    return total


def _native_geometry(shape):
    """Map a requested shape onto the (M, K) pair the native op needs.

    The element count is preserved: a rank >= 2 shape keeps its last dim as K
    (which must be positive), and rank 0/1 shapes are folded with
    ``math.gcd(numel, _K_TARGET)`` so M * K == numel for every requested shape.
    """
    shape = tuple(shape)
    if len(shape) >= 2 and shape[-1] > 0:
        return shape, shape[-1]
    numel = _numel(shape)
    if numel == 0:
        # The native op requires a positive K, so an empty request keeps its
        # zero element count as a (0, 1) activation.
        return (0, 1), 1
    k = math.gcd(numel, _K_TARGET)
    return (numel // k, k), k


def _case_fn(shape, dtype):
    del dtype
    inp_shape, k = _native_geometry(shape)
    yield base.BenchmarkCasePlan(
        shape={"input": list(inp_shape), "weight": [OUTPUT_WIDTH, k]},
        params={"n": OUTPUT_WIDTH, "k": k},
        builder_args=(inp_shape, k),
    )


def _build_inputs_fn(plan, dtype, device):
    # CPU-only operator: operands (and the packed handle) live on the host.
    del device
    inp_shape, k = plan.builder_args
    inp = utils.generate_tensor_input(inp_shape, dtype, "cpu")
    weight = utils.generate_tensor_input((OUTPUT_WIDTH, k), dtype, "cpu")
    bias = utils.generate_tensor_input((OUTPUT_WIDTH,), dtype, "cpu")
    (
        qweight,
        col_offsets,
        weight_scale,
        weight_zero_point,
    ) = torch.ops.aten.fbgemm_linear_quantize_weight(weight)
    packed = torch.ops.aten.fbgemm_pack_quantized_matrix(qweight)
    # unpack_to_args_kwargs takes flat positional operands plus a kwargs dict.
    return (
        inp,
        weight,
        packed,
        col_offsets,
        float(weight_scale),
        int(weight_zero_point),
        bias,
        {},
    )


class FbgemmLinearInt8WeightBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        # Keep the shared grid (and any caller-supplied shape file) untouched,
        # then add the extra native-valid shapes it does not already contain.
        super().set_shapes(shape_file_path)
        known = {tuple(shape) for shape in self.shapes}
        for shape in NATIVE_EXTRA_SHAPES:
            if tuple(shape) not in known:
                self.shapes.append(shape)
                known.add(tuple(shape))


@pytest.mark.fbgemm_linear_int8_weight
def test_fbgemm_linear_int8_weight():
    bench = FbgemmLinearInt8WeightBenchmark(
        op_name="fbgemm_linear_int8_weight",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.fbgemm_linear_int8_weight,
        gems_op=getattr(flag_gems, "fbgemm_linear_int8_weight", None),
        dtypes=[torch.float32],
    )
    bench.run()
