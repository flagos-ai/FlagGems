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

"""Benchmark for fbgemm_linear_fp16_weight_fp32_activation (host FBGEMM path)."""

import math

import pytest
import torch

import flag_gems

from . import base, utils

# Output columns of the packed weight.
N = 32

# Operator-specific activation shapes (rank >= 2; the last dim is K). They are
# unioned with the shared shape grid, so the generic pointwise shapes and any
# caller-supplied shape file keep their own workloads.
_OP_SHAPES = [
    (1024, 1024),
    (4096, 4096),
    (20, 320, 15),
    (16, 128, 64, 60),
    (16, 7, 57, 32, 29),
]

_PACKED = {}


def _native_geometry(shape):
    """Return a rank >= 2 activation geometry with the same element count.

    The shared grid is operator-agnostic and leads with a rank-1 shape, which
    this op cannot consume ("Expected input.dim() >= 2 to be true"). A
    lower-rank request is folded into a matrix instead of being dropped, so the
    requested numel is preserved.
    """
    shape = tuple(int(extent) for extent in shape)
    if len(shape) >= 2:
        return shape
    numel = 1
    for extent in shape:
        numel *= extent
    if numel == 0:
        return (0, 1)
    rows = math.isqrt(numel)
    while rows > 1 and numel % rows:
        rows -= 1
    return (rows, numel // rows)


def _packed_weight(k, n):
    # The packer reads K = weight.size(1), so the host weight is (n, k). Packing
    # is a separate host op and stays out of the measured region; the cache also
    # keeps the packed handle alive for as long as the benchmark runs.
    if (k, n) not in _PACKED:
        weight = utils.generate_tensor_input((n, k), torch.float32, "cpu")
        _PACKED[(k, n)] = torch.ops.aten.fbgemm_pack_gemm_matrix_fp16(weight)
    return _PACKED[(k, n)]


def _case_fn(shape, dtype):
    del dtype
    geometry = _native_geometry(shape)
    entry = {"activation": list(geometry), "weight": [N, geometry[-1]]}
    if tuple(geometry) != tuple(shape):
        entry["requested"] = [int(extent) for extent in shape]
    yield base.BenchmarkCasePlan(
        shape=entry,
        builder_args=(geometry, N),
    )


def _build_inputs_fn(plan, dtype, device):
    del device  # host-only operator: every operand stays on the CPU
    shape, n = plan.builder_args
    activation = utils.generate_tensor_input(tuple(shape), dtype, "cpu")
    bias = utils.generate_tensor_input((n,), dtype, "cpu")
    # unpack_to_args_kwargs consumes a flat sequence and merges dict entries into
    # kwargs, so the operands are returned positionally.
    return activation, _packed_weight(shape[-1], n), bias, {}


class _FbgemmLinearBenchmark(base.GenericBenchmark):
    """GenericBenchmark keeping the shared shape grid for this host operator."""

    DEFAULT_SHAPE_DESC = (
        "activation shape (last dim is K); weight is packed from (32, K)"
    )

    def set_shapes(self, shape_file_path=None):
        # Resolve the shared shape grid (or a caller-supplied shape list/file)
        # exactly as the base class does, then add this operator's own
        # native-valid geometries and express any rank < 2 request as a matrix
        # with the same element count.
        super().set_shapes(shape_file_path)
        merged = [_native_geometry(shape) for shape in list(self.shapes) + _OP_SHAPES]
        self.shapes = list(dict.fromkeys(merged))


@pytest.mark.fbgemm_linear_fp16_weight_fp32_activation
def test_fbgemm_linear_fp16_weight_fp32_activation():
    # Only float32 activations are accepted natively, so this op is benchmarked
    # with that dtype; the timing mode is left to the caller (--mode operator).
    bench = _FbgemmLinearBenchmark(
        op_name="fbgemm_linear_fp16_weight_fp32_activation",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.fbgemm_linear_fp16_weight_fp32_activation,
        gems_op=getattr(flag_gems, "fbgemm_linear_fp16_weight_fp32_activation", None),
        dtypes=[torch.float32],
    )
    bench.run()
