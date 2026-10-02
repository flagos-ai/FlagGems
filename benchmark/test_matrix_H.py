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

# aten::matrix_H is a zero-copy conjugate-transpose view: it reads no element, so
# the benchmark measures dispatch plus view construction and the input can be an
# uninitialized allocation, since the harness never compares values. Only the
# native-supported matrix and deprecated scalar ranks are used.
_DTYPE_CAPABILITY = {
    torch.bfloat16: "support_bf16",
    torch.float64: "support_fp64",
    torch.complex128: "support_fp64",
    torch.int64: "support_int64",
    torch.float8_e4m3fn: "support_fp8",
    torch.float8_e5m2: "support_fp8",
}


def _dtype_supported(dtype):
    flag_name = _DTYPE_CAPABILITY.get(dtype)
    if flag_name is None:
        return True
    return bool(getattr(flag_gems.runtime.device, flag_name, False))


# Every dtype the operator accepts on the active backend: the required integer
# and FP8 types, bf16/fp16/fp32, plus fp64/bool/complex where supported. The
# view path never widens them, so all of them stay timed.
MATRIX_H_DTYPES = [
    dtype
    for dtype in (
        torch.int8,
        torch.uint8,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float32,
        torch.bfloat16,
        torch.float16,
        torch.int32,
        torch.int64,
        torch.float64,
        torch.bool,
        torch.complex64,
        torch.complex128,
    )
    if _dtype_supported(dtype)
]

# Semantic boundaries added to the shared shape sets: a tiny matrix, both
# vector orientations and both non-square orientations (wide and tall).
MATRIX_H_BOUNDARY_SHAPES = [
    (),
    (0, 3),
    (3, 5),
    (1, 4096),
    (4096, 1),
    (128, 512),
    (512, 128),
]


def _case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    return torch.empty(shape, dtype=dtype, device=device), {}


class MatrixHBenchmark(base.GenericBenchmark):
    """Shared shape sets plus the boundary shapes, restricted to native-supported ranks 0 and 2.

    aten::matrix_H raises RuntimeError for rank != 2 (rank 0 is a deprecated
    scalar path), while the shared default and comprehensive sets contain 1-D
    and 3-D entries, so those ranks are dropped and every valid requested 2-D
    shape (including shape-file entries) is kept unchanged.
    """

    @staticmethod
    def _rank2(shapes):
        return [tuple(shape) for shape in shapes if len(shape) in (0, 2)]

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        self.shapes = list(
            dict.fromkeys(self._rank2(self.shapes) + MATRIX_H_BOUNDARY_SHAPES)
        )


@pytest.mark.matrix_H
def test_matrix_H():
    bench = MatrixHBenchmark(
        op_name="matrix_H",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.matrix_H,
        gems_op=getattr(flag_gems, "matrix_H", None),
        dtypes=MATRIX_H_DTYPES,
    )
    bench.run()
