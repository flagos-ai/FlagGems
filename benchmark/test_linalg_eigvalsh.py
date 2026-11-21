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

from math import prod

import pytest
import torch

import flag_gems

from . import base
from .generated_operator_utils import OperatorBenchmark

# linalg_eigvalsh operates on matrices, so every shape needs a square trailing
# pair. The first seven rows are the previously reviewed workloads; the last
# three are the spec's (20, 320, 15), (16, 128, 64, 60) and (16, 7, 57, 32, 29)
# workloads with their batch axes untouched and the trailing pair squared at a
# real trailing side, so those large batched geometries are timed as such. Both
# UPLO values are exercised for every row.
LINALG_EIGVALSH_SHAPES = [
    (16, 16),
    (64, 64),
    (256, 256),
    (1024, 1024),
    (2, 19, 19),
    (2, 3, 8, 8),
    (2, 3, 1, 8, 8),
    (20, 320, 320),
    (16, 128, 64, 64),
    (16, 7, 57, 32, 32),
]

_UPLO_VALUES = ("L", "U")

# The default overload has these additional real symmetric workloads on master.
_MASTER_SYMMETRIC_CASES = (
    ((2, 2), "L"),
    ((1, 3, 3), "U"),
    ((2, 4, 4), "L"),
    ((2, 3, 5, 5), "U"),
    ((1, 7, 7), "L"),
    ((3, 3, 3), "U"),
    ((2, 2, 4, 4), "L"),
    ((4, 6, 6), "U"),
)

# Native kernels exist only for these dtypes; complex inputs return the
# corresponding real eigenvalue dtype.
LINALG_EIGVALSH_DTYPES = [torch.float32, torch.complex64]
if flag_gems.runtime.device.support_fp64:
    LINALG_EIGVALSH_DTYPES += [torch.float64, torch.complex128]


def _validate_shapes(shapes):
    """Validate shape metadata without touching a device.

    Runs once at the planning boundary (defaults, or a caller-supplied shape
    file), so listing and execution see the same checked plans. Zero-size matrix
    and batch dimensions are valid; only the container type, the dimension types
    and the operator's rank/square contract are checked.
    """
    if isinstance(shapes, (str, bytes)) or not isinstance(shapes, (list, tuple)):
        raise TypeError("expected a list of shape tuples, got " + repr(shapes))
    for shape in shapes:
        if isinstance(shape, (str, bytes)) or not isinstance(shape, (list, tuple)):
            raise TypeError("expected a shape tuple, got " + repr(shape))
        dims = tuple(shape)
        for dim in dims:
            if isinstance(dim, bool) or not isinstance(dim, int):
                raise TypeError(
                    "shape dimensions must be ints, got "
                    + repr(dim)
                    + " in "
                    + str(dims)
                )
            if dim < 0:
                raise ValueError(
                    "shape dimensions must be non-negative, got " + str(dims)
                )
        if len(dims) < 2 or dims[-1] != dims[-2]:
            raise ValueError(
                "linalg_eigvalsh needs rank >= 2 with a square trailing pair, got "
                + str(dims)
            )


_validate_shapes(LINALG_EIGVALSH_SHAPES)


def _case_fn(shape, dtype):
    del dtype
    for uplo in _UPLO_VALUES:
        yield base.BenchmarkCasePlan(
            shape={"input": shape},
            params={"UPLO": uplo},
            builder_args=(shape, uplo),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, uplo, *symmetric = plan.builder_args
    # benchmark.utils.generate_tensor_input produces nothing for float64 and
    # complex128, so the input is built directly for every supported dtype.
    inp = torch.randn(shape, dtype=dtype, device=device)
    if symmetric:
        inp = inp + inp.transpose(-1, -2)
    return inp, {"UPLO": uplo}


class LinalgEigvalshBenchmark(OperatorBenchmark):
    def get_case_iter(self, dtype):
        yield from super().get_case_iter(dtype)
        if not self._include_master_cases or dtype.is_complex:
            return
        offset = len(self.shapes) * len(_UPLO_VALUES)
        for ordinal, (shape, uplo) in enumerate(_MASTER_SYMMETRIC_CASES, offset):
            plan = base.BenchmarkCasePlan(
                shape={"input": shape},
                params={"UPLO": uplo},
                builder_args=(shape, uplo, True),
            )
            yield self._case_from_plan(dtype, ordinal, plan)

    def build_inputs(self, case):
        if flag_gems.vendor_name == "thead" and not (
            base.Config.preflight_only or base.Config.profile_only
        ):
            shape = case.builder_args[0].builder_args[0]
            if case.dtype.is_complex:
                reason = "PPU native eigvalsh aborts on complex inputs"
            elif prod(shape[:-2]) > 1 and shape[-1] <= 32:
                reason = "PPU native eigvalsh does not support its small batched solver"
            else:
                reason = None
            if reason:
                base.Config.skipped_case_ids.add(case.case_id)
                pytest.skip(reason)
        return super().build_inputs(case)

    def set_shapes(self, shape_file_path=None, *, default_shapes=None):
        # core_shapes.yaml has no linalg_eigvalsh entry, so the local square
        # shapes are the default while a caller-supplied shape file still wins.
        # OperatorBenchmark.set_shapes selects that full list, so the generic
        # rectangular extras are never merged in and set_more_shapes does not
        # need an override. Validation happens here, once, before any tensor is
        # allocated.
        shapes = default_shapes or LINALG_EIGVALSH_SHAPES
        if shape_file_path is None:
            self.shapes = [tuple(shape) for shape in shapes]
            self.shape_desc = self.DEFAULT_SHAPE_DESC
        else:
            super().set_shapes(shape_file_path, default_shapes=shapes)
        _validate_shapes(self.shapes)
        self._include_master_cases = self.shapes == LINALG_EIGVALSH_SHAPES


@pytest.mark.linalg_eigvalsh
def test_linalg_eigvalsh():
    if flag_gems.vendor_name == "mthreads" and not (
        base.Config.list_cases or base.Config.preflight_only or base.Config.profile_only
    ):
        pytest.skip("MUSA native eigvalsh requires unavailable CPU LAPACK support")
    bench = LinalgEigvalshBenchmark(
        op_name="linalg_eigvalsh",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.linalg_eigvalsh,
        dtypes=LINALG_EIGVALSH_DTYPES,
    )
    bench.run()
