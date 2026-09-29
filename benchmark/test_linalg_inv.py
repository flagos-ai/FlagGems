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

# Every descriptor is a batch of square matrices: the native kernel rejects
# rank < 2 and non-square input, so no generic rank-1 or rectangular shape can be
# a valid linalg_inv workload. The last entry is the rank-5 counterpart of the
# correctness suite.
_BENCH_SHAPES = [
    (256, 256),
    (1024, 1024),
    (20, 320, 320),
    (16, 128, 64, 64),
    (2, 16, 16),
    (16, 7, 57, 32, 32),
]

# Measured dtype support on the active backend: float32 / complex64 always,
# float64 / complex128 when the device reports fp64 support. fp16 and bf16 are
# rejected ('Low precision dtypes not supported') and int / bool raise
# 'Expected a floating point or complex tensor as input', so none of them can be
# benchmarked.
_BENCH_DTYPES = [torch.float32, torch.complex64]
if flag_gems.runtime.device.support_fp64:
    _BENCH_DTYPES += [torch.float64, torch.complex128]


def _validated_shape(descriptor):
    """Validate a shape descriptor from _BENCH_SHAPES or from the shapes file.

    An unusable descriptor is reported here -- for listing and for execution
    alike -- instead of being filtered out or replaced.
    """
    if not isinstance(descriptor, (tuple, list)):
        raise ValueError(
            f"linalg_inv shape descriptor must be a tuple or list, got {descriptor!r}"
        )
    shape = tuple(descriptor)
    for extent in shape:
        if isinstance(extent, bool) or not isinstance(extent, int):
            raise ValueError(
                f"linalg_inv shape extents must be integers, got {descriptor!r}"
            )
        if extent < 0:
            raise ValueError(
                f"linalg_inv shape extents must be nonnegative, got {descriptor!r}"
            )
    if len(shape) < 2 or shape[-1] != shape[-2]:
        raise ValueError(
            f"linalg_inv expects batches of square matrices, got {descriptor!r}"
        )
    return shape


def _case_fn(shape, dtype):
    del dtype
    shape = _validated_shape(shape)
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={"matrix_size": shape[-1]},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    n = shape[-1]
    if n == 0 or any(size == 0 for size in shape[:-2]):
        # An empty matrix or an empty batch is a valid input; there is nothing to
        # condition, and a reduction over the empty matrix dimensions would fail.
        return torch.empty(shape, dtype=dtype, device=device), {}
    # Well-conditioned dense input: unit-scaled values shifted by (n + 0.5) * I
    # keep the benchmark matrix invertible for every supported dtype.
    inp = torch.randn(shape, dtype=dtype, device=device)
    inp = inp / inp.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1)
    eye = torch.eye(n, dtype=dtype, device=device)
    return inp + (n + 0.5) * eye, {}


class LinalgInvBenchmark(OperatorBenchmark):
    """Two-phase GenericBenchmark over batches of square matrices."""

    def set_shapes(self, shape_file_path=None):
        # Defaults live here and stay overridable by an op / class entry in the
        # shapes file, which also keeps the generic extra shapes out of this
        # operator's case list.
        super().set_shapes(shape_file_path, default_shapes=_BENCH_SHAPES)


@pytest.mark.linalg_inv
def test_linalg_inv():
    bench = LinalgInvBenchmark(
        op_name="linalg_inv",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.linalg_inv,
        gems_op=getattr(flag_gems, "linalg_inv", None),
        dtypes=_BENCH_DTYPES,
    )
    bench.run()
