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

"""Benchmark for the sparse COO accessor aten::indices.

The returned index tensor has sparse_dim * nnz int64 entries and the operand
never has to materialize its logical shape, so a case is described as
(sparse_shape, dense_shape, nnz).  A plain shape entry from a shape file is read
as a sparse shape with the default nnz.
"""

import math

import pytest
import torch

import flag_gems

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

_DTYPE_FLAGS = {
    torch.bfloat16: flag_gems.runtime.device.support_bf16,
    torch.float64: flag_gems.runtime.device.support_fp64,
    torch.complex128: flag_gems.runtime.device.support_fp64,
    torch.int64: flag_gems.runtime.device.support_int64,
    torch.float8_e4m3fn: flag_gems.runtime.device.support_fp8,
    torch.float8_e5m2: flag_gems.runtime.device.support_fp8,
}

_DEFAULT_NNZ = 65536

# The original COO workloads: rank, dense block size and nnz.
_INDICES_SHAPES = [
    ((1024, 1024), (), 65536),
    ((1024, 1024), (), 1048576),
    ((1024, 1024), (16,), 262144),
    ((256, 256, 256), (), 1048576),
    ((128, 128, 128, 128), (8,), 1048576),
    ((4096, 4096), (8,), 262144),
]


# Every dtype group consts defines, plus float64 and FP8 where the device
# exposes them.  The accessor never reads the values, but each storage dtype
# still has to build a legal sparse operand.
_FP8_DTYPES = [
    dtype
    for name in ("float8_e4m3fn", "float8_e5m2")
    for dtype in (getattr(torch, name, None),)
    if dtype is not None
]
_DTYPES = (
    list(consts.FLOAT_DTYPES)
    + [torch.float64]
    + list(consts.INT_DTYPES)
    + list(consts.EXTRA_INT_DTYPES)
    + list(consts.BOOL_DTYPES)
    + list(consts.COMPLEX_DTYPES)
    + _FP8_DTYPES
)

_DTYPES = [dtype for dtype in _DTYPES if _DTYPE_FLAGS.get(dtype, True)]


def _normalize_shape(spec):
    """Accept (sparse_shape, dense_shape, nnz), a sparse shape, or a 1-D size."""
    if isinstance(spec, int):
        spec = [spec]
    if isinstance(spec, (str, bytes)) or not isinstance(spec, (list, tuple)):
        raise ValueError(f"invalid indices case shape {spec!r}")
    entries = list(spec)
    if entries and isinstance(entries[0], (list, tuple)):
        sparse_part = entries[0]
        dense_part = entries[1] if len(entries) > 1 else ()
        nnz = int(entries[2]) if len(entries) > 2 else _DEFAULT_NNZ
    else:
        sparse_part, dense_part, nnz = entries, (), _DEFAULT_NNZ
    try:
        sparse_shape = tuple(int(extent) for extent in sparse_part)
        dense_shape = tuple(int(extent) for extent in dense_part)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid indices case shape {spec!r}: {exc}") from exc
    if any(extent < 0 for extent in sparse_shape + dense_shape) or nnz < 0:
        raise ValueError(f"negative extent in indices case shape {spec!r}")
    return sparse_shape, dense_shape, nnz


def _distinct_flat(numel, nnz, device, seed=0):
    """Exactly min(nnz, numel) distinct flat row-major coordinates.

    A step coprime with numel makes index * step % numel injective, so the
    realized nnz always matches the request instead of shrinking when random
    draws collide; sorting restores the row-major order of a coalesced operand.
    Only nnz values are allocated, never the logical shape.
    """
    count = min(nnz, numel)
    if count == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if count == 1:
        return torch.zeros(1, dtype=torch.long, device=device)
    step = max(1, numel // count)
    while math.gcd(step, numel) != 1:
        step += 1
    index = torch.arange(count, dtype=torch.long, device=device)
    return torch.sort((index * step + seed) % numel).values


def _unravel(flat, extents):
    """Row-major unflatten of distinct flat coordinates into extents."""
    coords = torch.empty(
        (len(extents), flat.numel()), dtype=torch.long, device=flat.device
    )
    rest = flat
    for dim in reversed(range(len(extents))):
        coords[dim] = rest % extents[dim]
        rest = rest // extents[dim]
    return coords


def _make_coords(sparse_shape, nnz, device):
    coords = _unravel(
        _distinct_flat(math.prod(sparse_shape), nnz, device), sparse_shape
    )
    return coords.contiguous()


def _case_fn(shape, dtype):
    del dtype
    sparse_shape, dense_shape, nnz = _normalize_shape(shape)
    yield base.BenchmarkCasePlan(
        shape={"input": list(sparse_shape + dense_shape)},
        params={
            "sparse_shape": list(sparse_shape),
            "dense_shape": list(dense_shape),
            # Realized coordinate count: never above the logical element count,
            # which is never allocated either.
            "nnz": min(nnz, math.prod(sparse_shape)),
        },
        builder_args=(sparse_shape, dense_shape, nnz),
    )


def _build_inputs_fn(plan, dtype, device):
    sparse_shape, dense_shape, nnz = plan.builder_args
    coords = _make_coords(sparse_shape, nnz, device)
    values = torch.empty(
        (coords.shape[1],) + tuple(dense_shape), dtype=dtype, device=device
    )
    # Distinct row-major coordinates make the operand genuinely coalesced.  The
    # flag is set through the constructor because the coalesce kernel has no
    # instantiation for every supported dtype (e.g. Float8_e4m3fn), and the index
    # storage is created on the benchmark device so no transfer copies it.
    inp = torch.sparse_coo_tensor(
        coords,
        values,
        tuple(sparse_shape) + tuple(dense_shape),
        device=device,
        is_coalesced=True,
    )
    return inp, {}


class IndicesBenchmark(OperatorBenchmark):
    """Coalesced COO operands across rank, dense block size and nnz."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        self.shapes = list(
            dict.fromkeys(
                _normalize_shape(shape) for shape in list(self.shapes) + _INDICES_SHAPES
            )
        )


@pytest.mark.indices
def test_indices():
    bench = IndicesBenchmark(
        op_name="indices",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.indices,
        gems_op=getattr(flag_gems, "indices", None),
        dtypes=_DTYPES,
    )
    bench.run()
