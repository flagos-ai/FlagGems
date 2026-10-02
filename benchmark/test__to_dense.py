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

# aten::_to_dense(Tensor self) -> Tensor : sparse COO -> dense strided.
#
# Every fixture builds its coordinates as an int64 index tensor, so the COO cases
# need the int64 element capability. The flag is read statically from the runtime
# device description and gates the descriptor and dtype collection for listing and
# execution alike; there is no runtime probe and no skip. Compressed layouts carry
# int32 metadata and are covered by the correctness tests instead.
_SUPPORT_INT64 = flag_gems.runtime.device.support_int64
#
# Each descriptor is (sparse logical shape, stored entries); the stored count is
# always <= prod(shape) so the requested entries actually fit. The rank-0 rows
# keep the sparse convention (indices of shape (0, nnz)) and the empty rows
# measure the allocation path of a tensor with no stored elements.
_DEFAULT_DESCRIPTORS = [
    ((1024, 1024), 65536),
    ((1024, 1024), 1048576),
    ((4096, 4096), 1048576),
    ((64, 512, 512), 1048576),
    ((20, 320, 15), 65536),
    ((16, 128, 64, 60), 524288),
    ((16, 7, 57, 32, 29), 262144),
    ((1024, 1024), 0),
    ((), 1),
    ((), 0),
]
if not _SUPPORT_INT64:
    _DEFAULT_DESCRIPTORS = []

# bf16 descriptors are listed only when the target advertises support, read from
# the runtime capability flag rather than from the Python import environment.
_BENCH_DTYPES = list(consts.FLOAT_DTYPES) if _SUPPORT_INT64 else []
if not flag_gems.runtime.device.support_bf16:
    _BENCH_DTYPES = [dtype for dtype in _BENCH_DTYPES if dtype is not torch.bfloat16]


def _extent(shape):
    total = 1
    for dim in shape:
        total *= dim
    return total


def _sampled_indices(shape, nnz, device):
    """Distinct stored coordinates for ``nnz`` entries of ``shape``.

    ``arange(nnz) * (extent // nnz)`` stays strictly inside the extent, so the
    sampled coordinates are distinct and the measured work is the densification
    itself rather than duplicate accumulation.
    """
    if len(shape) == 0:
        return torch.zeros((0, nnz), dtype=torch.int64, device=device)
    if nnz == 0:
        return torch.empty((len(shape), 0), dtype=torch.int64, device=device)
    extent = _extent(shape)
    step = max(1, extent // nnz)
    flat = (torch.arange(nnz, dtype=torch.int64) * step) % extent
    return torch.stack(torch.unravel_index(flat, tuple(shape))).to(device)


def _validate(descriptor):
    """Validate one shape-file entry before it becomes a case plan.

    A descriptor is ``(sparse shape, stored entries)`` and comes from a
    user-editable shape file, so every field is checked as it is written: nothing
    is coerced, clamped or rounded, and an invalid entry is rejected instead of
    being silently re-planned. The stored count must fit inside the extent,
    because a descriptor cannot ask for more distinct entries than the shape has
    coordinates.
    """
    if not isinstance(descriptor, (list, tuple)) or len(descriptor) != 2:
        raise ValueError(f"descriptor must be (shape, nnz), got {descriptor!r}")
    shape, nnz = descriptor
    if not isinstance(shape, (list, tuple)):
        raise ValueError(f"sparse shape must be a sequence, got {type(shape).__name__}")
    dims = []
    for dim in shape:
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise ValueError(f"sparse shape dimension must be an int, got {dim!r}")
        if dim < 0:
            raise ValueError(f"sparse shape dimension must be non-negative, got {dim}")
        dims.append(dim)
    if isinstance(nnz, bool) or not isinstance(nnz, int):
        raise ValueError(f"stored entries must be an int, got {nnz!r}")
    if nnz < 0:
        raise ValueError(f"stored entries must be non-negative, got {nnz}")
    if nnz > _extent(dims):
        raise ValueError(
            f"stored entries {nnz} exceed the {_extent(dims)} distinct coordinates "
            f"of sparse shape {tuple(dims)}"
        )
    return tuple(dims), nnz


def _case_fn(shape, dtype):
    del dtype
    sparse_shape, nnz = _validate(shape)
    yield base.BenchmarkCasePlan(
        shape={"input": list(sparse_shape), "nnz": nnz},
        params={},
        builder_args=(sparse_shape, nnz),
    )


def _build_inputs_fn(plan, dtype, device):
    sparse_shape, nnz = plan.builder_args
    indices = _sampled_indices(sparse_shape, nnz, device)
    values = torch.rand(nnz, dtype=dtype, device=device)
    inp = torch.sparse_coo_tensor(indices, values, tuple(sparse_shape), device=device)
    return inp, {}


class ToDenseBenchmark(OperatorBenchmark):
    """Two-phase GenericBenchmark with the dedicated sparse layouts.

    ``core_shapes.yaml`` has no entry for this op, so the descriptor list above
    is supplied as defaults; operator entries in a shape file still override it.
    """

    DEFAULT_SHAPE_DESC = "sparse_coo(shape, nnz)"

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_DEFAULT_DESCRIPTORS)


@pytest.mark.to_dense
def test__to_dense():
    bench = ToDenseBenchmark(
        op_name="_to_dense",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._to_dense,
        gems_op=getattr(flag_gems, "_to_dense", None),
        dtypes=_BENCH_DTYPES,
    )
    bench.run()
