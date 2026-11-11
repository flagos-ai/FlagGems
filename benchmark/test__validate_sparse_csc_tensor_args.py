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

from contextlib import nullcontext

import pytest
import torch

import flag_gems

from . import base, consts, utils
from .generated_operator_utils import OperatorBenchmark

_ASCEND_VALUES_RANK_SKIP = pytest.mark.skipif(
    flag_gems.vendor_name == "ascend",
    reason="Ascend normal_ does not support values with more than 8 dimensions",
)

# Workloads as (size, nnz, dense_ndim, index dtype name). The op reads CSC index
# metadata instead of a dense shape, so no entry of core_shapes.yaml describes it;
# a caller-supplied shape file still takes precedence over these descriptors.
# A timed call covers the host preconditions of
# _validate_sparse_compressed_tensor_args_worker and, for non-meta index tensors,
# the device kernel that checks the index invariants: the launch is sized by the
# batch of column blocks and the stored entries, so the measured work grows with
# nnz and ncols, not with the logical nrows * ncols geometry.
_WORKLOADS = [
    ((1024, 1024), 65536, 0, "int64"),
    ((1024, 1024), 262144, 0, "int64"),
    ((4096, 4096), 1048576, 0, "int64"),
    ((100000, 100), 100000, 0, "int64"),
    ((16, 128, 64, 60), 1920, 0, "int64"),
    ((2, 3, 4, 5, 7), 20, 1, "int64"),
    ((1,) * 8 + (2, 3), 3, 0, "int64"),
    ((1024, 1024), 262144, 0, "int32"),
]

# Master constructs two entries per column and zero-filled values for these
# four workloads. Retain the existing descriptors and their order.
_MASTER_WORKLOADS = (
    ((5, 3), 6, 0, "int64"),
    ((256, 256), 512, 0, "int64"),
    ((4096, 4096), 8192, 0, "int64"),
    ((8, 64, 64), 128, 0, "int64"),
)

# The index dtype travels as a name so the listed metadata stays JSON-compatible;
# builder_args keeps the real torch dtype.
_INDEX_DTYPES = {"int32": torch.int32, "int64": torch.int64}
_INDEX_LIMITS = {torch.int32: 2**31, torch.int64: 2**63}

_RUNTIME_DEVICE = flag_gems.runtime.device

_DTYPE_GATES = {
    torch.bfloat16: "support_bf16",
    torch.float64: "support_fp64",
    torch.float8_e4m3fn: "support_fp8",
    torch.float8_e4m3fnuz: "support_fp8",
    torch.float8_e5m2: "support_fp8",
    torch.float8_e5m2fnuz: "support_fp8",
    torch.int64: "support_int64",
}


def _dtype_supported(dtype):
    # Static capability flags only: nothing is probed at listing or run time. A
    # flag that is absent is treated as available, so an unsupported dtype fails
    # at build time instead of being dropped silently.
    flag = _DTYPE_GATES.get(dtype)
    return flag is None or bool(getattr(_RUNTIME_DEVICE, flag, True))


def _index_dtype(name):
    # The dtype named by the descriptor is used as-is. A descriptor whose index
    # dtype the backend cannot build is rejected by _plan_for; it is never
    # relabelled to another dtype, so a listed index_dtype always names the dtype
    # that is actually allocated.
    return _INDEX_DTYPES[name]


# Descriptors whose index dtype the backend cannot build are dropped here, once,
# statically. Listing and execution read the same filtered list, so a listed case
# is always one that execution can build; on a backend with int64 support all
# eight descriptors are kept.
WORKLOADS = [
    descriptor
    for descriptor in (*_WORKLOADS, *_MASTER_WORKLOADS)
    if _dtype_supported(_index_dtype(descriptor[3]))
]

# utils.generate_tensor_input only produces floating point values.
_BENCH_DTYPES = [dtype for dtype in consts.FLOAT_DTYPES if _dtype_supported(dtype)]


def _split_size(size, dense_ndim):
    batch = tuple(size[: len(size) - 2 - dense_ndim])
    nrows = size[len(batch)]
    ncols = size[len(batch) + 1]
    dense = tuple(size[len(batch) + 2 :])
    return batch, nrows, ncols, dense


def _column_counts(batch_count, ncols, nnz):
    # One valid count vector (column j holds rows 0..count_j-1, and every count is
    # at most ceil(nnz / ncols) <= nrows because nnz <= nrows * ncols), rotated by
    # one column per batch: each batch keeps the total nnz and the per-column
    # bound while the batches get different column distributions whenever the
    # counts are not all equal.
    counts = torch.zeros(batch_count, ncols, dtype=torch.int64)
    if ncols == 0:
        return counts
    counts += nnz // ncols
    counts[:, : nnz % ncols] += 1
    if batch_count > 1 and ncols > 1:
        columns = torch.arange(ncols, dtype=torch.int64)[None, :]
        shift = (torch.arange(batch_count, dtype=torch.int64) % ncols)[:, None]
        counts = torch.gather(counts, 1, (columns - shift) % ncols)
    return counts


def _plan_for(descriptor):
    # Validate one listed descriptor and return its plan fields. Invalid
    # descriptors are rejected here, during listing as well as execution, rather
    # than clamped: nnz is never reduced and the requested geometry is never
    # reshaped into something the native validator happens to accept.
    if not isinstance(descriptor, (tuple, list)) or len(descriptor) != 4:
        raise ValueError(
            "descriptor must be (size, nnz, dense_ndim, index_dtype), got "
            + repr(descriptor)
        )
    size, nnz, dense_ndim, index_dtype = descriptor
    if not isinstance(size, (tuple, list)) or len(size) < 2:
        raise ValueError(
            "size must be a sequence of at least 2 dimensions, got " + repr(size)
        )
    if any(
        not isinstance(dim, int) or isinstance(dim, bool) or dim < 0 for dim in size
    ):
        raise ValueError("size entries must be non-negative ints, got " + repr(size))
    if not isinstance(nnz, int) or isinstance(nnz, bool) or nnz < 0:
        raise ValueError("nnz must be a non-negative int, got " + repr(nnz))
    if (
        not isinstance(dense_ndim, int)
        or isinstance(dense_ndim, bool)
        or dense_ndim < 0
    ):
        raise ValueError(
            "dense_ndim must be a non-negative int, got " + repr(dense_ndim)
        )
    if index_dtype not in _INDEX_DTYPES:
        raise ValueError(
            "index_dtype must be one of "
            + repr(sorted(_INDEX_DTYPES))
            + ", got "
            + repr(index_dtype)
        )
    if len(size) < dense_ndim + 2:
        raise ValueError(
            "dense_ndim " + repr(dense_ndim) + " does not fit size " + repr(size)
        )
    _, nrows, ncols, _ = _split_size(tuple(size), dense_ndim)
    if nnz > nrows * ncols:
        raise ValueError(
            "nnz "
            + repr(nnz)
            + " exceeds the "
            + repr(nrows)
            + "x"
            + repr(ncols)
            + " sorted-distinct column capacity of size "
            + repr(list(size))
        )
    resolved = _index_dtype(index_dtype)
    if not _dtype_supported(resolved):
        raise ValueError(
            index_dtype
            + " indices are not available on the active backend; drop this"
            + " descriptor or name a supported index dtype"
        )
    limit = _INDEX_LIMITS[resolved]
    if max(nnz + 1, ncols + 1, nrows) >= limit:
        raise ValueError(
            index_dtype + " indices cannot represent size " + repr(list(size))
        )
    return tuple(size), nnz, dense_ndim, index_dtype


def _case_fn(shape, dtype):
    # Two-phase benchmark: get_case_iter calls this once per shape entry, so one
    # plan per workload descriptor, and building the case list allocates no
    # tensors.
    del dtype
    size, nnz, dense_ndim, index_dtype = _plan_for(shape)
    params = {
        "nnz": nnz,
        "dense_ndim": dense_ndim,
        "index_dtype": index_dtype,
    }
    if (size, nnz, dense_ndim, index_dtype) in _MASTER_WORKLOADS:
        params["value_init"] = "zeros"
    yield base.BenchmarkCasePlan(
        shape={"input": list(size)},
        params=params,
        builder_args=(list(size), nnz, dense_ndim, index_dtype),
    )


def _build_inputs_fn(plan, dtype, device):
    size, nnz, dense_ndim, index_dtype = plan.builder_args
    batch, _, ncols, dense = _split_size(size, dense_ndim)
    batch_count = 1
    for dim in batch:
        batch_count *= dim
    counts = _column_counts(batch_count, ncols, nnz)
    ccol = torch.zeros(batch_count, ncols + 1, dtype=torch.int64)
    if ncols:
        ccol[:, 1:] = torch.cumsum(counts, 1)
    row = torch.zeros(batch_count, nnz, dtype=torch.int64)
    if nnz:
        offsets = (
            torch.arange(nnz, dtype=torch.int64).expand(batch_count, nnz).contiguous()
        )
        starts = ccol[:, :-1].contiguous()
        column = torch.searchsorted(starts, offsets, right=True) - 1
        row = offsets - torch.gather(starts, 1, column)
    value_shape = batch + (nnz,) + dense
    values = (
        torch.zeros(value_shape, dtype=dtype, device=device)
        if plan.params.get("value_init") == "zeros"
        else utils.generate_tensor_input(value_shape, dtype, device)
    )
    resolved = _index_dtype(index_dtype)
    return (
        ccol.reshape(batch + (ncols + 1,)).to(device, resolved),
        row.reshape(batch + (nnz,)).to(device, resolved),
        values,
        {"size": list(size)},
    )


class ValidateSparseCscTensorArgsBenchmark(OperatorBenchmark):
    high_dimensional = None

    def _candidate_call(self):
        registry = base.Config.override_registry
        override = registry.get_override(self.op_name) if registry is not None else None
        candidate = (
            override if override is not None else getattr(flag_gems, self.op_name, None)
        )
        if not callable(candidate):
            raise RuntimeError(f"No callable candidate for {self.op_name}")
        return candidate, nullcontext(), override is not None

    # These descriptors are the workloads for --list-cases and for normal
    # execution; a caller-supplied shape file still replaces them.
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=WORKLOADS)
        if self.high_dimensional is not None:
            # _plan_for returns (size, ...). Values replace the two sparse matrix
            # axes with one nnz axis, so their rank is len(size) - 1.
            self.shapes = [
                shape
                for shape in self.shapes
                if (len(_plan_for(shape)[0]) - 1 > 8) == self.high_dimensional
            ]
            if not self.shapes:
                pytest.skip("No CSC workloads in this values-rank group")


@pytest.mark.underscore_validate_sparse_csc_tensor_args
@pytest.mark.parametrize(
    "high_dimensional",
    [
        pytest.param(
            high_dimensional, marks=_ASCEND_VALUES_RANK_SKIP if high_dimensional else ()
        )
        for high_dimensional in [False, True]
    ],
    ids=["values-rank-le8", "values-rank-gt8"],
)
def test__validate_sparse_csc_tensor_args(high_dimensional):
    # A skip in the builder would abort the entire benchmark node, including
    # supported shapes. Isolate the high-rank inputs in their own pytest node.
    # Both implementations return None; candidate injection supplies the direct
    # callable while the original catalog and default metrics stay unchanged.
    bench = ValidateSparseCscTensorArgsBenchmark(
        op_name="_validate_sparse_csc_tensor_args",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._validate_sparse_csc_tensor_args,
        dtypes=_BENCH_DTYPES,
        high_dimensional=high_dimensional,
    )
    bench.run()
