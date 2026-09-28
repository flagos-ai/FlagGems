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

# Benchmark for aten::to_sparse (dense to sparse layout conversion).
#
# The rows below cover the large spec shapes, the scalar and empty boundaries,
# every layout target the operator accepts (default COO, csr, csc, bsr with
# blocksize, bsc with blocksize, strided) and the optional sparse_dim and
# dense_dim parameters. Each row is decoded once: the decoded keyword arguments
# drive execution and the listing metadata is derived from those same decoded
# values, so listing and replay cannot diverge.
#
# Native probe: the dense to COO path runs the nonzero kernel, which has no FP8
# instantiation (RuntimeError: "nonzero_cuda" not implemented for
# 'Float8E4m3fn'), so FP8 has no COO row here. FP8 keeps its coverage on the
# compressed and strided layouts in tests/test_to_sparse.py.
#
# The COO plans are additionally filtered by the static int64 capability flag
# read from the active backend, so a backend without int64 index tensors keeps
# the compressed and strided plans instead of listing unreachable COO rows.

import pytest
import torch

import flag_gems

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

_COMPRESSED_LAYOUTS = (
    torch.sparse_csr,
    torch.sparse_csc,
    torch.sparse_bsr,
    torch.sparse_bsc,
)
_BLOCK_LAYOUTS = (torch.sparse_bsr, torch.sparse_bsc)
# Dtypes whose dense source can be converted to COO by the native operator,
# filtered by the same static capability flag tests/accuracy_utils.py reads: the
# input builder must not allocate a dtype the backend cannot compute on.
_COO_CAPABLE_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_BF16_SUPPORTED = flag_gems.runtime.device.support_bf16
_BENCH_DTYPES = [
    dtype
    for dtype in consts.FLOAT_DTYPES
    if dtype in _COO_CAPABLE_DTYPES and (dtype != torch.bfloat16 or _BF16_SUPPORTED)
]

# Case rows: (target, shape, sparse_dim, dense_dim, blocksize). The target is
# 'default' for the schema defaults, 'sparse_dim' for the positional overload or
# a layout name from _layout_for. A bare shape row read from a shape file (empty
# for a scalar) means the schema-default workload.
_ALL_BENCH_CASES = (
    ("default", (1024, 1024), None, None, None),
    ("default", (20, 320, 15), None, None, None),
    ("default", (16, 128, 64, 60), None, None, None),
    ("default", (16, 7, 57, 32, 29), None, None, None),
    ("default", (), None, None, None),
    ("default", (0, 512), None, None, None),
    ("sparse_dim", (1024, 1024), 2, None, None),
    ("dense_dim", (1024, 1024), None, 1, None),
    # Supplement: an explicit dense_dim=0 is a real argument and has to reach
    # execution instead of being dropped as a falsey default.
    ("dense_dim", (1024, 1024), None, 0, None),
    ("csr", (20, 320, 15), None, None, None),
    ("csc", (1024, 1024), None, None, None),
    ("bsr", (1024, 1024), None, None, (64, 64)),
    ("bsc", (1024, 1024), None, None, (64, 64)),
    ("strided", (1024, 1024), None, None, None),
)

# Static capability gate, read once from the active backend at import: a COO
# result carries an int64 index tensor, so the COO plans (the schema default,
# the positional sparse_dim overload and dense_dim) are dropped when the backend
# has no int64 support. The compressed plans keep the index dtype they are built
# with and the strided plan has no index tensor at all, so those stay enabled
# independently. The filter is applied once to the table that both --list-cases
# and execution read, so listing and replay cannot disagree.
_COO_PLANS_AVAILABLE = flag_gems.runtime.device.support_int64
_COO_TARGETS = ("default", "sparse_dim", "dense_dim", "coo")
_BENCH_CASES = tuple(
    row
    for row in _ALL_BENCH_CASES
    if _COO_PLANS_AVAILABLE or row[0] not in _COO_TARGETS
)


def _plain_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _nonneg_int(value, field):
    # Validated before any normalisation: a fractional, boolean or negative
    # descriptor must raise here instead of being coerced into a different call.
    if not _plain_int(value):
        raise ValueError(field + " must be an int, got " + repr(value))
    if value < 0:
        raise ValueError(field + " must not be negative, got " + repr(value))
    return int(value)


def _int_or_none(value, field):
    if value is None:
        return None
    return _nonneg_int(value, field)


def _layout_for(name):
    if name == "coo":
        return torch.sparse_coo
    if name == "csr":
        return torch.sparse_csr
    if name == "csc":
        return torch.sparse_csc
    if name == "bsr":
        return torch.sparse_bsr
    if name == "bsc":
        return torch.sparse_bsc
    if name == "strided":
        return torch.strided
    raise ValueError("unknown layout descriptor " + repr(name))


def _layout_name(layout):
    for name in ("coo", "csr", "csc", "bsr", "bsc", "strided"):
        if _layout_for(name) == layout:
            return name
    raise ValueError("unknown layout value " + repr(layout))


def _decode_case(case):
    # A bare shape row (an empty row is a scalar) means the schema defaults, and
    # every extent on that path is validated too. Otherwise all five descriptors
    # are validated and kept: an irrelevant or conflicting field raises here
    # instead of being silently dropped, so listing and execution cannot
    # disagree.
    case = tuple(case)
    if all(_plain_int(item) for item in case):
        return tuple(_nonneg_int(size, "shape") for size in case), {}
    if len(case) != 5:
        raise ValueError("case row needs 5 fields, got " + repr(case))
    target = case[0]
    shape = tuple(_nonneg_int(size, "shape") for size in case[1])
    kwargs = {}
    dense_dim = _int_or_none(case[3], "dense_dim")
    if target == "sparse_dim":
        sparse_dim = _int_or_none(case[2], "sparse_dim")
        if sparse_dim is None or sparse_dim > len(shape):
            raise ValueError("sparse_dim out of range for this shape: " + repr(case))
        if dense_dim is not None:
            # The native positional overload is to_sparse(Tensor, int
            # sparse_dim); it takes no dense_dim argument, so the two cannot be
            # combined in one call.
            raise ValueError(
                "sparse_dim and dense_dim cannot be combined: " + repr(case)
            )
        kwargs["sparse_dim"] = sparse_dim
    elif target in ("default", "dense_dim"):
        if case[2] is not None:
            raise ValueError("unexpected sparse_dim for " + target + ": " + repr(case))
    else:
        if case[2] is not None:
            raise ValueError(
                "sparse_dim does not apply to a layout descriptor: " + repr(case)
            )
        kwargs["layout"] = _layout_for(target)
    if dense_dim is not None:
        if dense_dim >= max(len(shape), 1):
            raise ValueError("dense_dim out of range for this shape: " + repr(case))
        kwargs["dense_dim"] = dense_dim
    blocksize = case[4]
    layout = kwargs.get("layout")
    if layout in _COMPRESSED_LAYOUTS:
        core_len = len(shape) - (dense_dim or 0)
        if core_len < 2:
            raise ValueError("compressed layouts need rank >= 2: " + repr(case))
        batches = 1
        for size in shape[: core_len - 2]:
            batches = batches * size
        if batches <= 0:
            # A zero-sized batch dimension cannot carry one stored entry per
            # batch, so the native op rejects it.
            raise ValueError("compressed layouts need a non-empty batch: " + repr(case))
        if layout in _BLOCK_LAYOUTS:
            if blocksize is None:
                raise ValueError(
                    "block layouts need an explicit blocksize: " + repr(case)
                )
            block = tuple(_nonneg_int(size, "blocksize") for size in blocksize)
            if len(block) != 2 or block[0] <= 0 or block[1] <= 0:
                raise ValueError("blocksize must be two positive ints: " + repr(case))
            if shape[core_len - 2] % block[0] or shape[core_len - 1] % block[1]:
                raise ValueError("blocksize must tile the matrix: " + repr(case))
            kwargs["blocksize"] = [block[0], block[1]]
        elif blocksize is not None:
            raise ValueError("blocksize only applies to block layouts: " + repr(case))
    elif blocksize is not None:
        raise ValueError("blocksize only applies to block layouts: " + repr(case))
    return shape, kwargs


def _describe(kwargs):
    # JSON-compatible listing metadata derived from the decoded kwargs.
    desc = {}
    for key in ("sparse_dim", "dense_dim"):
        if key in kwargs:
            desc[key] = int(kwargs[key])
    if "layout" in kwargs:
        desc["layout"] = _layout_name(kwargs["layout"])
    if "blocksize" in kwargs:
        block = kwargs["blocksize"]
        desc["blocksize"] = [int(block[0]), int(block[1])]
    return desc


def _equal_count_source(inp, layout, blocksize, dense_dim):
    # Batched CSR/CSC/BSR/BSC reject an unequal per-batch stored count natively
    # (Expect the same number of specified elements per batch.), and a fixed
    # position mask is unsafe because low precision rounding can produce an
    # exact zero. The mask is therefore derived from the tensor own nonzero
    # pattern: every batch keeps exactly the smallest per-batch count (blocks
    # for the block layouts) at its own positions, so the stored count matches
    # across batches while the pattern differs and a kernel that replicates
    # batch 0 fails accuracy. dense_dim moves the matrix dimensions away from
    # the last two axes, so the dense tail is collapsed first.
    dense_dim = 0 if dense_dim is None else int(dense_dim)
    core_len = inp.dim() - dense_dim
    core_shape = inp.shape[:core_len]
    rows, cols = core_shape[-2], core_shape[-1]
    batch_shape = core_shape[:-2]
    zero = torch.zeros((), dtype=inp.dtype, device=inp.device)
    nonzero = inp != zero
    if dense_dim:
        nonzero = nonzero.any(dim=tuple(range(core_len, inp.dim())))
    if layout in _BLOCK_LAYOUTS:
        grid_rows = rows // blocksize[0]
        grid_cols = cols // blocksize[1]
        blocked = nonzero.reshape(
            batch_shape + (grid_rows, blocksize[0], grid_cols, blocksize[1])
        )
        grid = blocked.any(-1).any(-2)
    else:
        grid_rows, grid_cols = rows, cols
        grid = nonzero.reshape(batch_shape + (grid_rows, grid_cols))
    batches = 1
    for size in batch_shape:
        batches = batches * size
    flat = grid.reshape(batches, grid_rows * grid_cols)
    # The per-batch stored count is bounded by the grid size (at most rows*cols
    # = 1048576 positions for the widest row here) and the running rank is a
    # position count bounded by the same grid, so int32 is the adequate
    # accumulator for both; a wider dtype would only change the temporary dtype.
    keep = int(flat.sum(-1, dtype=torch.int32).min().item())
    rank = torch.cumsum(flat.to(torch.int32), dim=-1, dtype=torch.int32)
    kept = flat & (rank <= keep)
    mask = kept.reshape(batch_shape + (grid_rows, grid_cols))
    if layout in _BLOCK_LAYOUTS:
        mask = mask.repeat_interleave(blocksize[0], -2)
        mask = mask.repeat_interleave(blocksize[1], -1)
    if dense_dim:
        for _ in range(dense_dim):
            mask = mask.unsqueeze(-1)
        mask = mask.expand(inp.shape)
    return torch.where(mask, inp, zero)


def _case_fn(case, dtype):
    del dtype
    shape, kwargs = _decode_case(case)
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params=_describe(kwargs),
        builder_args=(shape, kwargs),
    )


def _build_inputs_fn(plan, dtype, device):
    shape, kwargs = plan.builder_args
    inp = torch.randn(shape, dtype=dtype, device=device)
    layout = kwargs.get("layout")
    dense_dim = kwargs.get("dense_dim") or 0
    core_len = len(shape) - dense_dim
    if layout in _COMPRESSED_LAYOUTS and core_len > 2:
        inp = _equal_count_source(inp, layout, kwargs.get("blocksize"), dense_dim)
    return inp, kwargs


class ToSparseBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        # Always delegate: the parent reads the shape file when one is given and
        # otherwise falls back to the built-in rows, so custom shape files keep
        # working for both listing and execution.
        super().set_shapes(shape_file_path, default_shapes=_BENCH_CASES)


def _benchmark():
    return ToSparseBenchmark(
        op_name="to_sparse",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.to_sparse,
        gems_op=getattr(flag_gems, "to_sparse", None),
        dtypes=_BENCH_DTYPES,
    )


@pytest.mark.to_sparse
def test_to_sparse():
    _benchmark().run()
