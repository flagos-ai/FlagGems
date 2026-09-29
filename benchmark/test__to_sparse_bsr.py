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

from . import base, consts, generated_operator_utils

_DEV = flag_gems.runtime.device

# Static capability gate only: the file never probes the native operator to pick
# a dtype, so collection and listing stay free of tensor work.
_BENCH_DTYPES = list(consts.FLOAT_DTYPES)
if not _DEV.support_bf16 and torch.bfloat16 in _BENCH_DTYPES:
    _BENCH_DTYPES.remove(torch.bfloat16)

# Mask indices are small block-grid coordinates, so int32 is adequate; naming
# the dtype keeps the builder off arange's implicit int64 default.
_MASK_DTYPE = torch.int32

_CALL_FORMS = ("default", "none", "omit")

# Default descriptors: (shape, blocksize, dense_dim[, call_form]). Every row is
# native-valid: the sparse part keeps rank >= 2, the blocksize divides its last
# two axes, and a batched sparse part stores the same number of blocks in every
# instance, which the builder guarantees. 'none' passes dense_dim=None and
# 'omit' leaves the argument out; the remaining rows pass the integer.
_BENCH_CASES = (
    ((1024, 1024), (16, 16), 0),
    ((1024, 1024), (1, 1), 0),
    ((20, 320, 15), (4, 5), 0),
    ((20, 320, 15), (2, 3), 0),
    ((16, 128, 64, 60), (4, 4), 0),
    ((16, 128, 64, 60), (2, 2), 1),
    ((16, 7, 57, 32, 29), (1, 1), 0),
    ((256, 256), (8, 8), 0),
    ((512, 1024), (4, 4), 0),
    ((32, 64, 64), (4, 4), 1),
    ((2, 8, 8, 8), (2, 2), 0),
    ((1024, 1024), (1, 1), None, "none"),
    ((1024, 1024), (1, 1), 0, "omit"),
)


def _split_case(case):
    """Normalize one shape-file entry into (shape, blocksize, dense_dim, call_form).

    An entry whose first item is a sequence is an explicit descriptor; anything
    else is a plain shape, which gets the schema defaults (blocksize [1, 1],
    dense_dim 0) so a caller can request any valid custom shape without knowing
    this operator's descriptor syntax. The entry structure is checked before any
    field is consumed, so a malformed descriptor is rejected here instead of
    having fields silently dropped.
    """
    if not isinstance(case, (tuple, list)):
        raise ValueError(f"a case entry must be a sequence, got {case!r}")
    if case and isinstance(case[0], (tuple, list)):
        if len(case) > 4:
            raise ValueError(
                f"a descriptor holds at most 4 fields, got {len(case)} in {case!r}"
            )
        shape = tuple(case[0])
        if len(case) > 1:
            if not isinstance(case[1], (tuple, list)):
                raise ValueError(
                    f"the blocksize field must be a sequence, got {case!r}"
                )
            blocksize = tuple(case[1])
        else:
            blocksize = (1, 1)
        dense_dim = case[2] if len(case) > 2 else 0
        call_form = case[3] if len(case) > 3 else "default"
    else:
        shape, blocksize, dense_dim, call_form = tuple(case), (1, 1), 0, "default"
    return shape, blocksize, dense_dim, call_form


def _dense_extent(dense_dim):
    return 0 if dense_dim is None else dense_dim


def _metadata_validate(shape, blocksize, dense_dim, call_form):
    """Reject descriptors the native operator cannot accept, before tensor work."""
    if call_form not in _CALL_FORMS:
        raise ValueError(f"unknown call_form {call_form!r}")
    if not isinstance(shape, (tuple, list)) or len(shape) < 2:
        raise ValueError(f"shape must be a sequence of rank >= 2, got {shape!r}")
    for dim in shape:
        if isinstance(dim, bool) or not isinstance(dim, int) or dim < 0:
            raise ValueError(f"shape entries must be non-negative ints, got {shape!r}")
    if not isinstance(blocksize, (tuple, list)) or len(blocksize) != 2:
        raise ValueError(f"blocksize must hold two entries, got {blocksize!r}")
    for block in blocksize:
        if isinstance(block, bool) or not isinstance(block, int) or block < 1:
            raise ValueError(
                f"blocksize entries must be positive ints, got {blocksize!r}"
            )
    if call_form == "none":
        if dense_dim is not None:
            raise ValueError(
                f"the none call form requires dense_dim=None, got {dense_dim!r}"
            )
    elif (
        dense_dim is None
        or isinstance(dense_dim, bool)
        or not isinstance(dense_dim, int)
        or dense_dim < 0
    ):
        raise ValueError(f"dense_dim must be a non-negative int, got {dense_dim!r}")
    if call_form == "omit" and dense_dim != 0:
        raise ValueError(f"the omitted argument is dense_dim=0, got {dense_dim!r}")
    if dense_dim is not None and dense_dim > len(shape) - 2:
        raise ValueError(
            "dense_dim must leave at least two sparse axes, so it cannot exceed"
            f" rank - 2 = {len(shape) - 2}, got {dense_dim!r}"
        )
    sparse = tuple(shape[: len(shape) - _dense_extent(dense_dim)])
    if len(sparse) < 2:
        raise ValueError(f"the sparse part must have rank >= 2, got {sparse!r}")
    if sparse[-2] % blocksize[0] or sparse[-1] % blocksize[1]:
        raise ValueError(
            f"blocksize {tuple(blocksize)!r} does not divide {sparse[-2:]!r}"
        )
    batch = sparse[:-2]
    product = 1
    for dim in batch:
        product *= dim
    if batch and product == 0:
        # Native: 'to_sparse_bsr: Expected product of batch dimensions to be
        # non-zero.' A zero sparse or dense extent is valid and is not rejected
        # here; only a zero batch product is.
        raise ValueError(f"a zero batch product is rejected natively, got {batch!r}")
    return sparse


def _masked_input(shape, blocksize, dense_extent, dtype, device):
    """Dense input holding a batch-uniform set of non-zero stored blocks.

    Every batch instance must store the same number of blocks, so one mask is
    reused for all of them. Zero extents inside the sparse part or the dense tail
    are natively valid and yield an all-zero input.
    """
    if any(dim == 0 for dim in shape):
        return torch.zeros(shape, dtype=dtype, device=device)
    sparse = tuple(shape[: len(shape) - dense_extent])
    batch = sparse[:-2]
    block_rows = sparse[-2] // blocksize[0]
    block_cols = sparse[-1] // blocksize[1]
    rows = torch.arange(block_rows, dtype=_MASK_DTYPE, device=device)
    cols = torch.arange(block_cols, dtype=_MASK_DTYPE, device=device)
    mask = cols[None, :] < (1 + rows % block_cols)[:, None]
    grid = mask.reshape((1,) * len(batch) + (block_rows, block_cols)).to(torch.float32)
    grid = grid.repeat_interleave(blocksize[0], dim=-2).repeat_interleave(
        blocksize[1], dim=-1
    )
    blocks = grid.expand(batch + (sparse[-2], sparse[-1]))
    if dense_extent:
        tail = tuple(shape[len(shape) - dense_extent :])
        blocks = blocks.reshape(batch + (sparse[-2], sparse[-1]) + (1,) * dense_extent)
        blocks = blocks.expand(batch + (sparse[-2], sparse[-1]) + tail)
    position = torch.arange(blocks.numel(), dtype=torch.float32, device=device).reshape(
        blocks.shape
    )
    magnitude = (position % 7) + 1
    values = torch.where(position % 2 == 0, magnitude, -magnitude)
    return (blocks * values).contiguous().to(dtype)


def _case_fn(case, dtype):
    del dtype
    shape, blocksize, dense_dim, call_form = _split_case(case)
    _metadata_validate(shape, blocksize, dense_dim, call_form)
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={
            "blocksize": list(blocksize),
            "dense_dim": "omitted" if call_form == "omit" else dense_dim,
        },
        builder_args=(shape, blocksize, dense_dim, call_form),
    )


def _build_inputs_fn(plan, dtype, device):
    shape, blocksize, dense_dim, call_form = plan.builder_args
    inp = _masked_input(shape, blocksize, _dense_extent(dense_dim), dtype, device)
    kwargs = {"blocksize": list(blocksize)}
    if call_form != "omit":
        kwargs["dense_dim"] = dense_dim
    return inp, kwargs


class ToSparseBsrBenchmark(generated_operator_utils.OperatorBenchmark):
    """Benchmark family for the dense to BSR conversion."""

    # The operator has no entry in core_shapes.yaml, so the full descriptor list
    # is the default shape source while a caller supplied shape file still wins.
    # Passing the whole rows keeps every blocksize/dense_dim/call-form variant;
    # a caller supplied shape file may still list plain custom shapes, which
    # _split_case gives the schema defaults.
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_BENCH_CASES)


@pytest.mark.to_sparse_bsr
def test__to_sparse_bsr():
    bench = ToSparseBsrBenchmark(
        op_name="_to_sparse_bsr",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._to_sparse_bsr,
        gems_op=getattr(flag_gems, "_to_sparse_bsr", None),
        dtypes=_BENCH_DTYPES,
    )
    bench.run()
