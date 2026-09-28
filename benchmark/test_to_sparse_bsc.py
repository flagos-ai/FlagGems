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

from . import base, consts, utils
from .generated_operator_utils import OperatorBenchmark

_RANDOM = "random"
_PRUNED = "block_rows_pruned"
_TILED = "tiled_batches"

# (blocksize, dense_dim, input pattern) per benchmark shape. Every entry is a
# native-valid callform: each requested pattern writes one in-range nonzero into
# every block (see _anchor_blocks), so a row stores the same block count in every
# batch as the batched form requires, and the tiled rows repeat one anchored tile
# along the batch dims. A payload alone cannot establish that: an integer payload
# may hit its zero bound and a block of any dtype may come out all-zero.
_SPLITS = {
    (256, 256): [((4, 4), 0, _RANDOM)],
    (1024, 1024): [((4, 4), 0, _RANDOM), ((4, 4), 0, _PRUNED)],
    (2048, 4096): [((4, 4), 0, _RANDOM), ((4, 8), 0, _RANDOM)],
    (64, 256, 256): [((4, 4), 1, _RANDOM)],
    (64, 512, 512): [((4, 4), 1, _RANDOM), ((4, 4), 0, _TILED)],
    (4, 256, 256, 64): [((4, 4), 1, _TILED)],
}
_BSC_SHAPES = list(_SPLITS)

# Integer payloads are generated on the benchmark device; the shared generator
# builds its integer tensors on the CPU and copies them over.
_INT_BOUNDS = {
    torch.int8: (-8, 8),
    torch.uint8: (0, 16),
    torch.int16: (-16, 16),
    torch.int32: (-1024, 1024),
    torch.int64: (-1024, 1024),
}
_BENCH_DTYPES = (
    [
        dtype
        for dtype in consts.FLOAT_DTYPES
        if dtype is not torch.bfloat16 or flag_gems.runtime.device.support_bf16
    ]
    + list(consts.INT_DTYPES)
    + [
        dtype
        for dtype in consts.EXTRA_INT_DTYPES
        if dtype is not torch.int64 or flag_gems.runtime.device.support_int64
    ]
)


def _block_row_axis(rank, dense_dim):
    """Axis of the sparse-matrix row dimension (start of the sparse footprint).

    An omitted dense_dim is the schema default 0; only the call kwargs keep the
    omitted form, so the fixture arithmetic always uses the effective value.
    """
    return rank - (0 if dense_dim is None else dense_dim) - 2


def _validate(shape, blocksize, dense_dim, pattern):
    """Reject a descriptor that native to_sparse_bsc cannot accept.

    Runs before any metadata is coerced to int, so a float, bool or negative
    extent in a caller-supplied descriptor fails loudly instead of being
    silently truncated.
    """
    if len(shape) < 2:
        raise ValueError(f"to_sparse_bsc needs rank >= 2, got shape {tuple(shape)}")
    if len(blocksize) != 2:
        raise ValueError(f"blocksize must be a pair, got {tuple(blocksize)}")
    for extent in tuple(shape) + tuple(blocksize):
        if isinstance(extent, bool) or not isinstance(extent, int) or extent < 0:
            raise ValueError(f"extent {extent!r} is not a non-negative int")
    if blocksize[0] == 0 or blocksize[1] == 0:
        raise ValueError(f"blocksize {tuple(blocksize)} needs positive extents")
    if dense_dim is not None:
        if isinstance(dense_dim, bool) or not isinstance(dense_dim, int):
            raise ValueError(f"dense_dim {dense_dim!r} is neither an int nor None")
        if not 0 <= dense_dim <= len(shape) - 2:
            raise ValueError(
                f"dense_dim {dense_dim} is outside [0, {len(shape) - 2}] for {tuple(shape)}"
            )
    row_axis = _block_row_axis(len(shape), dense_dim)
    if shape[row_axis] % blocksize[0] or shape[row_axis + 1] % blocksize[1]:
        raise ValueError(f"blocksize {tuple(blocksize)} must divide {tuple(shape)}")
    # an empty matrix or dense tail is legal, but a zero-sized batch axis cannot
    # hold the uniform stored-block count the batched form requires
    batch = shape[:row_axis]
    if batch and 0 in batch:
        raise ValueError(
            f"a zero-sized batch axis is not a valid input: {tuple(shape)}"
        )
    if pattern not in (_RANDOM, _PRUNED, _TILED):
        raise ValueError(f"unknown input pattern {pattern!r}")


def _derived_splits(shape):
    """Split for a shape that came from a user shape file."""
    blocksize = next(
        (
            candidate
            for candidate in ((4, 4), (2, 2))
            if all(size % block == 0 for size, block in zip(shape[:2], candidate))
        ),
        (1, 1),
    )
    return [(blocksize, len(shape) - 2, _RANDOM)]


def _splits_for(shape):
    return _SPLITS.get(tuple(shape)) or _derived_splits(tuple(shape))


def _is_descriptor(entry):
    """A descriptor carries its own shape, blocksize, dense_dim and pattern."""
    if not isinstance(entry, (list, tuple)) or len(entry) != 4:
        return False
    shape, blocksize, _, pattern = entry
    return (
        isinstance(shape, (list, tuple))
        and isinstance(blocksize, (list, tuple))
        and isinstance(pattern, str)
    )


def _case_rows(entry):
    """Resolve one case entry into explicit (shape, blocksize, dense_dim, pattern) rows.

    A plain shape uses the curated split table, or a derived split when it came
    from a user shape file; a full descriptor is passed through unchanged so a
    caller can request an exact split.
    """
    if _is_descriptor(entry):
        shape, blocksize, dense_dim, pattern = entry
        return [(tuple(shape), tuple(blocksize), dense_dim, pattern)]
    if not isinstance(entry, (list, tuple)):
        raise ValueError(f"case entry {entry!r} is neither a shape nor a descriptor")
    shape = tuple(entry)
    return [
        (shape, blocksize, dense_dim, pattern)
        for blocksize, dense_dim, pattern in _splits_for(shape)
    ]


def _case_fn(shape, dtype):
    # The case descriptors are dtype independent; the framework pairs them with
    # every benchmark dtype.
    del dtype
    for case_shape, blocksize, dense_dim, pattern in _case_rows(shape):
        _validate(case_shape, blocksize, dense_dim, pattern)
        yield base.BenchmarkCasePlan(
            shape={"input": [int(size) for size in case_shape]},
            params={
                "blocksize": [int(block) for block in blocksize],
                "dense_dim": None if dense_dim is None else int(dense_dim),
                "input_pattern": pattern,
            },
            builder_args=(case_shape, blocksize, dense_dim, pattern),
        )


def _dense_input(shape, dtype, device):
    if dtype in _INT_BOUNDS:
        low, high = _INT_BOUNDS[dtype]
        return torch.randint(low, high, shape, dtype=dtype, device=device)
    return utils.generate_tensor_input(shape, dtype, device)


def _anchor_blocks(inp, blocksize, dense_dim):
    """Write one in-range nonzero into every block of the sparse footprint.

    The payload is not trusted to fill the blocks: an integer payload may hit its
    zero bound and a block may come out all-zero, which the operator prunes and
    which would break the equal-stored-block requirement of the batched form.
    """
    row_axis = _block_row_axis(inp.dim(), dense_dim)
    bs0, bs1 = blocksize
    shape = inp.shape
    blocks = (
        shape[:row_axis]
        + (shape[row_axis] // bs0, bs0, shape[row_axis + 1] // bs1, bs1)
        + shape[row_axis + 2 :]
    )
    index = [slice(None)] * len(blocks)
    index[row_axis + 1] = 0
    index[row_axis + 3] = 0
    inp.view(blocks)[tuple(index)] = 1


def _prune_block_rows(inp, blocksize, dense_dim):
    """Zero whole block rows of the declared sparse axis, block-aligned."""
    row_axis = _block_row_axis(inp.dim(), dense_dim)
    start = (inp.size(row_axis) // blocksize[0] // 2) * blocksize[0]
    inp.narrow(row_axis, start, inp.size(row_axis) - start).zero_()


def _build_inputs_fn(plan, dtype, device):
    shape, blocksize, dense_dim, pattern = plan.builder_args
    row_axis = _block_row_axis(len(shape), dense_dim)
    if pattern == _TILED:
        tile = _dense_input(shape[row_axis:], dtype, device)
        _anchor_blocks(tile, blocksize, dense_dim)
        inp = tile.repeat(*shape[:row_axis], *([1] * (len(shape) - row_axis)))
    else:
        inp = _dense_input(shape, dtype, device)
        _anchor_blocks(inp, blocksize, dense_dim)
        if pattern == _PRUNED:
            _prune_block_rows(inp, blocksize, dense_dim)
    kwargs = {"blocksize": list(blocksize)}
    if dense_dim is not None:
        kwargs["dense_dim"] = dense_dim
    return inp, kwargs


class ToSparseBscBenchmark(OperatorBenchmark):
    """Block-sparse relayout benchmark with the curated scales as default shapes.

    A shape file that names this operator or this class still overrides them
    through the shared resolver.
    """

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_BSC_SHAPES)


@pytest.mark.to_sparse_bsc
def test_to_sparse_bsc():
    bench = ToSparseBscBenchmark(
        op_name="to_sparse_bsc",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.to_sparse_bsc,
        gems_op=getattr(flag_gems, "to_sparse_bsc", None),
        dtypes=_BENCH_DTYPES,
    )
    bench.run()
