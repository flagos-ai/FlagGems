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

"""Benchmark for aten::_to_sparse_bsc.

aten::_to_sparse_bsc(self, blocksize, dense_dim=None) -> Tensor folds a dense
tensor into BSC: the sparse matrix occupies the two dims immediately before the
trailing dense_dim value dims, so the leading dims are batch dims. The measured
work is the fold itself, so the reference and the candidate receive the exact
same call, built by the same two-phase case pair that also serves --list-cases:
listing allocates nothing, and --case-id replays exactly the rows a normal run
executes.

Each descriptor is (tensor_shape, blocksize, dense_dim, omit_dense_dim). The omit
flag separates the omitted default from an explicitly passed None, because both
spellings are part of the public surface and the omitted one makes the native
operator read different axes.

The shapes are the performance-relevant rows of the shared shape set (rank-3,
rank-4 and rank-5 plus the 1024x1024 fold). A zero extent inside the sparse dims
is a valid input and stays in the list unchanged, never coerced or truncated:
the sparse axes move with dense_dim, so both a plain and a hybrid degenerately
empty row are listed, and each executes with no stored block. A zero extent in a
real batch dim is refused while the descriptor is validated, because the native
operator rejects a zero product of batch dimensions ('Expected product of batch
dimensions to be non-zero.') and a descriptor that cannot execute must not be
advertised as a case. A caller-supplied shape file runs through the same
validation, so a rectangular block size or a hybrid dense_dim row is accepted
while an unexecutable row reports its own error.

Broadcast and backward do not apply here: the operator has a single tensor
operand and no scalar parameter, and this benchmark measures the forward fold.
"""

import pytest
import torch

import flag_gems

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

# (tensor_shape, blocksize, dense_dim, omit_dense_dim). The block sizes tile the
# sparse dims exactly, so the whole logical matrix is made of whole blocks.
_BENCH_CASES = [
    ((2, 19, 7), (19, 7), None, True),
    ((1024, 1024), (16, 16), None, False),
    ((20, 320, 15), (16, 5), None, False),
    ((16, 128, 64, 60), (16, 15), None, False),
    ((16, 7, 57, 32, 29), (19, 16), 1, False),
    ((0, 4), (1, 1), None, False),
    ((0, 4, 3), (1, 1), 1, False),
]

# Static capability flags, read once at import: no probe and no tensor. The
# operator's result is an int64-indexed compressed structure, so a runtime that
# does not advertise int64 cannot execute the fold and lists no row for it; the
# case list therefore never advertises a workload the same run would reject.
_INT64_SUPPORTED = getattr(flag_gems.runtime.device, "support_int64", True)

_BENCH_DTYPES = (
    [
        dtype
        for dtype in consts.FLOAT_DTYPES
        if dtype != torch.bfloat16 or flag_gems.runtime.device.support_bf16
    ]
    if _INT64_SUPPORTED
    else []
)


def _sparse_axes(rank, dense_dim):
    """The two dims holding the sparse matrix of a rank-``rank`` tensor."""
    value_dims = 0 if dense_dim is None else dense_dim
    return rank - 2 - value_dims, rank - 1 - value_dims


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_descriptor(descriptor):
    """Check one (shape, blocksize, dense_dim, omit) metadata row.

    The row is checked before it is unpacked, so a malformed row reports its own
    problem instead of an unpacking error, and this runs before any tensor
    exists: listing a case allocates nothing. A zero extent inside the sparse
    dims is accepted as given -- a positive block size tiles a zero extent
    exactly -- while an empty batch dimension is refused here, because the native
    operator rejects it and an unexecutable row must not be listed as a case.
    """
    if not isinstance(descriptor, (tuple, list)) or len(descriptor) != 4:
        raise ValueError(
            "each case must be a (shape, blocksize, dense_dim, omit) 4-sequence, "
            f"got {descriptor!r}"
        )
    tensor_shape, blocksize, dense_dim, omit_dense_dim = descriptor
    if not isinstance(tensor_shape, (tuple, list)) or len(tensor_shape) < 2:
        raise ValueError(f"tensor_shape must have rank >= 2, got {tensor_shape!r}")
    if not all(_is_int(extent) and extent >= 0 for extent in tensor_shape):
        raise ValueError(
            f"tensor_shape must hold non-negative ints, got {tensor_shape!r}"
        )
    if not isinstance(blocksize, (tuple, list)) or len(blocksize) != 2:
        raise ValueError(f"blocksize must be two positive ints, got {blocksize!r}")
    if not all(_is_int(extent) and extent > 0 for extent in blocksize):
        raise ValueError(f"blocksize must be two positive ints, got {blocksize!r}")
    if not isinstance(omit_dense_dim, bool):
        raise ValueError(f"omit_dense_dim must be a bool, got {omit_dense_dim!r}")
    valid_dense_dim = dense_dim is None or (
        _is_int(dense_dim) and 0 <= dense_dim <= len(tensor_shape) - 2
    )
    if not valid_dense_dim:
        raise ValueError(
            f"dense_dim must be None or between 0 and rank-2, got {dense_dim!r}"
        )
    if omit_dense_dim and dense_dim is not None:
        raise ValueError("omit_dense_dim only applies to the default dense_dim")
    tensor_shape = tuple(tensor_shape)
    blocksize = tuple(blocksize)
    row_axis, col_axis = _sparse_axes(len(tensor_shape), dense_dim)
    for axis, block in zip((row_axis, col_axis), blocksize):
        if tensor_shape[axis] % block:
            raise ValueError(
                f"blocksize {blocksize} does not tile shape {tensor_shape} "
                f"on axis {axis}"
            )
    # The batch dims are the leading dims above the sparse row axis, which moves
    # with dense_dim, so an empty hybrid extent lands in the sparse dims and is
    # accepted while a genuinely empty batch dim is not.
    batch_extent = 1
    for extent in tensor_shape[:row_axis]:
        batch_extent *= extent
    if batch_extent == 0:
        raise ValueError(
            f"tensor_shape {tensor_shape} has an empty batch dimension; "
            "the native operator rejects a zero product of batch dimensions"
        )
    return tensor_shape, blocksize, dense_dim, omit_dense_dim


def _bench_input(tensor_shape, dtype, device):
    """Deterministic, non-uniform dense input on the runner-supplied device.

    The device is the benchmark device handed over by the runner (the FlagGems
    device), so no backend name is hardcoded here, and the fixture builds one
    float ramp and no index tensor, so no case depends on a dtype-specific
    index construction. Every value lies in [1, 7], so no block is all zero:
    the stored-block count is the same for every batch, which the native
    operator requires, and the measured work stays identical across timing
    samples.
    """
    numel = 1
    for extent in tensor_shape:
        numel *= extent
    ramp = torch.arange(numel, dtype=torch.float32, device=device)
    return (torch.remainder(ramp, 7.0) + 1.0).reshape(tensor_shape).to(dtype)


def _case_fn(descriptor, dtype):
    # Two-phase benchmark: one BenchmarkCasePlan per descriptor keeps the block
    # metadata JSON-compatible and defers all tensor construction to
    # _build_inputs_fn, so listing allocates nothing. The validated values are
    # carried through unchanged, and this same pair drives listing and execution.
    del dtype
    tensor_shape, blocksize, dense_dim, omit_dense_dim = _validate_descriptor(
        descriptor
    )
    yield base.BenchmarkCasePlan(
        shape={"input": tensor_shape},
        params={
            "blocksize": blocksize,
            "dense_dim": dense_dim,
            "omit_dense_dim": omit_dense_dim,
        },
        builder_args=(tensor_shape, blocksize, dense_dim, omit_dense_dim),
    )


def _build_inputs_fn(plan, dtype, device):
    tensor_shape, blocksize, dense_dim, omit_dense_dim = plan.builder_args
    inp = _bench_input(tensor_shape, dtype, device)
    # Schema-named keyword arguments, and the omitted default really omits
    # dense_dim, so reference and candidate see the same call form.
    if omit_dense_dim:
        return inp, {"blocksize": blocksize}
    return inp, {"blocksize": blocksize, "dense_dim": dense_dim}


class _ToSparseBscBenchmark(OperatorBenchmark):
    """Two-phase benchmark over the operator's bespoke shape descriptors.

    ``default_shapes`` keeps the operator-name/class-name lookup of
    OperatorBenchmark.set_shapes while providing these descriptors as the
    fallback, so a caller-supplied shape file still wins when it is present.
    """

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_BENCH_CASES)


@pytest.mark.to_sparse_bsc
def test__to_sparse_bsc():
    bench = _ToSparseBscBenchmark(
        op_name="_to_sparse_bsc",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._to_sparse_bsc,
        gems_op=getattr(flag_gems, "_to_sparse_bsc", None),
        dtypes=_BENCH_DTYPES,
    )
    bench.run()
