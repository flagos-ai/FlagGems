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

# Static capability flag of the active runtime, the same policy tests/accuracy_utils.py
# applies: a bf16 kernel can be missing on some backends. complex64 has its own
# kernels and needs no FP64 support; a complex128 dtype would.
_BF16_SUPPORTED = flag_gems.runtime.device.support_bf16

BENCH_DTYPES = [
    dtype
    for dtype in consts.FLOAT_DTYPES + consts.COMPLEX_DTYPES
    if dtype is not torch.bfloat16 or _BF16_SUPPORTED
]


def _prod(extents):
    size = 1
    for extent in extents:
        size *= extent
    return size


def _blocks_mask(batch_shape, rows, cols, blocks, device):
    """Bool mask marking `blocks` stored (row, col) cells in every batch entry.

    Batch entry b stores the cyclic window starting at b, modulo rows * cols, which
    keeps the per-entry store count equal - the rule native to_sparse_csc enforces -
    while making the stored patterns differ. The positional arithmetic stays in int32
    while the larger of the batch count and the cell count is representable there and
    moves to int64 above that, so the window index never wraps.
    """
    batches = _prod(batch_shape) if batch_shape else 1
    total = rows * cols
    index_dtype = (
        torch.int32
        if max(batches, total) <= torch.iinfo(torch.int32).max
        else torch.int64
    )
    position = torch.arange(total, dtype=index_dtype, device=device)[None, :]
    start = torch.arange(batches, dtype=index_dtype, device=device)[:, None]
    keep = ((position - start) % total) < blocks
    return keep.reshape(*batch_shape, rows, cols)


def _dense_base(dtype, shape, device):
    if dtype.is_complex:
        return torch.randn(shape, dtype=dtype, device=device)
    return utils.generate_tensor_input(shape, dtype, device)


def _nonzero_value(dtype, device):
    """A representable nonzero value for `dtype` (every benchmark dtype has one)."""
    if dtype.is_complex:
        return torch.tensor(complex(0.5, 0.5), dtype=dtype, device=device)
    if dtype.is_floating_point:
        return torch.full((), 0.5, dtype=dtype, device=device)
    return torch.full((), 1, dtype=dtype, device=device)


def _sparse_blocks(dtype, shape, dense_dim, device, density=4):
    """Dense input with equal CSC-stored block counts in every batch entry.

    dense_dim = k splits `shape` into the batch extents shape[:rank-2-k], the
    matrix shape[rank-2-k:rank-k] and the dense tail shape[rank-k:]; native
    to_sparse_csc rejects inputs whose batch entries do not all store the same
    number of cells. Kept cells that the payload left at zero are pinned to a
    representable nonzero because a cell counts as stored only when one of its
    dense-tail values is nonzero. A shape with a zero-sized matrix, batch or dense
    extent stores nothing and is returned as the plain input.
    """
    base = _dense_base(dtype, shape, device)
    dense = 0 if dense_dim is None else dense_dim
    rank = len(shape)
    rows = shape[rank - 2 - dense]
    cols = shape[rank - 1 - dense]
    batch_shape = shape[: rank - 2 - dense]
    dense_shape = shape[rank - dense :]
    if rows * cols == 0 or _prod(batch_shape) == 0 or _prod(dense_shape) == 0:
        return base
    blocks = max(1, (rows * cols) // density)
    keep = _blocks_mask(batch_shape, rows, cols, blocks, device)
    if dense_shape:
        keep = keep.reshape(*batch_shape, rows, cols, *([1] * len(dense_shape))).expand(
            shape
        )
    zeros = torch.zeros((), dtype=dtype, device=device)
    inp = torch.where(keep, base, zeros)
    return torch.where(keep & inp.eq(0), _nonzero_value(dtype, device), inp)


# Flat, JSON-compatible case metadata; the actual shapes are rebuilt by
# _build_inputs_fn, so --list-cases allocates no tensor and runs no operator.
DEFAULT_PLANS = [
    {"shape": [4096, 64], "dense_dim": None, "layout": "contiguous"},
    {"shape": [1024, 1024], "dense_dim": None, "layout": "contiguous"},
    {"shape": [512, 512], "dense_dim": None, "layout": "contiguous"},
    {"shape": [256, 256], "dense_dim": None, "layout": "transpose"},
    {"shape": [512, 256], "dense_dim": None, "layout": "transpose"},
    {"shape": [128, 256], "dense_dim": None, "layout": "contiguous"},
    {"shape": [64, 64], "dense_dim": None, "layout": "contiguous"},
    {"shape": [32, 32], "dense_dim": None, "layout": "contiguous"},
    {"shape": [16, 16], "dense_dim": None, "layout": "contiguous"},
    {"shape": [8, 64, 64], "dense_dim": None, "layout": "contiguous"},
    {"shape": [8, 64, 64], "dense_dim": 0, "layout": "contiguous"},
    {"shape": [8, 64, 64], "dense_dim": 1, "layout": "contiguous"},
    {"shape": [4, 32, 32, 8], "dense_dim": None, "layout": "contiguous"},
    {"shape": [4, 32, 32, 8], "dense_dim": 0, "layout": "contiguous"},
    {"shape": [4, 32, 32, 8], "dense_dim": 1, "layout": "contiguous"},
    {"shape": [4, 32, 32, 8], "dense_dim": 2, "layout": "contiguous"},
]

_ALLOWED_KEYS = {"shape", "dense_dim", "layout"}
_LAYOUTS = ("contiguous", "transpose")


def _descriptor(spec):
    """Normalize one shape-file entry into the descriptor consumed by the plan.

    Entries are either a bare shape list/tuple or a dict with the keys shape,
    dense_dim and layout. Unknown keys are rejected so a mistyped field (for
    example dense_dims) cannot silently fall back to the default.
    """
    if isinstance(spec, (list, tuple)):
        shape, dense_dim, layout = list(spec), None, "contiguous"
    elif isinstance(spec, dict):
        unknown = sorted(set(spec) - _ALLOWED_KEYS)
        if unknown:
            raise ValueError(
                f"to_sparse_csc benchmark entry has unknown key(s) {unknown}: {spec!r}"
            )
        if "shape" not in spec:
            raise ValueError(f"to_sparse_csc benchmark entry needs a shape: {spec!r}")
        shape = spec["shape"]
        dense_dim = spec.get("dense_dim")
        layout = spec.get("layout", "contiguous")
    else:
        raise TypeError(
            f"to_sparse_csc benchmark entry must be a shape list or a dict, got {spec!r}"
        )

    if (
        not isinstance(shape, (list, tuple))
        or len(shape) < 2
        or not all(type(extent) is int and extent >= 0 for extent in shape)
    ):
        raise ValueError(
            f"to_sparse_csc benchmark shape must be at least 2 non-negative ints: {spec!r}"
        )
    shape = [int(extent) for extent in shape]

    if dense_dim is not None and type(dense_dim) not in (int, bool):
        raise ValueError(f"to_sparse_csc dense_dim must be an int: {spec!r}")
    dense = 0 if dense_dim is None else int(dense_dim)
    if not 0 <= dense <= len(shape) - 2:
        raise ValueError(
            f"to_sparse_csc dense_dim must be in [0, {len(shape) - 2}]: {spec!r}"
        )

    # Native splits the shape into batch / matrix / dense tail and rejects only a
    # zero product over the batch extents; a zero matrix or dense-tail extent yields
    # an empty CSC tensor and is valid, mirroring the empty-extent family in
    # tests/test_to_sparse_csc.py.
    if _prod(shape[: len(shape) - 2 - dense]) == 0:
        raise ValueError(
            f"to_sparse_csc benchmark shape has a zero batch product: {spec!r}"
        )

    if layout not in _LAYOUTS:
        raise ValueError(f"to_sparse_csc layout must be one of {_LAYOUTS}: {spec!r}")
    if layout == "transpose" and len(shape) != 2:
        raise ValueError(
            f"to_sparse_csc transpose layout needs a rank-2 shape: {spec!r}"
        )

    return {
        "shape": shape,
        "dense_dim": None if dense_dim is None else dense,
        "layout": layout,
    }


def _case_fn(descriptor, _dtype):
    shape = descriptor["shape"]
    dense_dim = descriptor["dense_dim"]
    layout = descriptor["layout"]
    yield base.BenchmarkCasePlan(
        shape={"input": shape, "dense_dim": dense_dim, "layout": layout},
        params={"dense_dim": dense_dim, "layout": layout},
        builder_args=(shape, dense_dim, layout),
    )


def _build_inputs_fn(plan, dtype, device):
    shape, dense_dim, layout = plan.builder_args
    if layout == "transpose":
        # Build the parent with the transposed extents so the tensor handed to the
        # operator really has the requested (rows, cols) shape; a non-square
        # transposed matrix is a valid rank-2 workload.
        parent = _sparse_blocks(
            dtype, [shape[1], shape[0]], dense_dim, device, density=1
        )
        inp = parent.transpose(0, 1)
    else:
        inp = _sparse_blocks(dtype, shape, dense_dim, device)
    return inp, {"dense_dim": dense_dim}


class ToSparseCscBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        # The shared resolver owns the file handling, so a missing or explicit bad
        # shape file behaves exactly as for every other operator; the resolved
        # entries are then normalized by the same _descriptor as the built-in
        # plans. A rectangular transpose plan cannot be written as a bare `shapes:`
        # entry, because a YAML list has no layout field; it is expressed as
        # {'shape': [512, 256], 'layout': 'transpose'}.
        super().set_shapes(shape_file_path, default_shapes=DEFAULT_PLANS)
        self.shapes = [_descriptor(spec) for spec in self.shapes]


@pytest.mark.to_sparse_csc
def test_to_sparse_csc_benchmark():
    bench = ToSparseCscBenchmark(
        op_name="to_sparse_csc",
        torch_op=torch.ops.aten.to_sparse_csc,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        gems_op=getattr(flag_gems, "to_sparse_csc", None),
        dtypes=BENCH_DTYPES,
    )
    bench.run()
