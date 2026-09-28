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

import math

import pytest
import torch

import flag_gems

from . import base, utils
from .generated_operator_utils import OperatorBenchmark

# Static capability flags of the active backend; read only, no tensor work.
_SUPPORT_BF16 = flag_gems.runtime.device.support_bf16
_STRUCT_INT64 = flag_gems.runtime.device.support_int64

# Conversion work is driven by the dense input size and by the requested sparse
# layout, so a workload is a (dense shape, layout, blocksize, call form)
# descriptor. The first fifteen entries are the operator's own plan list; the
# hybrid dense_dim forms, the accepted layout spellings, the scalar COO form and
# the positional sparse_dim call form are appended. core_shapes.yaml carries no
# _to_sparse entry, so this list is the default and a _to_sparse shape file still
# overrides it. The sparse_dim form is written 'sparse_dim:N': the
# OpOverloadPacket dispatches that positional argument on the shared call both
# torch_op and gems_op make (verified: _to_sparse(ones(2, 3), 1) gives sparse_dim
# 1 and _to_sparse(ones(()), 0) gives shape [] with sparse_dim 0 and one stored
# element), so no framework bypass is needed.
_TO_SPARSE_DESCRIPTORS = [
    ((1024, 1024), "sparse_coo", None, "default"),
    ((2048, 2048), "sparse_coo", None, "default"),
    ((4096, 4096), "sparse_coo", None, "default"),
    ((256, 256, 256), "sparse_coo", None, "default"),
    ((1024, 1024), "sparse_csr", None, "default"),
    ((2048, 2048), "sparse_csr", None, "default"),
    ((1024, 1024), "sparse_bsr", (2, 2), "default"),
    ((2048, 2048), "sparse_bsr", (4, 4), "default"),
    ((1024, 1024), "sparse_csc", None, "default"),
    ((2048, 2048), "sparse_csc", None, "default"),
    ((1024, 1024), "sparse_bsc", (2, 2), "default"),
    ((2048, 2048), "sparse_bsc", (4, 4), "default"),
    ((2, 1024, 1024), "sparse_csr", None, "default"),
    ((2, 512, 512), "sparse_bsr", (2, 2), "default"),
    ((4, 256, 256), "sparse_coo", None, "default"),
]

_TO_SPARSE_EXTRA_DESCRIPTORS = [
    ((8, 8, 8), "sparse_coo", None, "default:dense_dim=1"),
    ((8, 8, 8), "sparse_coo", None, "default:dense_dim=2"),
    ((64, 64), "sparse_coo", None, "default:dense_dim=1"),
    ((2, 8, 8), "sparse_csr", None, "default:dense_dim=1"),
    ((4, 6, 3), "sparse_bsr", (2, 2), "default:dense_dim=1"),
    ((512, 512), "torch.layout.sparse_csr", None, "default"),
    ((512, 512), "torch.sparse_coo", None, "default"),
    ((), "sparse_coo", None, "default"),
    ((1024, 1024), "sparse_coo", None, "sparse_dim:1"),
    ((2, 3), "sparse_coo", None, "sparse_dim:1"),
    ((), "sparse_coo", None, "sparse_dim:0"),
]

# Explicit canonical allowlist: the layout comes from a descriptor (or the shape
# file) and is never resolved with a free getattr. Only these exact spellings
# are accepted -- the bare name and the two documented torch prefixes -- so an
# arbitrary dotted prefix is not treated as a genuine torch layout name and a
# longer prefix cannot be shadowed by a shorter one.
_LAYOUTS = {
    "sparse_coo": torch.sparse_coo,
    "sparse_csr": torch.sparse_csr,
    "sparse_csc": torch.sparse_csc,
    "sparse_bsr": torch.sparse_bsr,
    "sparse_bsc": torch.sparse_bsc,
}
_BLOCK_LAYOUTS = ("sparse_bsr", "sparse_bsc")
_LAYOUT_SPELLINGS = {
    spelling: name
    for name in _LAYOUTS
    for spelling in (name, f"torch.{name}", f"torch.layout.{name}")
}


def _int64_gated(plans):
    """Plan and dtype lists gated on the backend's int64 allocation support.

    Every plan here converts a dense input into a COO or compressed layout,
    whose index structure is int64 (COO coordinates, crow/col or ccol/row
    pointers), and the batched fixture builds an int64 scatter index. Without
    int64 allocation there is no valid plan, so the default plans and the
    benchmark dtypes are emptied statically instead of listing work that cannot
    run. The same list feeds --list-cases and execution, so both stay identical.
    """
    return list(plans) if _STRUCT_INT64 else []


# FP16/FP32 always construct; bfloat16 only where the backend advertises it.
_BENCH_DTYPES = _int64_gated(
    [torch.float16, torch.float32] + ([torch.bfloat16] if _SUPPORT_BF16 else [])
)


def _canonical_layout(layout):
    """Canonical layout name from an exact, supported spelling only."""
    if not isinstance(layout, str):
        raise ValueError(f"layout must be a string, got {layout!r}")
    if layout not in _LAYOUT_SPELLINGS:
        raise ValueError(f"unknown layout {layout!r}, expected {sorted(_LAYOUTS)}")
    return _LAYOUT_SPELLINGS[layout]


def _canonical_call(call):
    """Canonical call form, as (dense_dim, sparse_dim).

    'default' optionally carries dense_dim=N. 'sparse_dim:N' is the positional
    sparse-dimension form; the explicit 'sparse_dim:sparse_dim=N' spelling is
    accepted too, so the bare value is never mistaken for a key. Metadata only,
    so listing stays allocation-free.
    """
    if not isinstance(call, str) or not call:
        raise ValueError(f"call form must be a non-empty string, got {call!r}")
    kind, _, tail = call.partition(":")
    if kind == "default":
        dense_dim = 0
        if tail:
            for part in tail.split(","):
                key, _, value = part.partition("=")
                if key != "dense_dim":
                    raise ValueError(f"unknown call-form key {key!r}")
                try:
                    dense_dim = int(value)
                except ValueError as exc:
                    raise ValueError(f"dense_dim must be an int: {part!r}") from exc
        if dense_dim < 0:
            raise ValueError(f"dense_dim must be non-negative, got {dense_dim}")
        return dense_dim, None
    if kind == "sparse_dim":
        key, sep, value = tail.partition("=")
        if not sep:
            # Bare 'sparse_dim:N' carries only the value.
            key, value = "sparse_dim", key
        if key != "sparse_dim":
            raise ValueError(f"unknown call-form key {key!r}")
        try:
            return 0, int(value)
        except ValueError as exc:
            raise ValueError(f"sparse_dim must be an int: {call!r}") from exc
    raise ValueError(f"unknown call form {kind!r}")


def _validated_descriptor(descriptor):
    """Canonicalize one case descriptor or raise ValueError.

    The descriptor list and the shape file are metadata only, so every
    constraint is checked before any tensor exists: --list-cases then reports
    exactly the plans that execution builds, and an invalid descriptor is never
    repaired, coerced or silently replaced by a default geometry.
    """
    if not isinstance(descriptor, (list, tuple)) or len(descriptor) != 4:
        raise ValueError(
            f"expected (shape, layout, blocksize, call), got {descriptor!r}"
        )
    shape, layout, blocksize, call = descriptor
    if not isinstance(shape, (list, tuple)):
        raise ValueError(f"shape must be a sequence, got {shape!r}")
    extents = []
    for extent in shape:
        if isinstance(extent, bool) or not isinstance(extent, int) or extent < 0:
            raise ValueError(f"shape extents must be non-negative ints, got {shape!r}")
        extents.append(int(extent))
    extents = tuple(extents)

    name = _canonical_layout(layout)
    rank = len(extents)
    dense_dim, sparse_dim = _canonical_call(call)

    # A dense->block conversion requires a blocksize pair; without it the native
    # kernel rejects the call, so a block layout with blocksize=None is rejected
    # during metadata validation instead of at execution.
    blocks = None
    if name in _BLOCK_LAYOUTS:
        if blocksize is None:
            raise ValueError(f"{name} requires a blocksize pair, got {blocksize!r}")
        if not isinstance(blocksize, (list, tuple)) or len(blocksize) != 2:
            raise ValueError(f"blocksize must be a pair, got {blocksize!r}")
        for extent in blocksize:
            if isinstance(extent, bool) or not isinstance(extent, int) or extent <= 0:
                raise ValueError(
                    f"blocksize entries must be positive ints, got {blocksize!r}"
                )
        blocks = (int(blocksize[0]), int(blocksize[1]))
    elif blocksize is not None:
        raise ValueError(f"blocksize is not valid for {name}")

    # The compressed layouts need the two matrix axes that the dense tail is
    # stripped from; a plain COO conversion only needs a dense_dim below the
    # rank, which keeps the scalar () form valid (native: _to_sparse(ones(()))).
    if name == "sparse_coo":
        if dense_dim and dense_dim >= rank:
            raise ValueError(
                f"dense_dim {dense_dim} must be < rank {rank} for {shape!r}"
            )
    elif rank - dense_dim < 2:
        raise ValueError(
            f"{name} needs a 2-D sparse grid, got rank {rank} with "
            f"dense_dim {dense_dim} for {shape!r}"
        )

    # The block pair applies to the matrix axes, which sit immediately before
    # the dense tail, not to the last axes of the whole shape.
    if blocks is not None:
        matrix = extents[rank - dense_dim - 2 : rank - dense_dim]
        if any(extent % block for extent, block in zip(matrix, blocks)):
            raise ValueError(f"blocksize {blocks!r} must divide {matrix} of {shape!r}")

    # The positional sparse_dim form is a plain COO conversion; its range is
    # rank-dependent (1..rank, or exactly 0 for the scalar input).
    if sparse_dim is not None:
        if name != "sparse_coo" or blocks is not None or dense_dim:
            raise ValueError(
                f"sparse_dim call form needs a plain COO conversion, got {descriptor!r}"
            )
        if rank == 0:
            if sparse_dim != 0:
                raise ValueError(
                    f"sparse_dim {sparse_dim} must be 0 for the scalar input"
                )
        elif not 1 <= sparse_dim <= rank:
            raise ValueError(
                f"sparse_dim {sparse_dim} must be in 1..{rank} for {shape!r}"
            )
    return extents, name, blocks, dense_dim, sparse_dim


def _case_fn(shape, dtype):
    # Metadata only: the descriptor is validated and canonicalized here and no
    # tensor is created, so listing cases works before a candidate exists.
    del dtype
    dense_shape, layout_name, blocks, dense_dim, sparse_dim = _validated_descriptor(
        shape
    )
    yield base.BenchmarkCasePlan(
        shape={"input": list(dense_shape)},
        params={
            "layout": layout_name,
            "blocksize": None if blocks is None else list(blocks),
            "dense_dim": dense_dim,
            "sparse_dim": sparse_dim,
        },
        builder_args=(dense_shape, layout_name, blocks, dense_dim, sparse_dim),
    )


def _window_mask(grid, blocksize, device):
    """Boolean mask with an equal stored-cell count per batch slice.

    The block expansion happens before the zero-cell short circuit, so a shape
    with a zero row or column (or an empty batch) still returns a mask matching
    the dense logical shape and torch.where cannot mismatch. A batched
    compressed conversion rejects an unequal number of specified elements per
    batch ('Expect the same number of specified elements per batch.'), so the
    window count is fixed while its start shifts per slice. The only int64
    tensor is the scatter index, which the API requires; the counter stays int32.
    """
    rows, cols = grid[-2], grid[-1]
    batch = tuple(grid[:-2])
    if blocksize is None:
        br, bc = rows, cols
    else:
        br, bc = rows // blocksize[0], cols // blocksize[1]
    cells = br * bc
    slices = math.prod(batch) if batch else 1
    if cells == 0:
        selected = torch.zeros((slices, br, bc), dtype=torch.bool, device=device)
    else:
        keep = max(cells // 4, 1)
        starts = (torch.arange(slices, dtype=torch.int32, device=device) * keep) % cells
        offsets = torch.arange(keep, dtype=torch.int32, device=device)
        index = ((starts[:, None] + offsets[None, :]) % cells).long()
        selected = torch.zeros((slices, cells), dtype=torch.bool, device=device)
        selected.scatter_(1, index, True)
        selected = selected.reshape((slices, br, bc))
    selected = selected.reshape(batch + (br, bc))
    if blocksize is None:
        return selected
    return selected.repeat_interleave(blocksize[0], -2).repeat_interleave(
        blocksize[1], -1
    )


def _empty_units(masked, mask, blocksize):
    """Selected cells/blocks that hold no nonzero value at all.

    The equal-count rule needs every selected unit to store something, but a
    zero inside a unit that already stores another nonzero value is real data
    and keeps the sampled value; only a unit with no nonzero entry at all is
    rewritten so the per-batch counts stay equal.
    """
    live = masked != 0
    while live.dim() > mask.dim():
        live = live.any(dim=-1)
    if blocksize is not None:
        block_rows, block_cols = blocksize
        rows, cols = live.shape[-2] // block_rows, live.shape[-1] // block_cols
        live = live.reshape(live.shape[:-2] + (rows, block_rows, cols, block_cols))
        live = live.any(dim=-1).any(dim=-2)
        live = live.repeat_interleave(block_rows, -2).repeat_interleave(block_cols, -1)
    return mask & ~live


def _build_inputs_fn(plan, dtype, device):
    dense_shape, layout_name, blocks, dense_dim, sparse_dim = plan.builder_args
    dense = utils.generate_tensor_input(dense_shape, dtype, device)
    grid = dense_shape[: len(dense_shape) - dense_dim]
    # Only a batched compressed grid needs the equal-count window; a 2-D grid
    # keeps the sampled values (zeros included), and so does every COO form.
    if layout_name != "sparse_coo" and len(grid) > 2:
        mask = _window_mask(grid, blocks, device)
        in_unit = mask.reshape(grid + (1,) * dense_dim)
        masked = torch.where(in_unit, dense, torch.zeros_like(dense))
        fill = _empty_units(masked, mask, blocks).reshape(grid + (1,) * dense_dim)
        dense = torch.where(fill, torch.ones((), dtype=dtype, device=device), masked)
    if sparse_dim is not None:
        # The sparse_dim overload is (Tensor self, int sparse_dim) and takes no
        # layout/blocksize/dense_dim keyword: sending layout as well makes the
        # packet reject all four overloads ('expected at most 2 argument(s) but
        # received 3'). This call form therefore passes the value positionally
        # and nothing else, on the same packet both sides use.
        return dense, sparse_dim, {}
    kwargs = {"layout": _LAYOUTS[layout_name]}
    if blocks is not None:
        kwargs["blocksize"] = list(blocks)
    if dense_dim:
        kwargs["dense_dim"] = dense_dim
    return dense, kwargs


class ToSparseBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(
            shape_file_path,
            default_shapes=_int64_gated(
                _TO_SPARSE_DESCRIPTORS + _TO_SPARSE_EXTRA_DESCRIPTORS
            ),
        )


@pytest.mark.to_sparse
def test__to_sparse():
    bench = ToSparseBenchmark(
        op_name="_to_sparse",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._to_sparse,
        gems_op=getattr(flag_gems, "_to_sparse", None),
        dtypes=_BENCH_DTYPES,
    )
    bench.run()
