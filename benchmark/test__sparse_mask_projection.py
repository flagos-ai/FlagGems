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


"""Benchmark cases for ``aten::_sparse_mask_projection``.

The operator intersects two COO sparse tensors, so its cost follows the stored
coordinate count and layout rather than a dense (M, N) grid. Each of the dense
scale shapes below is measured under every combination of

* the density of stored coordinates - the whole ``_BENCH_DENSITIES`` table runs
  for every shape, flag and layout, since the stored count is what the operator
  walks;
* the layout: coalesced without duplicates, uncoalesced with every mask
  coordinate repeated (the only form where ``accumulate_matches`` adds more than
  once), and - for rank > 1 - a hybrid operand with a dense trailing dimension;
* the ``accumulate_matches`` flag, so the two flag values are compared on
  identical inputs;

and only nnz values are allocated, so a large logical shape costs no dense
buffer.
"""

import math

import pytest
import torch

import flag_gems

from . import base
from .generated_operator_utils import OperatorBenchmark

# Dense scale shapes and the original density table. Both densities run for every
# one of these shapes and for every layout and flag.
_ORIGINAL_SHAPES = [(1024, 1024), (20, 320, 15), (16, 128, 64, 60)]
_BENCH_DENSITIES = [0.05, 0.2]
# Supplemental sparse shapes. The rank-0 operand is the scalar case - a native
# COO with ``sparse_dim == 0`` and one unaddressed stored entry - and ``(0,)``
# stores nothing at all. Both are real native forms and both are cheap.
_SUPPLEMENTAL_SHAPES = [(), (0,)]
_DEFAULT_SHAPES = _ORIGINAL_SHAPES + _SUPPLEMENTAL_SHAPES

# A default shape entry is a plain shape, expanded over the whole density table
# and every layout of its rank. A caller-supplied shape file may instead describe
# one plan with a mapping that pins any of these fields: a pinned field collapses
# to that single value while the omitted fields keep their full expansion.
_DESCRIPTOR_KEYS = (
    "shape",
    "density",
    "coalesced",
    "duplicate_factor",
    "sparse_dim",
    "accumulate_matches",
)
_LAYOUT_KEYS = ("coalesced", "duplicate_factor", "sparse_dim")

# Payload dtypes. fp16/fp32/int32 always exist on the target; bf16 and int64 are
# gated on the static device capability flags (the benchmark-side convention).
_ALWAYS_DTYPES = [torch.float16, torch.float32, torch.int32]


def _bench_dtypes():
    # torch normalizes every COO index tensor to int64 and the operator reads
    # those stored indices, so a backend without int64 support cannot build this
    # operator's inputs at all - a family-level structural prerequisite that no
    # payload dtype choice can work around.
    if not flag_gems.runtime.device.support_int64:
        return []
    dtypes = list(_ALWAYS_DTYPES)
    if flag_gems.runtime.device.support_bf16:
        dtypes.append(torch.bfloat16)
    dtypes.append(torch.int64)
    return dtypes


def _valid_shape(shape):
    if not isinstance(shape, (tuple, list)):
        raise ValueError(f"invalid shape {shape!r}")
    for dim in shape:
        if isinstance(dim, bool) or not isinstance(dim, int) or dim < 0:
            raise ValueError(f"invalid shape dimension {dim!r}")
    return tuple(shape)


def _valid_density(density):
    if (
        isinstance(density, bool)
        or not isinstance(density, float)
        or not 0 < density <= 1
    ):
        raise ValueError(f"invalid density {density!r}")
    return density


def _valid_flag(value, name):
    if not isinstance(value, bool):
        raise ValueError(f"invalid {name} {value!r}")
    return value


def _valid_duplicate_factor(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"invalid duplicate_factor {value!r}")
    return value


def _valid_sparse_dim(value, rank):
    # A rank-0 operand really has ``sparse_dim == 0``: its index tensor carries no
    # coordinate row and its single stored entry has no address, so any value in
    # 0 <= sparse_dim <= rank is accepted here.
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= rank:
        raise ValueError(f"invalid sparse_dim {value!r} for rank {rank}")
    return value


def _validate_case(
    shape, density, coalesced, accumulate_matches, duplicate_factor, sparse_dim
):
    """Reject malformed metadata before any tensor is allocated.

    A rank-0 operand (``sparse_dim == 0``) and a zero-extent shape are both valid
    native forms and are accepted; a coalesced COO with repeated coordinates is
    rejected outright, never silently rewritten to a different flag.
    """
    rank = len(_valid_shape(shape))
    _valid_density(density)
    _valid_flag(coalesced, "coalesced")
    _valid_flag(accumulate_matches, "accumulate_matches")
    _valid_duplicate_factor(duplicate_factor)
    _valid_sparse_dim(sparse_dim, rank)
    if coalesced and duplicate_factor > 1:
        raise ValueError("a coalesced COO cannot store repeated coordinates")


def _layout_forms(rank):
    """(coalesced, duplicate_factor, sparse_dim) layouts of one rank.

    A coalesced COO stores one entry per coordinate, so the repeated-coordinate
    layout only exists uncoalesced; the hybrid layout (a dense trailing
    dimension) needs ``sparse_dim < rank``. The base layout of a rank-0 operand is
    ``sparse_dim == rank == 0``, which is the native scalar form.
    """
    forms = [(True, 1, rank), (False, 3, rank)]
    if rank > 1:
        forms.append((True, 1, rank - 1))
    return forms


def _plan_rows(entry):
    """Expand one shape entry into validated ``(shape, density, ...)`` rows."""
    entry = dict(entry) if isinstance(entry, dict) else {"shape": entry}
    unknown = sorted(set(entry) - set(_DESCRIPTOR_KEYS))
    if unknown:
        raise ValueError(f"unknown sparse plan field(s) {unknown}")
    if "shape" not in entry:
        raise ValueError("a sparse plan descriptor requires a 'shape' field")
    shape = _valid_shape(entry["shape"])
    rank = len(shape)

    # Validate every pinned field before it is used, so an invalid value is
    # reported as such instead of silently selecting nothing.
    if "density" in entry:
        _valid_density(entry["density"])
    if "coalesced" in entry:
        _valid_flag(entry["coalesced"], "coalesced")
    if "accumulate_matches" in entry:
        _valid_flag(entry["accumulate_matches"], "accumulate_matches")
    if "duplicate_factor" in entry:
        _valid_duplicate_factor(entry["duplicate_factor"])
    if "sparse_dim" in entry:
        _valid_sparse_dim(entry["sparse_dim"], rank)
    if entry.get("coalesced") and entry.get("duplicate_factor", 1) > 1:
        raise ValueError(
            "a sparse plan cannot request coalesced=True with repeated coordinates"
        )

    densities = [entry["density"]] if "density" in entry else list(_BENCH_DENSITIES)
    flags = (
        [entry["accumulate_matches"]]
        if "accumulate_matches" in entry
        else [False, True]
    )
    if any(key in entry for key in _LAYOUT_KEYS):
        # A pinned layout is exactly the requested one - it is constructed, not
        # filtered out of the default set, so every valid descriptor yields a
        # case instead of silently an empty one.
        layouts = [
            (
                entry.get("coalesced", True),
                entry.get("duplicate_factor", 1),
                entry.get("sparse_dim", rank),
            )
        ]
    else:
        layouts = _layout_forms(rank)

    rows = []
    for density in densities:
        for coalesced, duplicate_factor, sparse_dim in layouts:
            for accumulate_matches in flags:
                _validate_case(
                    shape,
                    density,
                    coalesced,
                    accumulate_matches,
                    duplicate_factor,
                    sparse_dim,
                )
                rows.append(
                    (
                        shape,
                        density,
                        coalesced,
                        accumulate_matches,
                        duplicate_factor,
                        sparse_dim,
                    )
                )
    return rows


def _case_fn(shape, dtype):
    del dtype
    for row in _plan_rows(shape):
        (
            shape_,
            density,
            coalesced,
            accumulate_matches,
            duplicate_factor,
            sparse_dim,
        ) = row
        yield base.BenchmarkCasePlan(
            shape={"self": list(shape_), "mask": list(shape_)},
            params={
                "density": density,
                "coalesced": coalesced,
                "accumulate_matches": accumulate_matches,
                "duplicate_factor": duplicate_factor,
                "sparse_dim": sparse_dim,
            },
            builder_args=row,
        )


def _stored_values(shape, dtype, device):
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, device=device).to(dtype)
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=dtype, device=device)
    return torch.randint(-4, 5, shape, device=device, dtype=dtype)


def _coordinates(lead_shape, positions):
    """Index tensor addressing ``positions`` inside ``lead_shape``.

    A rank-0 operand has ``sparse_dim == 0``, so its index tensor has no
    coordinate row at all (shape ``(0, nnz)``); a one-dimensional leading shape
    needs no unwrapping either.
    """
    if not lead_shape:
        return torch.zeros(
            (0, positions.numel()), dtype=torch.int64, device=positions.device
        )
    if len(lead_shape) == 1:
        return positions.reshape(1, -1)
    return torch.stack(torch.unravel_index(positions, torch.Size(lead_shape)))


def _coo(shape, sparse_dim, positions, values, coalesced):
    index = _coordinates(shape[:sparse_dim], positions)
    return torch.sparse_coo_tensor(
        index, values, torch.Size(shape), is_coalesced=coalesced
    )


def _projection_pair(
    shape, density, duplicate_factor, sparse_dim, coalesced, dtype, device
):
    """A self/mask pair that partially overlaps, plus mask-only coordinates."""
    lead_shape = shape[:sparse_dim]
    tail = tuple(shape[sparse_dim:])
    lead_size = math.prod(lead_shape) if lead_shape else 1
    # A zero-extent leading shape stores nothing at all: nnz is 0 here and there
    # is no stride to derive, so both the position count and the stride are
    # guarded instead of dividing by zero.
    nnz = min(lead_size, max(1, int(lead_size * density))) if lead_size else 0
    step = max(lead_size // nnz, 1) if nnz else 1
    self_positions = torch.arange(nnz, device=device, dtype=torch.int64) * step

    # At least one self entry is mirrored in the mask and the rest are stored
    # nowhere and must come back as zeros; mask-only coordinates are added when
    # the leading shape has room for a distinct position.
    matched = max(1, nnz // 2) if nnz else 0
    mask_unique = self_positions[:matched]
    if nnz and step >= 2:
        mask_only = self_positions[matched:] + step // 2
        mask_unique = torch.cat([mask_unique, mask_only])

    self_values = _stored_values((nnz,) + tail, dtype, device)
    mask_values = _stored_values(
        (mask_unique.numel() * duplicate_factor,) + tail, dtype, device
    )
    mask_positions = (
        mask_unique.repeat(duplicate_factor) if duplicate_factor > 1 else mask_unique
    )
    inp = _coo(shape, sparse_dim, self_positions, self_values, coalesced)
    mask = _coo(shape, sparse_dim, mask_positions, mask_values, coalesced)
    return inp, mask


def _build_inputs_fn(plan, dtype, device):
    (
        shape,
        density,
        coalesced,
        accumulate_matches,
        duplicate_factor,
        sparse_dim,
    ) = plan.builder_args
    inp, mask = _projection_pair(
        shape, density, duplicate_factor, sparse_dim, coalesced, dtype, device
    )
    # The flag travels as a keyword: a bare bool in this tuple is dropped by the
    # framework's unpacking, and a nested tuple would be passed as one operand.
    return inp, mask, {"accumulate_matches": accumulate_matches}


class SparseMaskProjectionBenchmark(OperatorBenchmark):
    # The operator only accepts COO operands, so the dense (M, N) entries in
    # core_shapes.yaml do not describe it; default to the sparse workloads above
    # unless the caller supplies shapes for this operator.
    def set_shapes(self, shape_file_path=None, *, default_shapes=None):
        super().set_shapes(
            shape_file_path, default_shapes=default_shapes or _DEFAULT_SHAPES
        )


@pytest.mark.sparse_mask_projection
def test_sparse_mask_projection():
    bench = SparseMaskProjectionBenchmark(
        op_name="_sparse_mask_projection",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._sparse_mask_projection,
        gems_op=getattr(flag_gems, "_sparse_mask_projection", None),
        dtypes=_bench_dtypes(),
    )
    bench.run()
