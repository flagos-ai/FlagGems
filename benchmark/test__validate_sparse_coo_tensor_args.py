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

# The workload of this validator is a COO descriptor, not a dense shape: each
# entry is (nnz, size, dense_shape) with sparse_dim = len(size) - len(dense_shape).
# The set spans 0-4 sparse dims, a dense payload, zero nnz, a size-0 extent and
# extents that differ by many orders of magnitude, so the timing covers both the
# index scan and the coordinate-ordering checks.
_DESCRIPTORS = [
    (65536, (1024, 1024), ()),
    (262144, (1024, 1024), ()),
    (1048576, (4096, 4096), ()),
    (262144, (256, 256, 256), ()),
    (1048576, (4194304,), ()),
    (131072, (64, 64, 64, 64), ()),
    (0, (1048576,), ()),
    (65536, (1048576, 16), (16,)),
    (0, (1048576, 0), ()),
    (0, (1048576, 0), (0,)),
    # Extent 256 * 256 = 65536 < nnz, so is_coalesced=True (which needs nnz
    # distinct coordinates) is not constructible for this descriptor; the other
    # three variants are timed as usual.
    (262144, (256, 256), ()),
    # Rank-0 sparse descriptors: size and dense_shape have equal length, so there
    # are no coordinates to order and is_coalesced=True holds for every nnz.
    (0, (), ()),
    (1, (3,), (3,)),
    (2, (3, 4), (3, 4)),
]

# Every descriptor above builds its coordinate buffer in int64: the schema
# accepts no other index dtype, so this is a required *operand* storage
# capability rather than a values dtype. It is read once from the static device
# capability descriptor and carried by its own eligibility gate below; nothing is
# probed and nothing is skipped while listing or running.
_INT64_STORAGE_SUPPORTED = bool(flag_gems.runtime.device.support_int64)


def _int64_gated(descriptors):
    # Listing and execution read the same table, so an int64-incapable device
    # claims no COO validator workload rather than a descriptor it cannot build.
    return list(descriptors) if _INT64_STORAGE_SUPPORTED else []


# The descriptors this module actually offers to the benchmark.
_RUNNABLE_DESCRIPTORS = _int64_gated(_DESCRIPTORS)

# The values payload dtype only has to be allocatable -- the validator never
# reads it. bf16 comes from the shared benchmark dtype list and is dropped when
# the device capability descriptor reports no bf16 support; the flag is a static
# descriptor and no probe or skip is involved.
_DTYPES = [
    dtype
    for dtype in consts.FLOAT_DTYPES
    if dtype is not torch.bfloat16 or flag_gems.runtime.device.support_bf16
]

# The omitted schema default, explicit None, False and True are four distinct
# call forms, so they stay separate cases in the listing.
_COALESCED_FLAGS = {"none": None, "false": False, "true": True}
_VARIANTS = ["omitted", "none", "false", "true"]


def _sparse_extents(size, dense):
    """The leading extents of ``size`` that are sparse, i.e. not dense_shape."""
    return tuple(size)[: len(size) - len(dense)]


def _capacity(extents):
    total = 1
    for dim in extents:
        total *= dim
    return total


def _require_valid_descriptor(nnz, size, dense):
    """Representation checks on one descriptor, before any tensor allocation.

    These describe this table's own encoding (nnz and extents are used exactly
    as given, never clamped, and no coordinate is ever substituted); they make
    no claim about what the operator must accept. Only coordinate domains that
    cannot hold the requested nnz are impossible: a zero-length sparse extent
    admits no coordinate at all, while an empty tuple of sparse extents is the
    rank-0 case and holds the single empty coordinate (capacity 1, not 0). A
    zero *dense* extent and nnz = 0 stay representable and stay timed.
    """
    if isinstance(nnz, bool) or not isinstance(nnz, int) or nnz < 0:
        raise ValueError(f"nnz must be a non-negative int, got {nnz!r}")
    for extent in tuple(size) + tuple(dense):
        if isinstance(extent, bool) or not isinstance(extent, int) or extent < 0:
            raise ValueError(f"extents must be non-negative ints, got {extent!r}")
    if len(dense) > len(size):
        raise ValueError("dense_shape cannot be longer than size")
    sparse = _sparse_extents(size, dense)
    if nnz > 0 and any(extent == 0 for extent in sparse):
        # No coordinate of the zero-length extent exists, so this descriptor is
        # not a positive workload the validator could be asked to accept.
        raise ValueError(f"nnz={nnz} cannot be stored in the sparse extents {sparse}")


def _variants(nnz, size, dense):
    """Variants valid for one descriptor, used for both listing and execution.

    is_coalesced=True requires nnz sorted unique coordinates: with no sparse dim
    there is no coordinate to order, otherwise the sparse extent must hold at
    least nnz distinct positions. Impossible True cases are omitted instead of
    shrinking nnz or duplicating coordinates.
    """
    _require_valid_descriptor(nnz, size, dense)
    sparse = _sparse_extents(size, dense)
    if not sparse or _capacity(sparse) >= nnz:
        return list(_VARIANTS)
    return [variant for variant in _VARIANTS if variant != "true"]


def _case_fn(shape, dtype):
    # Two-phase GenericBenchmark: the entries passed in through set_shapes are
    # the COO descriptors, and one BenchmarkCasePlan is emitted per valid
    # is_coalesced variant so no tensor is built while listing cases.
    del dtype
    if not _INT64_STORAGE_SUPPORTED:
        # This generator is the one planner boundary every shape source reaches
        # -- the default table, a caller-supplied shape file, and both the
        # listing and the execution path -- so the eligibility check lives here:
        # an int64-incapable device yields no plan at all instead of a COO
        # descriptor whose coordinate buffer it cannot build.
        return
    nnz, size, dense = shape
    for variant in _variants(nnz, size, dense):
        yield base.BenchmarkCasePlan(
            shape={
                "indices": [len(_sparse_extents(size, dense)), nnz],
                "values": [nnz, *dense],
            },
            params={
                "nnz": nnz,
                "size": list(size),
                "dense_shape": list(dense),
                "is_coalesced": variant,
            },
            builder_args=(nnz, tuple(size), tuple(dense), variant),
        )


def _build_inputs_fn(plan, dtype, device):
    nnz, size, dense, variant = plan.builder_args
    indices = _build_indices(_sparse_extents(size, dense), nnz, variant, device)
    values = utils.generate_tensor_input((nnz,) + dense, dtype, device)
    if variant == "omitted":
        return indices, values, list(size), {}
    return indices, values, list(size), {"is_coalesced": _COALESCED_FLAGS[variant]}


def _build_indices(extents, nnz, variant, device):
    # Coordinate buffers are int64 because the schema accepts no other index
    # dtype. That requirement is carried by the static eligibility gate above,
    # which is why this family is absent rather than skipped on a device whose
    # capability descriptor reports no int64 storage.
    if nnz == 0:
        # Zero stored coordinates: an empty (sparse_dim, 0) matrix is the only
        # coordinate buffer, whichever is_coalesced variant is being timed.
        return torch.empty((len(extents), 0), dtype=torch.int64, device=device)
    if not extents:
        # Rank-0 sparse description: one empty coordinate row per stored column.
        return torch.empty((0, nnz), dtype=torch.int64, device=device)
    if variant != "true":
        return torch.stack(
            [
                torch.randint(0, int(dim), (nnz,), dtype=torch.int64, device=device)
                for dim in extents
            ]
        )
    # Row-major linear positions are already sorted and unique, which is what
    # is_coalesced=True requires; _variants guarantees extent >= nnz here.
    dims = [int(dim) for dim in extents]
    strides = []
    running = 1
    for dim in reversed(dims):
        strides.append(running)
        running *= dim
    strides.reverse()
    step = max(running // nnz, 1)
    linear = torch.arange(nnz, dtype=torch.int64, device=device) * step
    return torch.stack([(linear // strides[i]) % dims[i] for i in range(len(dims))])


class ValidateSparseCooTensorArgsBenchmark(OperatorBenchmark):
    # core_shapes.yaml describes dense tensors and has no entry for a COO
    # validator, so the descriptors above are the defaults; a caller-supplied
    # shape file still overrides them. Either way only descriptors the device
    # can actually build are offered.
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_RUNNABLE_DESCRIPTORS)
        # A caller-supplied shape file replaces the defaults wholesale, so the
        # same operand-storage prerequisite is applied to whatever ends up
        # configured: no shape source may reintroduce an int64 descriptor this
        # device cannot build. On a capable device this is the identity.
        self.shapes = _int64_gated(self.shapes)


@pytest.mark.validate_sparse_coo_tensor_args
def test__validate_sparse_coo_tensor_args():
    bench = ValidateSparseCooTensorArgsBenchmark(
        op_name="_validate_sparse_coo_tensor_args",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._validate_sparse_coo_tensor_args,
        gems_op=getattr(flag_gems, "_validate_sparse_coo_tensor_args", None),
        dtypes=_DTYPES,
    )
    bench.run()
