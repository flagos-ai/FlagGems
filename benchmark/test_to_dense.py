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

"""Benchmarks for ``aten::to_dense``.

The strided form is a metadata-only identity; the sparse form writes a dense
output whose cost depends on the logical shape, the layout and the stored entry
count.  ``core_shapes.yaml`` only holds dense shapes, so each workload is
described by a descriptor that carries the logical shape plus the sparse
metadata, and ``set_shapes`` hands those descriptors to the framework as this
operator's default plan (a ``--shape-file`` entry for ``to_dense`` still
overrides them).

A descriptor is a 9-field tuple
``(shape, layout, sparse_dim, nnz, coordinates, coalesced, masked_grad,
out_dtype, transposed)``:

``shape``        logical dense shape; ``layout`` is ``"coo"``/``"csr"``/
                 ``"strided"``.
``sparse_dim``   number of sparse dimensions; ``shape[sparse_dim:]`` is the
                 dense tail (empty tail = fully sparse, ``sparse_dim ==
                 len(shape)``).
``nnz``          number of *stored* entries.  Index construction below makes
                 this exact, so the listed metadata reports the same
                 ``stored_nnz`` that the input builder materializes.
``coordinates``  ``"distinct"`` (deterministic distinct coordinates, so no
                 summation happens), ``"duplicates"`` (every entry sits at one
                 coordinate and the values are summed) or ``"random"``
                 (independent random coordinates inside the sparse dims).
``coalesced``    build the COO tensor coalesced; only valid with distinct
                 coordinates, because a merged tensor would store fewer
                 entries than ``nnz``.  Must be ``None`` for a ``csr`` case: the
                 compressed constructor takes no such flag, so accepting one
                 would silently ignore requested metadata.
``masked_grad``  forward the argument to the op (``None`` leaves it out).
``out_dtype``    dtype argument; strided cases only, since ATen rejects
                 ``dtype`` for a sparse input.
``transposed``   build the strided input as a non-contiguous view whose logical
                 shape is exactly the listed shape.

Fields that do not apply are ``None`` and are dropped from the JSON ``params``.
Every descriptor is validated while it is listed, so an impossible or
self-contradictory workload is reported by ``--list-cases`` instead of failing
later inside the input builder.

Structural index dtype: stored COO coordinates are int64 (``torch.
sparse_coo_tensor`` normalizes the coordinate tensors it is given to int64, and
``Tensor.to_sparse`` emits int64 coordinates), so COO descriptors only run
where ``flag_gems.runtime.device.support_int64`` reports int64.  The CSR
row/column pointers are assembled from python ints at int64 when it is
supported and at int32 otherwise, so the compressed families never need int64;
a descriptor whose column count or stored-entry count does not fit that width
is rejected instead of building a silently overflowing pointer tensor.  A
strided descriptor needs only its value dtype.
"""

import os

import pytest
import torch

import flag_gems

from . import base, consts, utils
from .generated_operator_utils import OperatorBenchmark

_CASE_FIELDS = (
    "shape",
    "layout",
    "sparse_dim",
    "nnz",
    "coordinates",
    "coalesced",
    "masked_grad",
    "out_dtype",
    "transposed",
)
_LAYOUTS = ("coo", "csr", "strided")
_COORDINATES = ("distinct", "duplicates", "random")
_DTYPE_NAMES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
# One stored entry per hundred logical elements, applied per shape.
_SPARSE_DENSITY = 1 / 100
# The five large dense shapes this operator is dispatched for.
_STRIDED_SHAPES = (
    (1048576,),
    (1024, 1024),
    (4096, 4096),
    (20, 320, 15),
    (16, 128, 64, 60),
)
# A sparse entry can be much larger than a dense element, so the hybrid cases
# keep few stored entries.
_HYBRID_NNZ = 8


def _case(
    shape,
    layout,
    *,
    sparse_dim=None,
    nnz=None,
    coordinates=None,
    coalesced=None,
    masked_grad=None,
    out_dtype=None,
    transposed=None,
):
    return (
        tuple(shape),
        layout,
        sparse_dim,
        nnz,
        coordinates,
        coalesced,
        masked_grad,
        out_dtype,
        transposed,
    )


def _density_nnz(shape):
    """Stored entries for the dense-ish sparse shapes."""
    numel = 1
    for dim in shape:
        numel *= dim
    return max(1, round(numel * _SPARSE_DENSITY))


def _planned_cases():
    cases = []
    for shape in _STRIDED_SHAPES:
        cases.append(_case(shape, "strided"))
    # Square and rectangular transposes: the built tensor keeps the listed
    # logical shape while being a non-contiguous view.
    cases.append(_case((4096, 4096), "strided", transposed=True))
    cases.append(_case((2048, 4096), "strided", transposed=True))
    cases.append(_case((1024, 1024), "strided", out_dtype="float32"))
    cases.append(_case((1024, 1024), "strided", out_dtype="bfloat16"))
    cases.append(_case((1024, 1024), "strided", masked_grad=True))
    # Sparse COO at the dense-ish densities: random coordinates keep the
    # duplicate-summation path in the measurement.
    for shape in _STRIDED_SHAPES:
        cases.append(
            _case(
                shape,
                "coo",
                sparse_dim=len(shape),
                nnz=_density_nnz(shape),
                coordinates="random",
            )
        )
    for shape in ((1024, 1024), (4096, 4096)):
        cases.append(
            _case(
                shape,
                "coo",
                sparse_dim=2,
                nnz=_density_nnz(shape),
                coordinates="distinct",
                coalesced=True,
            )
        )
    cases.append(
        _case(
            (1024, 1024),
            "coo",
            sparse_dim=2,
            nnz=_density_nnz((1024, 1024)),
            coordinates="distinct",
            coalesced=True,
            masked_grad=True,
        )
    )
    cases.append(
        _case(
            (1024, 1024),
            "coo",
            sparse_dim=2,
            nnz=_density_nnz((1024, 1024)),
            coordinates="distinct",
            coalesced=True,
            masked_grad=False,
        )
    )
    cases.append(
        _case((1024, 1024), "coo", sparse_dim=2, nnz=0, coordinates="distinct")
    )
    # Fully sparse rank-3 (no dense tail) and a genuine hybrid with a dense
    # tail of shape[1:] per stored entry.
    cases.append(
        _case((5, 4096, 4096), "coo", sparse_dim=3, nnz=16384, coordinates="random")
    )
    cases.append(
        _case(
            (1024, 64, 64), "coo", sparse_dim=1, nnz=_HYBRID_NNZ, coordinates="distinct"
        )
    )
    cases.append(
        _case((1024, 64, 64), "coo", sparse_dim=1, nnz=4, coordinates="duplicates")
    )
    # Scalar sparse tensors: sparse_dim == 0 with a single entry, with two
    # entries stored at the same (only) coordinate, and with two random
    # entries.  The random form has no coordinates to draw, so it must build a
    # correctly shaped empty index tensor.
    cases.append(_case((), "coo", sparse_dim=0, nnz=1, coordinates="distinct"))
    cases.append(_case((), "coo", sparse_dim=0, nnz=2, coordinates="duplicates"))
    cases.append(_case((), "coo", sparse_dim=0, nnz=2, coordinates="random"))
    # An empty coordinate domain stores nothing; the random form again needs an
    # empty index tensor instead of randint over a zero extent.
    cases.append(_case((0, 3), "coo", sparse_dim=2, nnz=0, coordinates="random"))
    # CSR: the compressed row pointers make the row/column lookup the cost.
    cases.append(
        _case((4096, 4096), "csr", sparse_dim=2, nnz=65536, coordinates="distinct")
    )
    cases.append(
        _case((4096, 4096), "csr", sparse_dim=2, nnz=262144, coordinates="distinct")
    )
    cases.append(
        _case((1024, 1024), "csr", sparse_dim=2, nnz=65536, coordinates="distinct")
    )
    # A compressed hybrid: the row pointers address the sparse block and the
    # dense tail multiplies the written payload.
    cases.append(
        _case((4096, 512, 8), "csr", sparse_dim=2, nnz=4096, coordinates="distinct")
    )
    return cases


def _out_dtype_supported(name):
    if name == "bfloat16":
        return flag_gems.runtime.device.support_bf16
    return True


def _index_dtype():
    """Structural width for the compressed CSR pointers."""
    if flag_gems.runtime.device.support_int64:
        return torch.int64
    return torch.int32


def _pointer_limit():
    """Largest pointer value the supported structural width can represent."""
    return 2**63 - 1 if flag_gems.runtime.device.support_int64 else 2**31 - 1


def _sparse_dims(shape, sparse_dim):
    return tuple(shape[:sparse_dim])


def _sparse_dim_numel(shape, sparse_dim):
    numel = 1
    for dim in _sparse_dims(shape, sparse_dim):
        numel *= dim
    return numel


def _validated_case(descriptor):
    """Check one descriptor and return it as a field dict."""
    if not isinstance(descriptor, (tuple, list)) or len(descriptor) != len(
        _CASE_FIELDS
    ):
        raise ValueError(
            f"a to_dense case must be a {len(_CASE_FIELDS)}-field tuple, got {descriptor!r}"
        )
    case = dict(zip(_CASE_FIELDS, descriptor))
    shape = case["shape"]
    if not isinstance(shape, (tuple, list)) or not all(
        isinstance(dim, int) and not isinstance(dim, bool) and dim >= 0 for dim in shape
    ):
        raise ValueError(f"to_dense case shape needs non-negative ints, got {shape!r}")
    case["shape"] = tuple(shape)
    if case["layout"] not in _LAYOUTS:
        raise ValueError(f"unknown to_dense case layout {case['layout']!r}")
    for name in ("coalesced", "masked_grad", "transposed"):
        if case[name] is not None and not isinstance(case[name], bool):
            raise ValueError(
                f"to_dense case {name} must be a bool or None, got {case[name]!r}"
            )
    if case["out_dtype"] is not None and case["out_dtype"] not in _DTYPE_NAMES:
        raise ValueError(f"unknown to_dense case out_dtype {case['out_dtype']!r}")
    if case["layout"] == "strided":
        for name in ("sparse_dim", "nnz", "coordinates", "coalesced"):
            if case[name] is not None:
                raise ValueError(f"a strided to_dense case carries no {name} metadata")
        if case["transposed"] and len(case["shape"]) != 2:
            raise ValueError("a transposed strided to_dense case needs a 2-D shape")
        return case
    if case["transposed"] or case["out_dtype"] is not None:
        raise ValueError("transposed/out_dtype only apply to strided cases")
    sparse_dim = case["sparse_dim"]
    if (
        not isinstance(sparse_dim, int)
        or isinstance(sparse_dim, bool)
        or not 0 <= sparse_dim <= len(case["shape"])
    ):
        raise ValueError(
            f"a sparse to_dense case needs 0 <= sparse_dim <= rank, got {sparse_dim!r}"
        )
    nnz = case["nnz"]
    if not isinstance(nnz, int) or isinstance(nnz, bool) or nnz < 0:
        raise ValueError(
            f"a sparse to_dense case needs a non-negative nnz, got {nnz!r}"
        )
    if case["coordinates"] not in _COORDINATES:
        raise ValueError(f"unknown to_dense case coordinates {case['coordinates']!r}")
    # A zero *sparse* coordinate domain cannot hold the requested entries, but a
    # zero *dense* tail is a valid geometry: COO (3, 0) with sparse_dim 1 and
    # nnz 2 builds and densifies to a (3, 0) tensor (measured).
    if nnz and any(dim == 0 for dim in case["shape"][:sparse_dim]):
        raise ValueError(
            f"sparse dims of {case['shape']} store no entries, but nnz={nnz}"
        )
    if case["coordinates"] == "distinct" and nnz > _sparse_dim_numel(
        case["shape"], sparse_dim
    ):
        raise ValueError(
            f"shape {case['shape']} with sparse_dim={sparse_dim} has "
            f"{_sparse_dim_numel(case['shape'], sparse_dim)} distinct coordinates, but nnz={nnz}"
        )
    if case["coalesced"]:
        # A coalesced tensor has no duplicates and holds exactly nnz entries,
        # so it must be built from distinct coordinates.
        if case["layout"] != "coo" or case["coordinates"] != "distinct":
            raise ValueError(
                "coalesced=True needs a COO case with distinct coordinates"
            )
    if case["layout"] != "coo" and case["coalesced"] is not None:
        # The compressed constructor has no such flag; accepting one would
        # silently ignore requested metadata.
        raise ValueError(
            f"a {case['layout']} to_dense case carries no coalesced metadata"
        )
    if case["layout"] == "csr":
        if len(case["shape"]) < 2 or sparse_dim != 2:
            raise ValueError("a csr to_dense case needs rank >= 2 with sparse_dim=2")
        if case["coordinates"] != "distinct":
            raise ValueError(
                "csr construction merges duplicates, so it needs distinct coordinates"
            )
        if nnz > case["shape"][0] * case["shape"][1]:
            raise ValueError(
                f"shape {case['shape']} has fewer (row, column) pairs than nnz={nnz}"
            )
        # The pointers are materialized at the supported structural width, so a
        # caller-supplied descriptor whose row length or stored-entry count does
        # not fit that width is rejected here rather than overflowing the int32
        # pointer tensor at build time.
        if case["shape"][1] > _pointer_limit() or nnz > _pointer_limit():
            raise ValueError(
                f"shape {case['shape']} with nnz={nnz} does not fit the supported "
                "structural pointer width on this backend"
            )
    return case


def _unsupported_reason(case):
    """Why this backend cannot run ``case``, or ``None`` when it can.

    Applied to the default plan and to every caller-supplied descriptor, so a
    shape-file override cannot smuggle in a case the backend cannot build.
    """
    if case["layout"] == "coo" and not flag_gems.runtime.device.support_int64:
        # Stored COO coordinates are int64 on this backend (the constructor
        # normalizes the coordinates it is given), so a backend without int64
        # support cannot materialize them.
        return "COO indices need int64 support on this backend"
    if case["out_dtype"] is not None and not _out_dtype_supported(case["out_dtype"]):
        return f"the backend does not support the {case['out_dtype']} output dtype"
    return None


# Built after the validators above, because each descriptor is validated here.
# Drop descriptors this backend cannot build.  The same list feeds listing and
# execution, so a listed workload always runs.
_TO_DENSE_CASES = [
    case
    for case in _planned_cases()
    if _unsupported_reason(_validated_case(case)) is None
]
_BENCH_DTYPES = [
    dtype
    for dtype in consts.FLOAT_DTYPES
    if dtype != torch.bfloat16 or flag_gems.runtime.device.support_bf16
]


def _case_fn(shape, dtype):
    # Case metadata is built here and nowhere else, so listing and execution
    # always describe the same workloads.
    del dtype
    case = _validated_case(shape)
    reason = _unsupported_reason(case)
    if reason is not None:
        raise ValueError(
            f"to_dense case {tuple(case['shape'])} cannot run here: {reason}"
        )
    params = {
        name: case[name]
        for name in _CASE_FIELDS
        if name != "shape" and case[name] is not None
    }
    if case["layout"] != "strided":
        params["dense_tail"] = list(case["shape"][case["sparse_dim"] :])
        params["stored_nnz"] = case["nnz"]
    yield base.BenchmarkCasePlan(
        shape={"input": list(case["shape"])},
        params=params,
        builder_args=(case,),
    )


def _distinct_indices(shape, sparse_dim, nnz, device):
    """Deterministic distinct coordinates, one per stored entry."""
    dims = _sparse_dims(shape, sparse_dim)
    linear = torch.arange(nnz, dtype=torch.int64, device=device)
    rows = []
    stride = 1
    for dim in reversed(dims):
        rows.append((linear // stride) % dim)
        stride *= dim
    return torch.stack(list(reversed(rows)))


def _sparse_indices(case, device):
    """Index tensor for the stored COO entries.

    A scalar descriptor (sparse_dim 0) and an empty coordinate domain have no
    coordinates to draw, so the tensor is created empty with the (sparse_dim,
    nnz) shape the constructor expects instead of stacking or drawing from
    zero extents.
    """
    sparse_dim = case["sparse_dim"]
    nnz = case["nnz"]
    if nnz == 0 or sparse_dim == 0:
        return torch.empty((sparse_dim, nnz), dtype=torch.int64, device=device)
    if case["coordinates"] == "random":
        return torch.stack(
            [
                torch.randint(0, dim, (nnz,), dtype=torch.int64, device=device)
                for dim in _sparse_dims(case["shape"], sparse_dim)
            ]
        )
    if case["coordinates"] == "duplicates":
        return torch.zeros((sparse_dim, nnz), dtype=torch.int64, device=device)
    return _distinct_indices(case["shape"], sparse_dim, nnz, device)


def _csr_input(case, dtype, device):
    """CSR input built from explicit compressed pointers.

    ``crow``/``col`` are assembled from python ints at the supported structural
    width, so no int64 intermediate, sort or cumsum is involved and the row
    pointers are exact by construction while every stored column is distinct.
    """
    rows, cols = case["shape"][0], case["shape"][1]
    nnz = case["nnz"]
    column = []
    if rows and nnz:
        counts = [nnz // rows + (1 if row < nnz % rows else 0) for row in range(rows)]
        crow = [0]
        for row, count in enumerate(counts):
            start = (row * 5) % (cols - count + 1)
            column.extend(range(start, start + count))
            crow.append(crow[-1] + count)
    else:
        crow = [0] * (rows + 1)
    index_dtype = _index_dtype()
    values = utils.generate_tensor_input(
        (nnz,) + tuple(case["shape"][2:]), dtype, device
    )
    return torch.sparse_csr_tensor(
        torch.tensor(crow, dtype=index_dtype, device=device),
        torch.tensor(column, dtype=index_dtype, device=device),
        values,
        tuple(case["shape"]),
        device=device,
    )


def _sparse_input(case, dtype, device):
    if case["layout"] == "csr":
        return _csr_input(case, dtype, device)
    indices = _sparse_indices(case, device)
    values = utils.generate_tensor_input(
        (case["nnz"],) + tuple(case["shape"][case["sparse_dim"] :]), dtype, device
    )
    inp = torch.sparse_coo_tensor(indices, values, tuple(case["shape"]), device=device)
    return inp.coalesce() if case["coalesced"] else inp


def _build_inputs_fn(plan, dtype, device):
    case = plan.builder_args[0]
    kwargs = {}
    if case["masked_grad"] is not None:
        kwargs["masked_grad"] = case["masked_grad"]
    if case["out_dtype"] is not None:
        kwargs["dtype"] = _DTYPE_NAMES[case["out_dtype"]]
    shape = case["shape"]
    if case["layout"] == "strided":
        if case["transposed"]:
            # A parent of the reversed extents transposed is exactly the listed
            # logical shape: torch.empty((cols, rows)).t().shape == (rows, cols)
            # with stride (1, rows) (measured for square and rectangular
            # extents alike), so no other layout is substituted.
            rows, cols = shape
            return utils.generate_tensor_input((cols, rows), dtype, device).t(), kwargs
        return utils.generate_tensor_input(shape, dtype, device), kwargs
    return _sparse_input(case, dtype, device), kwargs


class ToDenseBenchmark(OperatorBenchmark):
    # to_dense's sparse path has no entries in core_shapes.yaml, so the plan
    # above (logical shape + sparse metadata) is this operator's default; a
    # --shape-file entry for "to_dense" still overrides it.
    def set_shapes(self, shape_file_path=None, *, default_shapes=_TO_DENSE_CASES):
        # The framework hands us its own relative default even for --list-cases,
        # where no dense shape file is loaded: that name is not a user request,
        # so a missing copy falls back to the plan.  Every other path --
        # including a missing --shape_file -- is delegated to the shared
        # resolver, which raises and honours per-operator/per-class entries.
        if shape_file_path is None or (
            shape_file_path == self.DEFAULT_SHAPE_FILES
            and not os.path.isfile(shape_file_path)
        ):
            self.shapes = [tuple(shape) for shape in default_shapes]
            return
        super().set_shapes(shape_file_path, default_shapes=default_shapes)


@pytest.mark.to_dense
def test_to_dense():
    bench = ToDenseBenchmark(
        op_name="to_dense",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.to_dense,
        gems_op=getattr(flag_gems, "to_dense", None),
        dtypes=_BENCH_DTYPES,
    )
    bench.run()
