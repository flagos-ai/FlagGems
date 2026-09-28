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

"""Benchmark for aten::to_dense_backward.

Two-phase GenericBenchmark: _case_fn turns one shape or workload entry into a
metadata-only plan (the same plans drive --list-cases and --case-id replay) and
_build_inputs_fn materializes the operands from builder_args.  The dense path is
measured at the four scales below and the sparse path at its own descriptors; a
shape supplied by a custom file is honoured as a dense workload instead of
being dropped.  A sparse descriptor additionally needs the device's INT64
support for its coordinates; without it the sparse plans are not produced and
the dense scales still run.

A configured list may hold plain shapes or the descriptors described in _plan.
Everything read back from such a file is decoded YAML/JSON, so the optional
argument is matched by value: the sentinel is a plain string, never an object
identity, and no entry is silently coerced into a different workload.
"""

import pytest
import torch

import flag_gems

from . import base, consts, utils
from .generated_operator_utils import OperatorBenchmark

# Sentinel meaning: call the operator without the optional argument.
OMITTED = "omitted"

# The four dense scales first, then the sparse descriptors
# (shape, kind, nnz, masked_grad).  A plain shape entry is accepted as well, so
# a shape file that configures this operator or this class keeps working
# unchanged.
DEFAULT_WORKLOADS = [
    (1024, 1024),
    (4096, 4096),
    (20, 320, 15),
    (64, 512, 512),
    ((1024, 1024), "sparse", 4096, OMITTED),
    ((1024, 1024), "sparse", 4096, True),
    ((1024, 1024), "sparse", 4096, False),
    ((20, 320, 15), "sparse", 2048, True),
]

# The input builder supports the float families only, and bfloat16 is gated on
# the static device capability so the comparison reference stays available.
BENCH_DTYPES = [
    dtype
    for dtype in consts.FLOAT_DTYPES
    if dtype != torch.bfloat16 or flag_gems.runtime.device.support_bf16
]

# Structural prerequisite of the sparse plans: the coordinate list the sparse
# input builder allocates is INT64 (torch requires INT64 COO indices, and the
# flat positions reach the element count of the domain, which a legal shape can
# push past the INT32 range), and there is no INT32 construction to fall back
# on.  The prerequisite is applied where the plan is produced, so the cases
# listed by --list-cases, replayed by --case-id and supplied by a shape file are
# decided by the same rule; a sparse descriptor has no valid operand without it,
# while every dense plan is kept.
STRUCTURAL_INT64 = flag_gems.runtime.device.support_int64


def _count(shape):
    total = 1
    for dim in shape:
        total *= dim
    return total


def _shape_extents(shape):
    """Plain non-negative integer extents.

    A float, bool or string extent is rejected instead of being silently
    truncated: a caller that asks for [2.7, 3] or [True, 3] has made a mistake,
    and turning that into (2, 3) or (1, 3) would measure a different workload.
    Zero and negative-free extents of any rank, including the 0-dim shape, stay
    valid.
    """
    if not isinstance(shape, (tuple, list)):
        raise ValueError(
            "a workload shape must be a sequence of extents: %r" % (shape,)
        )
    extents = []
    for dim in shape:
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise ValueError("shape extents must be integers: %r" % (shape,))
        if dim < 0:
            raise ValueError("negative extent in %r" % (shape,))
        extents.append(int(dim))
    return tuple(extents)


def _normalize_masked_grad(value):
    """Return the canonical optional argument to call the operator with.

    The sentinel is matched by value: a descriptor decoded from YAML or JSON
    carries a string equal to OMITTED but not the same object, so an identity
    test would forward the string itself to the native operator.  None and bool
    pass through, and an integer passes through as well because the schema is
    Optional[bool] and the native operator accepts integers (0 is the only false
    value), so a descriptor may request that form directly.  Anything else is a
    mistake in the request and is reported instead of being guessed at.
    """
    if isinstance(value, str):
        if value != OMITTED:
            raise ValueError("unsupported masked_grad: %r" % (value,))
        return OMITTED
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    raise ValueError(
        "masked_grad must be %r, None, a bool or an int: %r" % (OMITTED, value)
    )


def _coo_indices(shape, nnz, device):
    """Deterministic sorted coordinates hitting nnz exactly.

    The mixed-radix decomposition of an increasing flat index list is
    lexicographically sorted and the count is exact, so no overdraw or
    deduplication is needed and an empty domain stays valid.  The structural
    arithmetic runs in INT64 because the flat position reaches the total element
    count of the domain, which a legal shape can push past the INT32 range, and
    because torch requires INT64 COO indices in any case.
    """
    total = _count(shape)
    if nnz == 0:
        return torch.empty((len(shape), 0), dtype=torch.int64, device=device)
    step = max(1, total // nnz)
    flat = torch.arange(nnz, dtype=torch.int64, device=device) * step
    coordinates = []
    remainder = flat
    for dim in reversed(shape):
        coordinates.append(remainder % dim)
        remainder = remainder // dim
    return torch.stack(list(reversed(coordinates))).to(torch.int64)


def _plan(entry):
    """Validate one workload entry and normalize it into a case descriptor.

    Everything is checked before any allocation, so a malformed entry fails
    loudly instead of silently producing a different shape or a denser
    workload.  An entry is either a plain shape (a dense workload) or the
    four-item descriptor (shape, kind, nnz, masked_grad).  set_shapes() looks
    the list up under this operator's name and under the benchmark class name
    alike.
    """
    nested = (
        isinstance(entry, (tuple, list))
        and entry
        and isinstance(entry[0], (tuple, list))
    )
    if nested:
        if len(entry) != 4:
            raise ValueError("a workload descriptor needs 4 items: %r" % (entry,))
        shape, kind, nnz, masked_grad = entry
    else:
        shape, kind, nnz, masked_grad = entry, "dense", None, OMITTED
    shape = _shape_extents(shape)
    if not isinstance(kind, str) or kind not in ("dense", "sparse"):
        raise ValueError("unsupported workload kind: %r" % (kind,))
    masked_grad = _normalize_masked_grad(masked_grad)
    if kind == "sparse":
        if not shape:
            raise ValueError("a sparse workload needs at least one dimension")
        if isinstance(nnz, bool) or not isinstance(nnz, int):
            raise ValueError("nnz must be an integer: %r" % (nnz,))
        if nnz < 0 or nnz > _count(shape):
            raise ValueError("nnz out of range for %r" % (shape,))
    else:
        if nnz is not None:
            raise ValueError("a dense workload takes no nnz: %r" % (entry,))
        nnz = None
    return kind, shape, nnz, masked_grad


def _case_fn(shape, dtype):
    del dtype
    kind, case_shape, nnz, masked_grad = _plan(shape)
    if kind == "sparse" and not STRUCTURAL_INT64:
        # Without INT64 coordinates the sparse descriptor has no valid operand,
        # so no unsupported tensor is built: the plan is simply not produced and
        # a --case-id request for it finds no case, as for any other input that
        # was not planned.
        return
    yield base.BenchmarkCasePlan(
        shape={"grad": list(case_shape), "input": list(case_shape)},
        params={
            "kind": kind,
            "nnz": -1 if nnz is None else nnz,
            "masked_grad": str(masked_grad),
        },
        builder_args=(kind, case_shape, nnz, masked_grad),
    )


def _build_inputs_fn(plan, dtype, device):
    kind, shape, nnz, masked_grad = plan.builder_args
    grad = utils.generate_tensor_input(shape, dtype, device)
    if kind == "dense":
        inp = utils.generate_tensor_input(shape, dtype, device)
    else:
        inp = torch.sparse_coo_tensor(
            _coo_indices(shape, nnz, device),
            torch.ones(nnz, dtype=dtype, device=device),
            shape,
            device=device,
            is_coalesced=True,
        )
    if masked_grad == OMITTED:
        return grad, inp
    return grad, inp, {"masked_grad": masked_grad}


class ToDenseBackwardBenchmark(OperatorBenchmark):
    """Uses this operator's own workloads when the shape file has no entry.

    A configured list may hold plain shapes or the descriptors described in
    _plan; both go through the same validation, so listing and execution always
    agree.
    """

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=DEFAULT_WORKLOADS)


@pytest.mark.to_dense_backward
def test_to_dense_backward():
    bench = ToDenseBackwardBenchmark(
        op_name="to_dense_backward",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.to_dense_backward,
        gems_op=getattr(flag_gems, "to_dense_backward", None),
        dtypes=BENCH_DTYPES,
    )
    bench.run()
