# Copyright 2025, The FlagGems Authors.
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

"""Benchmark for ``_convert_indices_from_csr_to_coo``.

Two-phase benchmark: ``case_fn`` alone produces the case list, so listing works
without a candidate and without allocating inputs, and ``build_inputs_fn``
builds the csr operands for the very same plans at execution time, including
the output flag described by the plan metadata.
"""

import pytest
import torch

import flag_gems

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

_OP_NAME = "_convert_indices_from_csr_to_coo"

# Original scales: the 1-D scales stress row-pointer traversal, the higher-rank
# scales stress batched blocks with independent offsets.
_BENCH_SHAPES = [
    (2**20,),
    (2**22,),
    (4096, 256),
    (8192, 512),
    (64, 128, 128),
    (16, 256, 64, 4),
]

_INT64_SUPPORTED = bool(getattr(flag_gems.runtime.device, "support_int64", False))
_INDEX_INTERMEDIATE = torch.int64 if _INT64_SUPPORTED else torch.int32

_BENCH_DTYPES = [torch.int8, torch.uint8] + list(consts.INT_DTYPES)
if _INT64_SUPPORTED:
    _BENCH_DTYPES.append(torch.int64)

# Valid csr layouts that move the written entries differently: one entry per row
# versus every entry in the first row of a block.
_CROW_PATTERNS = ("identity_rows", "first_row_holds_all")

# The schema default output is int64.  A device without int64 output support
# cannot run that default, so both the plan metadata and the builder kwargs
# describe the explicitly supported int32 output instead of leaving the case
# list claiming a default call the configuration cannot execute.
_OUT_KWARGS = {} if _INT64_SUPPORTED else {"out_int32": True}


def _numel(shape):
    total = 1
    for extent in shape:
        total *= extent
    return total


def _crow_shape(col_shape):
    return tuple(col_shape[:-1]) + (col_shape[-1] + 1,)


def _validated_shape(shape, dtype):
    """Return the descriptor shape as a tuple, or ``None`` when unsupported.

    Scalar descriptors, bool/fractional extents and negative extents are caller
    mistakes and raise before any allocation; zero extents stay valid.  A
    block's terminal offset equals its stored extent (``shape[-1]``; the whole
    extent for a 1-D input), so a small index dtype cannot address the larger
    scales: those (scale, dtype) pairs return ``None`` rather than being built
    from wrapped offsets.  No size cap is applied beyond that.
    """
    if not isinstance(shape, (tuple, list)) or len(shape) == 0:
        raise ValueError(
            f"shape descriptor must be a non-empty sequence, got {shape!r}"
        )
    for extent in shape:
        if isinstance(extent, bool) or not isinstance(extent, int):
            raise TypeError(f"shape extents must be integers, got {shape!r}")
        if extent < 0:
            raise ValueError(f"shape extents must be non-negative, got {shape!r}")
    shape = tuple(shape)
    if shape[-1] > int(torch.iinfo(dtype).max):
        return None
    return shape


def _crow_offsets(col_shape, pattern, dtype, device):
    nnz = col_shape[-1]
    rows = nnz + 1
    if pattern == "identity_rows":
        offsets = torch.arange(rows, dtype=_INDEX_INTERMEDIATE, device=device)
    else:
        offsets = torch.full((rows,), nnz, dtype=_INDEX_INTERMEDIATE, device=device)
        offsets[0] = 0
    offsets = offsets.repeat(_numel(col_shape[:-1]))
    return offsets.reshape(_crow_shape(col_shape)).to(dtype)


def _case_fn(shape, dtype):
    col_shape = _validated_shape(shape, dtype)
    if col_shape is None:
        return
    for pattern in _CROW_PATTERNS:
        yield base.BenchmarkCasePlan(
            shape={"col": list(col_shape), "crow": list(_crow_shape(col_shape))},
            params={
                "crow_pattern": pattern,
                "out_int32": _OUT_KWARGS.get("out_int32", False),
            },
            builder_args=(col_shape, pattern),
        )


def _build_inputs_fn(plan, dtype, device):
    col_shape, pattern = plan.builder_args
    crow = _crow_offsets(col_shape, pattern, dtype, device)
    col = torch.zeros(col_shape, dtype=dtype, device=device)
    # The harness takes the builder's return as the flat argument sequence
    # followed by the keyword mapping, so both csr operands are separate
    # positional arguments and the output flag matches the plan metadata.
    return crow, col, dict(_OUT_KWARGS)


class ConvertIndicesFromCsrToCooBenchmark(OperatorBenchmark):
    """Resolves the operator's configured shapes through the shared resolver."""

    def set_shapes(self, shape_file_path=None):
        # The shared resolver owns the op-name/class-name lookup order and uses
        # the built-in scales only for entries a caller's file does not define,
        # so a missing file raises instead of being silently replaced.
        super().set_shapes(
            base.Config.shape_file if shape_file_path is None else shape_file_path,
            default_shapes=_BENCH_SHAPES,
        )


@pytest.mark.convert_indices_from_csr_to_coo
def test__convert_indices_from_csr_to_coo():
    bench = ConvertIndicesFromCsrToCooBenchmark(
        op_name=_OP_NAME,
        torch_op=torch.ops.aten._convert_indices_from_csr_to_coo,
        gems_op=getattr(flag_gems, _OP_NAME, None),
        dtypes=_BENCH_DTYPES,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
    )
    bench.run()
