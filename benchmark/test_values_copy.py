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

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
_LAYOUTS = {
    "csr": torch.sparse_csr_tensor,
    "csc": torch.sparse_csc_tensor,
    "bsr": torch.sparse_bsr_tensor,
    "bsc": torch.sparse_bsc_tensor,
}
_BLOCK_LAYOUTS = ("bsr", "bsc")
# Sparse index metadata is built in the widest index dtype this backend declares
# support for, so no implicit int64 tensor is created on int32-only devices.
_INDEX_DTYPE = torch.int64 if flag_gems.runtime.device.support_int64 else torch.int32

# (layout, size, nnz, blocks, nbatch). values_copy materializes the stored values
# of a compressed sparse tensor, so its cost follows the stored values - nnz times
# the BSR/BSC block or the dense trailing dims - and not the logical matrix size.
# The cases sweep nnz magnitudes, dense trailing dims, block layouts, one and two
# batch axes, and a zero-extent storage; only nnz values per batch are allocated.
_CASES = [
    ("csr", (1024, 1024), 65536, None, 0),
    ("csr", (4096, 4096), 1048576, None, 0),
    ("csr", (1024, 1024, 64), 65536, None, 0),
    ("csr", (8, 4096, 4096), 131072, None, 1),
    ("csc", (4096, 4096), 1048576, None, 0),
    ("bsr", (4096, 4096), 262144, (8, 8), 0),
    ("bsr", (4, 4096, 4096), 65536, (8, 8), 1),
    ("bsc", (4096, 4096), 262144, (8, 8), 0),
    ("csr", (4, 2, 4096, 4096), 131072, None, 2),
    ("csr", (0, 1024), 0, None, 0),
]


def _geometry(layout, size, nnz, blocks, nbatch):
    """Validate one case row and return its index/values metadata.

    ``groups`` counts the compressed groups (rows for CSR/BSR, columns for
    CSC/BSC) and ``bound`` the exclusive plain-index bound inside a group, so
    ``groups * bound`` is the storage capacity of the layout. Everything is
    checked before any device allocation, so an invalid shape file entry cannot
    reach the device as a malformed sparse tensor.
    """
    if layout not in _LAYOUTS:
        raise ValueError(f"unknown sparse layout: {layout!r}")
    if not isinstance(nbatch, int) or isinstance(nbatch, bool) or nbatch < 0:
        raise ValueError(f"nbatch must be a non-negative int: {nbatch!r}")
    size = tuple(size)
    if len(size) < nbatch + 2:
        raise ValueError(
            f"size {size!r} cannot carry {nbatch} batch dims plus a matrix"
        )
    for extent in size:
        if not isinstance(extent, int) or isinstance(extent, bool) or extent < 0:
            raise ValueError(f"extents must be non-negative ints: {size!r}")
    if not isinstance(nnz, int) or isinstance(nnz, bool) or nnz < 0:
        raise ValueError(f"nnz must be a non-negative int: {nnz!r}")
    batch = size[:nbatch]
    m, n = size[nbatch], size[nbatch + 1]
    dense = size[nbatch + 2 :]
    if layout in _BLOCK_LAYOUTS:
        if not isinstance(blocks, (tuple, list)) or len(blocks) != 2:
            raise ValueError(
                f"{layout} needs a (block_rows, block_cols) pair: {blocks!r}"
            )
        blocks = tuple(blocks)
        block_rows, block_cols = blocks
        for block in blocks:
            if not isinstance(block, int) or isinstance(block, bool) or block <= 0:
                raise ValueError(f"block dims must be positive ints: {blocks!r}")
        if m % block_rows or n % block_cols:
            raise ValueError(f"{layout} blocks {blocks!r} do not tile {m}x{n}")
        groups = m // block_rows if layout == "bsr" else n // block_cols
        bound = n // block_cols if layout == "bsr" else m // block_rows
    elif blocks is not None:
        raise ValueError(
            f"{layout} is an element layout and takes no blocks: {blocks!r}"
        )
    elif layout == "csc":
        groups, bound = n, m
    else:
        groups, bound = m, n
    if groups == 0 or bound == 0:
        if nnz:
            raise ValueError(f"{layout} {size} has no room for {nnz} stored values")
    elif nnz > groups * bound:
        raise ValueError(f"{layout} {size} cannot store {nnz} values")
    values_shape = batch + (nnz,) + tuple(blocks or ()) + dense
    return {
        "batch": list(batch),
        "dense": list(dense),
        "groups": groups,
        "bound": bound,
        "blocks": list(blocks) if blocks else None,
        "compressed_shape": list(batch + (groups + 1,)),
        "index_shape": list(batch + (nnz,)),
        "values_shape": list(values_shape),
    }


def _stored_values(shape, dtype, device):
    """Generate stored values on the target device, with a device-side fp8 path."""
    if dtype in _FP8_DTYPES:
        # randn has no fp8 CUDA kernel, so generate in float32 on the device and
        # cast there instead of building the tensor on the host.
        return torch.randn(shape, dtype=torch.float32, device=device).to(dtype)
    if dtype.is_floating_point or dtype.is_complex:
        return torch.randn(shape, dtype=dtype, device=device)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, device=device, dtype=torch.bool)
    if dtype == torch.uint8:
        return torch.randint(0, 6, shape, device=device, dtype=torch.uint8)
    # randint accepts every remaining supported integer dtype directly, so no
    # int64 intermediate is allocated before the cast.
    return torch.randint(-5, 6, shape, device=device, dtype=dtype)


def _make_input(layout, size, nnz, blocks, nbatch, dtype, device):
    meta = _geometry(layout, size, nnz, blocks, nbatch)
    groups = meta["groups"]
    if groups == 0:
        # Zero-extent storage: one empty compressed-offset vector, no plain index.
        compressed = torch.zeros(1, dtype=_INDEX_DTYPE, device=device)
        plain = torch.empty(0, dtype=_INDEX_DTYPE, device=device)
    else:
        counts = torch.full((groups,), nnz // groups, dtype=_INDEX_DTYPE, device=device)
        counts[: nnz % groups] += 1
        # cumsum promotes int32 to int64 by default, which would allocate wider
        # index tensors than the declared index dtype; pin it.
        compressed = torch.cat(
            [
                torch.zeros(1, dtype=_INDEX_DTYPE, device=device),
                counts.cumsum(0, dtype=_INDEX_DTYPE),
            ]
        )
        plain = torch.arange(nnz, device=device, dtype=_INDEX_DTYPE) - (
            torch.repeat_interleave(compressed[:-1], counts)
        )
    if meta["batch"]:
        compressed = compressed.expand(*meta["batch"], -1).contiguous()
        plain = plain.expand(*meta["batch"], -1).contiguous()
    values = _stored_values(tuple(meta["values_shape"]), dtype, device)
    return _LAYOUTS[layout](compressed, plain, values, size=tuple(size), device=device)


def _case_fn(shape, dtype):
    del dtype
    layout, size, nnz, blocks, nbatch = shape
    meta = _geometry(layout, size, nnz, blocks, nbatch)
    yield base.BenchmarkCasePlan(
        shape={"input": list(size)},
        params={
            "layout": layout,
            "size": list(size),
            "nnz": nnz,
            "nbatch": nbatch,
            **meta,
        },
        builder_args=(layout, tuple(size), nnz, blocks, nbatch),
    )


def _build_inputs_fn(plan, dtype, device):
    layout, size, nnz, blocks, nbatch = plan.builder_args
    return _make_input(layout, size, nnz, blocks, nbatch, dtype, device), {}


def _gated_dtypes(dtypes):
    """Keep the dtypes this backend declares it supports (static device flags)."""
    flags = flag_gems.runtime.device
    gated = []
    for dtype in dtypes:
        if dtype in (torch.float64, torch.complex128) and not flags.support_fp64:
            continue
        if dtype == torch.bfloat16 and not flags.support_bf16:
            continue
        if dtype == torch.int64 and not flags.support_int64:
            continue
        if dtype in _FP8_DTYPES and not flags.support_fp8:
            continue
        gated.append(dtype)
    return gated


_BENCH_DTYPES = _gated_dtypes(
    consts.FLOAT_DTYPES
    + [
        torch.float64,
        torch.int8,
        torch.int32,
        torch.int64,
        torch.complex64,
        torch.complex128,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.bool,
    ]
)


class ValuesCopyBenchmark(OperatorBenchmark):
    # values_copy is a sparse accessor: core_shapes.yaml carries dense shapes that
    # this operator rejects, so benchmark dedicated compressed-sparse cases.
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_CASES)


@pytest.mark.values_copy
def test_values_copy():
    bench = ValuesCopyBenchmark(
        op_name="values_copy",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.values_copy,
        gems_op=getattr(flag_gems, "values_copy", None),
        dtypes=_BENCH_DTYPES,
    )
    bench.run()
