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

import itertools

import pytest
import torch

import flag_gems

from .generated_operator_utils import OperatorBenchmark
from . import base, consts

# aten::ccol_indices(Tensor(a) self) -> Tensor(a) returns the batch_dims +
# (ncols + 1,) int64 compressed column index tensor of a sparse CSC tensor — a
# metadata accessor whose result is an alias of the input's internal ccol
# storage. Its cost is proportional to ncols (the size of the returned tensor)
# and independent of the stored values, so benchmark a spread of logical shapes
# (nrows, ncols) / (batch, nrows, ncols) / (batch_1, batch_2, nrows, ncols) and
# nnz values. The device-side allocation stays small relative to the logical
# size because only nnz entries are stored.
_CCOL_SHAPES = [
    ((1024, 1024), 65536),
    ((1024, 1024), 1048576),
    ((4096, 4096), 1048576),
    ((16, 1024, 1024), 262144),
    ((8, 256, 256), 16384),
    ((64, 1024, 1024), 131072),
    ((4, 8, 256, 256), 16384),
]


def _make_structure(logical_shape, nnz, device):
    # Unique coordinates in compressed order.
    nrows, ncols = logical_shape[-2], logical_shape[-1]
    batch = logical_shape[:-2]
    entries_shape = batch + (nnz,)
    assert 0 <= nnz <= nrows * ncols
    positions = (
        torch.arange(nnz, dtype=torch.long, device=device)
        * (nrows * ncols)
        // max(nnz, 1)
    )
    rows = (positions % nrows).expand(entries_shape).contiguous()
    cols = (positions // nrows).expand(entries_shape).contiguous()
    counts = torch.stack(
        [
            torch.bincount(cols[idx], minlength=ncols)
            for idx in itertools.product(*(range(d) for d in batch))
        ]
    ).view(batch + (ncols,))
    ccol = torch.zeros(batch + (ncols + 1,), dtype=torch.long, device=device)
    ccol[..., 1:] = torch.cumsum(counts, -1)
    return ccol, rows


def _case_fn(shape, dtype):
    del dtype
    logical_shape, nnz = shape
    yield base.BenchmarkCasePlan(
        shape={"input": logical_shape},
        params={"nnz": nnz},
        builder_args=(logical_shape, nnz),
    )


def _build_inputs_fn(plan, dtype, device):
    logical_shape, nnz = plan.builder_args
    ccol, rows = _make_structure(logical_shape, nnz, device)
    values_shape = logical_shape[:-2] + (nnz,)
    values = torch.randn(values_shape, dtype=dtype, device=device)
    inp = torch.sparse_csc_tensor(ccol, rows, values, logical_shape)
    return inp, {}


class CcolIndicesBenchmark(OperatorBenchmark):
    # ccol_indices is a sparse metadata accessor; there are no meaningful dense
    # shapes in core_shapes.yaml, so benchmark dedicated (logical_shape, nnz)
    # pairs instead.
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_CCOL_SHAPES)


@pytest.mark.ccol_indices
def test_ccol_indices():
    bench = CcolIndicesBenchmark(
        op_name="ccol_indices",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.ccol_indices,
        gems_op=getattr(flag_gems, "ccol_indices", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
