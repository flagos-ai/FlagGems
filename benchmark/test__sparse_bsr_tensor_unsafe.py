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

# aten::_sparse_bsr_tensor_unsafe(Tensor crow_indices, Tensor col_indices,
#     Tensor values, int[] size, *, ScalarType? dtype=None, Layout? layout=None,
#     Device? device=None, bool? pin_memory=None) -> Tensor
#
# The measured work is the layout construction from the raw components, so every
# case feeds the three component tensors directly and the reference and the
# candidate receive the identical call.  Broadcast and backward do not apply:
# this is a constructor with no operand arithmetic and no autograd formula.
#
# A case is a (tensor_shape, block) descriptor: the trailing two dims of
# tensor_shape are the sparse extent, tiled by ``block`` (always 2-D), and any
# leading dims are batch dims.  The block grid stores a bounded number of blocks
# per row block so the values allocation follows nnz instead of the logical
# extent.
_BENCH_SHAPES = [
    ((512, 512), (16, 16)),
    ((1024, 1024), (32, 32)),
    ((2048, 2048), (64, 64)),
    ((4096, 4096), (128, 128)),
    ((64, 512, 512), (32, 32)),
    ((16, 1024, 1024), (64, 64)),
    # Empty/zero boundaries: a zero-length sparse extent and a zero column
    # count must build without dividing by the block size.
    ((0, 0), (1, 1)),
    ((8, 0), (2, 2)),
    ((0, 8), (2, 2)),
]

# Stored blocks per row-block (the original nnz policy for these workloads);
# bounded by the smallest col-block count above, so every col index is in range.
_BLOCKS_PER_ROW = 4


def _as_bsr_descriptor(entry):
    """Canonical (tensor_shape, block) for a shared or operator-owned case.

    An entry that already is a (tensor_shape, block) pair keeps its batch rank
    and block; the block may be 2-D only, while tensor_shape may have any rank
    the operator accepts.  Anything else is a bare shared shape: the logical
    size is kept unchanged and tiled with the always-legal 1x1 block.  Nothing
    is dropped or shrunk.
    """
    if (
        isinstance(entry, (tuple, list))
        and len(entry) == 2
        and all(isinstance(part, (tuple, list)) for part in entry)
        and all(isinstance(dim, int) for dim in entry[0])
        and len(entry[1]) == 2
        and all(isinstance(dim, int) for dim in entry[1])
    ):
        return (tuple(entry[0]), tuple(entry[1]))
    return (tuple(entry), (1, 1))


def _make_components(tensor_shape, block, dtype, device):
    """Build the raw crow/col/values triple for one descriptor."""
    block_rows, block_cols = block
    if len(tensor_shape) >= 2:
        extent_rows, extent_cols = tensor_shape[-2], tensor_shape[-1]
    else:
        # A rank<2 logical size has no sparse extent: it is stored with no
        # blocks at all, which this unchecked constructor accepts verbatim.
        extent_rows, extent_cols = 0, 0
    n_row_blocks = extent_rows // block_rows
    n_col_blocks = extent_cols // block_cols
    per_row = min(_BLOCKS_PER_ROW, n_col_blocks)
    nnz = n_row_blocks * per_row

    # Distinct, in-range, sorted col indices per row block.  Deriving them from
    # modular arithmetic keeps the peak allocation proportional to nnz even for
    # the wide core shapes, and makes each case reproducible without a seed.
    if n_row_blocks and per_row:
        starts = (torch.arange(n_row_blocks, device=device) * 7 + 3) % n_col_blocks
        spans = torch.arange(per_row, device=device)
        selected = (starts.unsqueeze(1) + spans.unsqueeze(0)) % n_col_blocks
        selected = selected.sort(dim=1).values
        crow = torch.arange(0, nnz + 1, per_row, dtype=torch.long, device=device)
    else:
        selected = torch.empty((n_row_blocks, per_row), dtype=torch.long, device=device)
        crow = torch.zeros(n_row_blocks + 1, dtype=torch.long, device=device)

    # Any leading dims are batch dims: crow/col repeat one block grid per batch.
    batch = tensor_shape[:-2]
    crow_t = crow.repeat(*batch, 1).contiguous()
    col_t = selected.reshape(-1).repeat(*batch, 1).contiguous()
    values_t = utils.generate_tensor_input(
        batch + (nnz, block_rows, block_cols), dtype, device
    )
    return crow_t, col_t, values_t


def _case_fn(shape, dtype):
    # Metadata only: listing allocates no tensors and runs no operator.
    del dtype
    tensor_shape, block = shape
    yield base.BenchmarkCasePlan(
        shape={"input": tensor_shape},
        params={"block": block},
        builder_args=(tensor_shape, block),
    )


def _build_inputs_fn(plan, dtype, device):
    tensor_shape, block = plan.builder_args
    crow, col, values = _make_components(tensor_shape, block, dtype, device)
    # The trailing dict is unpacked into call kwargs, so torch_op and gems_op
    # both receive (crow, col, values, size, dtype=..., device=...): the native
    # call defaults to CPU/float32 and requires the requested device to match the
    # components.
    return crow, col, values, list(tensor_shape), {"dtype": dtype, "device": device}


class SparseBsrTensorUnsafeBenchmark(OperatorBenchmark):
    """Two-phase benchmark feeding the raw BSR components to this operator."""

    def set_shapes(self, shape_file_path=None):
        # The shared loader runs first, so a shape-file entry for this operator
        # (or for the class hierarchy) and the shared core/comprehensive shape
        # sets are all honoured; this operator's own descriptors are then
        # UNIONed on top.  Every entry is normalised to a full descriptor and
        # none is filtered out.
        super().set_shapes(shape_file_path)
        entries = [_as_bsr_descriptor(shape) for shape in self.shapes]
        self.shapes = list(dict.fromkeys(entries + list(_BENCH_SHAPES)))


@pytest.mark.sparse_bsr_tensor_unsafe
def test_sparse_bsr_tensor_unsafe():
    bench = SparseBsrTensorUnsafeBenchmark(
        op_name="_sparse_bsr_tensor_unsafe",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._sparse_bsr_tensor_unsafe,
        gems_op=getattr(flag_gems, "_sparse_bsr_tensor_unsafe", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
