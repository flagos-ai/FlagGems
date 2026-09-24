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

from .generated_operator_utils import OperatorBenchmark
from . import base, consts, utils

# aten::sparse_bsr_tensor.crow_col_value_size(Tensor crow_indices,
#     Tensor col_indices, Tensor values, int[] size, *,
#     ScalarType? dtype=None, ...) -> Tensor constructs a sparse BSR tensor from
# its raw components: the trailing (rows, cols) dims of ``size`` are tiled by
# the block shape inferred from ``values`` (nnz, br, bc), or
# (batch, nnz, br, bc) for batched tensors. The measured work is the layout
# construction from the three component tensors, so the benchmark feeds the
# components directly (not a pre-built sparse tensor) and both the reference and
# the candidate receive the exact same call.
#
# Shape levels: the workloads below mirror the performance-relevant shapes of
# ``core_shapes.yaml`` for dense pointwise/layout ops, with a block size chosen
# so the stored values dominate the construction cost.
#
# Broadcast/backward do not apply: this is a constructor with no tensor
# arithmetic between operands and no autograd formula.
#
# Each benchmark case is (tensor_shape, block). The block grid stores a fixed
# number of blocks per row-block (``_BLOCKS_PER_ROW``), so nnz -- and thus the
# values allocation -- grows with the number of row-blocks while the logical
# matrix spans the full (rows, cols) extent.
_BENCH_SHAPES = [
    ((512, 512), (16, 16)),
    ((1024, 1024), (32, 32)),
    ((2048, 2048), (64, 64)),
    ((4096, 4096), (128, 128)),
    ((64, 512, 512), (32, 32)),
    ((16, 1024, 1024), (64, 64)),
]

# Stored blocks per row-block; must stay <= the smallest col-block count of any
# case above (16), so every generated col index is in range.
_BLOCKS_PER_ROW = 4


def _make_bsr_inputs(shape, block, dtype, device, seed=0):
    # Per-row-block grids are built on CPU for determinism, then moved to the
    # benchmark device. The values allocation is proportional to nnz, which is
    # bounded by _BLOCKS_PER_ROW * n_row_blocks regardless of the logical size.
    gen = torch.Generator("cpu").manual_seed(seed)
    rows, cols = shape[-2], shape[-1]
    br, bc = block
    assert rows % br == cols % bc == 0
    n_row_blocks = rows // br
    n_col_blocks = cols // bc
    crow = [0]
    col = []
    for _ in range(n_row_blocks):
        k = min(_BLOCKS_PER_ROW, n_col_blocks)
        chosen = torch.randperm(n_col_blocks, generator=gen)[:k].sort().values
        col.extend(chosen.tolist())
        crow.append(crow[-1] + k)
    nnz = crow[-1]
    values_shape = shape[:-2] + (nnz, br, bc)
    crow_t = torch.tensor(crow, dtype=torch.long, device=device)
    col_t = torch.tensor(col, dtype=torch.long, device=device)
    crow_t = crow_t.expand(shape[:-2] + (n_row_blocks + 1,)).contiguous()
    col_t = col_t.expand(shape[:-2] + (nnz,)).contiguous()
    values_t = utils.generate_tensor_input(values_shape, dtype, device)
    return crow_t, col_t, values_t


def _case_fn(shape, dtype):
    # Two-phase GenericBenchmark: the descriptors are (tensor_shape, block)
    # pairs; one BenchmarkCasePlan per descriptor keeps the block in the case id
    # and defers all tensor construction to _build_inputs_fn.
    del dtype
    tensor_shape, block = shape
    yield base.BenchmarkCasePlan(
        shape={"input": tensor_shape},
        params={"block": block},
        builder_args=(tensor_shape, block),
    )


def _build_inputs_fn(plan, dtype, device):
    tensor_shape, block = plan.builder_args
    crow, col, values = _make_bsr_inputs(tensor_shape, block, dtype, device)
    # The trailing dict is unpacked into kwargs by the benchmark runner, so
    # torch_op and gems_op both receive (crow, col, values, size, dtype=...,
    # device=...). The device keyword is required: on CUDA this torch build
    # fails to infer the device from the input tensors otherwise.
    return crow, col, values, list(tensor_shape), {"dtype": dtype, "device": device}


class SparseBsrTensorBenchmark(OperatorBenchmark):
    """Two-phase GenericBenchmark feeding the raw BSR components to the
    ``sparse_bsr_tensor`` factory call."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_BENCH_SHAPES)


@pytest.mark.sparse_bsr_tensor
def test_sparse_bsr_tensor():
    bench = SparseBsrTensorBenchmark(
        op_name="sparse_bsr_tensor",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.sparse_bsr_tensor,
        # flag_gems has no public ``sparse_bsr_tensor`` direct callable yet; the
        # candidate is supplied by the KernelGen process-local override keyed on
        # the public operator name (resolved via
        # Benchmark._candidate_call inside the benchmark).
        gems_op=getattr(flag_gems, "sparse_bsr_tensor", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
