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
from . import base, consts

# aten::sparse_compressed_tensor.comp_plain_value_size(Tensor compressed_indices,
#     Tensor plain_indices, Tensor values, SymInt[] size, *, ScalarType? dtype=None,
#     Layout? layout=None, Device? device=None, bool? pin_memory=False) -> Tensor
# constructs a sparse compressed tensor (CSR/CSC/BSR/BSC) from raw index tensors
# and values. The generic factory needs layout= (mandatory), and on GPU both
# dtype= and device= must be passed explicitly (the aten op does not infer dtype
# from values and creates the instance on CPU when device is omitted). The
# measured work is the layout construction from the component tensors, so the
# benchmark feeds the components directly (not a pre-built sparse tensor) and
# both the reference and the candidate receive the exact same call.
#
# Each benchmark case is (layout, tensor_shape, nnz). The nnz is distributed
# over the rows/columns with a deterministic structure, so the index arrays and
# the values allocation all scale with nnz while the logical matrix spans the
# full (rows, cols) extent. Block layouts (BSR/BSC) get (nnz, block, block)
# value tensors; higher-rank shapes are batched compressed tensors. The
# batched cases keep the per-batch index arrays modest so the component
# allocation stays within device memory.
_SPARSE_COMPRESSED_SHAPES = [
    (torch.sparse_csr, (1024, 1024), 65536),
    (torch.sparse_csr, (1024, 1024), 262144),
    (torch.sparse_csc, (1024, 1024), 262144),
    (torch.sparse_csr, (4096, 4096), 1048576),
    (torch.sparse_bsr, (2048, 2048), 262144),
    (torch.sparse_bsc, (2048, 2048), 262144),
    (torch.sparse_csr, (8, 256, 256), 16384),
]

_BLOCK_LAYOUTS = (torch.sparse_bsr, torch.sparse_bsc)
_BLOCK_SIZE = 2


def _make_input(layout, shape, nnz, dtype, device):
    # Sorted unique entries in each compressed segment.
    batch = shape[:-2]
    nrows, ncols = shape[-2], shape[-1]
    if layout in _BLOCK_LAYOUTS:
        bs0 = bs1 = _BLOCK_SIZE
    else:
        bs0 = bs1 = 1
    nblocks0, nblocks1 = nrows // bs0, ncols // bs1
    if layout in (torch.sparse_csr, torch.sparse_bsr):
        comp_dim, plain_dim = nblocks0, nblocks1
    else:  # csc / bsc
        comp_dim, plain_dim = nblocks1, nblocks0
    entries = batch + (nnz,)
    assert 0 <= nnz <= comp_dim * plain_dim
    counts = torch.full((comp_dim,), nnz // comp_dim, dtype=torch.long)
    counts[: nnz % comp_dim] += 1
    compressed = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    plain = torch.arange(nnz) - torch.repeat_interleave(compressed[:-1], counts)
    compressed = compressed.expand(batch + (comp_dim + 1,)).contiguous()
    plain = plain.expand(entries).contiguous()
    block_shape = (bs0, bs1) if bs0 > 1 else ()
    values = torch.randn(entries + block_shape, dtype=dtype, device=device)
    return compressed.to(device), plain.to(device), values


def _case_fn(shape, dtype):
    del dtype
    layout, shape_, nnz = shape
    if isinstance(layout, str):
        layout = getattr(torch, layout.removeprefix("torch."))
    yield base.BenchmarkCasePlan(
        shape={"input": shape_},
        params={"nnz": nnz, "layout": str(layout)},
        builder_args=(layout, shape_, nnz),
    )


def _build_inputs_fn(plan, dtype, device):
    layout, shape, nnz = plan.builder_args
    compressed, plain, values = _make_input(layout, shape, nnz, dtype, device)
    # The kwargs dict travels at the top level of the returned tuple so
    # unpack_to_args_kwargs places the tensors in args and the dict in kwargs;
    # size/layout/dtype/device must go through the dict (they are neither
    # tensors nor plain scalars), so torch_op and gems_op receive the exact
    # same (compressed, plain, values, size, dtype=..., layout=..., device=...)
    # call.
    return (
        compressed,
        plain,
        values,
        {
            "size": list(shape),
            "layout": layout,
            "dtype": dtype,
            "device": device,
        },
    )


class SparseCompressedTensorBenchmark(OperatorBenchmark):
    # Sparse constructor; there are no meaningful dense shapes in
    # core_shapes.yaml, so benchmark dedicated (layout, shape, nnz) triples.
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_SPARSE_COMPRESSED_SHAPES)


@pytest.mark.sparse_compressed_tensor
def test_sparse_compressed_tensor():
    bench = SparseCompressedTensorBenchmark(
        op_name="sparse_compressed_tensor",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.sparse_compressed_tensor,
        gems_op=getattr(flag_gems, "sparse_compressed_tensor", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
