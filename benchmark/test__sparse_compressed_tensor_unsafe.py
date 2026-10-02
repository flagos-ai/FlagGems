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

# aten::_sparse_compressed_tensor_unsafe(compressed_indices, plain_indices, values,
#     int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None,
#     bool? pin_memory=None) -> Tensor
#
# The measured work is building one sparse compressed instance from its component
# tensors, so a case is a (layout, logical size, nnz) descriptor: the components
# scale with nnz while the logical matrix spans the requested extent. The reference
# and the candidate receive the identical call, so dtype/layout/device travel with
# the components.
_SPARSE_COMPRESSED_UNSAFE_SHAPES = [
    (torch.sparse_csr, (1024, 1024), 65536),
    (torch.sparse_csr, (1024, 1024), 262144),
    (torch.sparse_csc, (1024, 1024), 262144),
    (torch.sparse_csr, (4096, 4096), 1048576),
    (torch.sparse_bsr, (2048, 2048), 262144),
    (torch.sparse_bsc, (2048, 2048), 262144),
    (torch.sparse_csr, (8, 256, 256), 16384),
]

_ROW_MAJOR_LAYOUTS = (torch.sparse_csr, torch.sparse_bsr)
_BLOCK_LAYOUTS = (torch.sparse_bsr, torch.sparse_bsc)
_BLOCK_SIZE = 2


def _make_input(layout, shape, nnz, dtype, device):
    if len(shape) < 2:
        return (
            torch.zeros(1, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=dtype, device=device),
        )
    batch = shape[:-2]
    nrows, ncols = shape[-2], shape[-1]
    if layout in _BLOCK_LAYOUTS:
        bs0 = bs1 = _BLOCK_SIZE
    else:
        bs0 = bs1 = 1
    nblocks0, nblocks1 = nrows // bs0, ncols // bs1
    if layout in _ROW_MAJOR_LAYOUTS:
        comp_dim, plain_dim = nblocks0, nblocks1
    else:  # csc / bsc
        comp_dim, plain_dim = nblocks1, nblocks0
    assert 0 <= nnz <= comp_dim * plain_dim
    # Sorted unique entries in each compressed segment.
    counts = torch.full(
        (comp_dim,), nnz // comp_dim if comp_dim else 0, dtype=torch.long
    )
    if comp_dim:
        counts[: nnz % comp_dim] += 1
    compressed = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    plain = torch.arange(nnz) - torch.repeat_interleave(compressed[:-1], counts)
    compressed = compressed.expand(batch + (comp_dim + 1,)).contiguous()
    plain = plain.expand(batch + (nnz,)).contiguous()
    block_shape = (bs0, bs1) if bs0 > 1 else ()
    values = torch.randn(batch + (nnz, *block_shape), dtype=dtype, device=device)
    return compressed.to(device), plain.to(device), values


def _stored_descriptor(entry):
    """(layout, tensor_shape, nnz) when a plan entry is a stored descriptor."""
    if not isinstance(entry, (tuple, list)) or len(entry) != 3:
        return None
    layout, tensor_shape, nnz = entry
    if isinstance(layout, str):
        layout = getattr(torch, layout.removeprefix("torch."), None)
    if isinstance(layout, torch.layout) and isinstance(tensor_shape, (tuple, list)):
        return layout, tuple(int(dim) for dim in tensor_shape), int(nnz)
    return None


def _dense_descriptor(entry):
    """Map an inherited dense size onto a CSR descriptor of the same extent.

    The shared resolver lists dense pointwise sizes for names that
    core_shapes.yaml does not describe, and a compressed constructor needs sparse
    components, so the requested extent is kept and one entry per compressed row is
    stored. The unsafe factory also accepts rank-zero and rank-one metadata;
    those descriptors retain their original size and use empty components.
    """
    tensor_shape = tuple(int(dim) for dim in entry)
    if len(tensor_shape) < 2:
        return torch.sparse_csr, tensor_shape, 0
    rows, cols = tensor_shape[-2], tensor_shape[-1]
    return torch.sparse_csr, tensor_shape, (rows if cols > 0 else 0)


def _case_fn(shape, dtype):
    del dtype
    descriptor = _stored_descriptor(shape) or _dense_descriptor(shape)
    layout, tensor_shape, nnz = descriptor
    yield base.BenchmarkCasePlan(
        shape={"input": tensor_shape},
        params={"nnz": nnz, "layout": str(layout)},
        builder_args=(layout, tensor_shape, nnz),
    )


def _build_inputs_fn(plan, dtype, device):
    layout, tensor_shape, nnz = plan.builder_args
    compressed, plain, values = _make_input(layout, tensor_shape, nnz, dtype, device)
    # size/layout/dtype/device travel in the kwargs dict so unpack_to_args_kwargs
    # keeps the tensors in args and passes these as keywords, giving torch_op and
    # gems_op the exact same (compressed, plain, values, size, dtype, layout,
    # device) call.
    return (
        compressed,
        plain,
        values,
        {
            "size": list(tensor_shape),
            "layout": layout,
            "dtype": dtype,
            "device": device,
        },
    )


class SparseCompressedTensorUnsafeBenchmark(OperatorBenchmark):
    """Union this operator's descriptors into the shared shape resolution.

    core_shapes.yaml describes this name with dense pointwise sizes, which cannot
    express sparse components. The normal shared resolver still runs (so a
    caller-supplied shape file keeps precedence and a missing file still raises),
    and the native-valid (layout, size, nnz) descriptors are unioned into whatever
    it produced instead of replacing that list.
    """

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(
            base.Config.shape_file if shape_file_path is None else shape_file_path
        )
        self.shapes = list(
            dict.fromkeys(list(self.shapes) + _SPARSE_COMPRESSED_UNSAFE_SHAPES)
        )


@pytest.mark.sparse_compressed_tensor_unsafe
def test__sparse_compressed_tensor_unsafe():
    bench = SparseCompressedTensorUnsafeBenchmark(
        op_name="_sparse_compressed_tensor_unsafe",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._sparse_compressed_tensor_unsafe,
        gems_op=getattr(flag_gems, "_sparse_compressed_tensor_unsafe", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
