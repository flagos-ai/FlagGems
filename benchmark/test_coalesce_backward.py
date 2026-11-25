# Copyright 2026, The FlagOS Contributors.
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

# (sparse shape, nnz, seed) workload descriptors, mirroring the sibling _coalesce
# benchmark. The restore work scales with nnz; nnz > numel(shape) guarantees
# duplicate coordinates and therefore a non-trivial key-match restore.
_COALESCE_BACKWARD_SHAPES = [
    ((1024, 1024), 65536, 2026),
    ((1024, 1024), 262144, 2027),
    ((1024, 1024), 1048576, 2028),
    ((4096, 4096), 1048576, 2029),
    ((2048, 2048), 2097152, 2030),
    ((256, 256, 256), 1048576, 2031),
]


def _make_uncoalesced_coo(sparse_shape, nnz, dtype, device, seed):
    # Draw coordinates with replacement so the input is uncoalesced (the CUDA
    # reference asserts on that) and the restore has to scatter by key match.
    gen = torch.Generator("cpu").manual_seed(seed)
    indices = torch.stack(
        [
            torch.randint(0, dim, (nnz,), dtype=torch.long, generator=gen)
            for dim in sparse_shape
        ]
    ).to(device)
    values = torch.randn(nnz, dtype=dtype, device=device)
    return torch.sparse_coo_tensor(indices, values, sparse_shape, device=device)


def _native_coalesce_backward(grad_output, self_):
    """Native reference with the CoalesceBackward0 semantics.

    ATen has no standalone ``aten::_coalesce_backward`` schema: the backward
    lives in the autograd node of ``_coalesce`` (derivatives.yaml ``self:
    grad``). Routing the upstream sparse gradient through ``autograd.grad`` on
    the coalesced forward reproduces it exactly, so both sides run the same
    native path for the latency baseline.
    """
    values = self_._values().detach().clone().requires_grad_(True)
    input_with_grad = torch.sparse_coo_tensor(
        self_._indices(), values, self_.shape, device=self_.device
    )
    coalesced = torch.ops.aten._coalesce(input_with_grad)
    (restored,) = torch.autograd.grad(coalesced, values, grad_outputs=grad_output)
    return torch.sparse_coo_tensor(
        self_._indices(), restored, self_.shape, device=self_.device
    )


class CoalesceBackwardBenchmark(base.Benchmark):
    """``aten::_coalesce`` backward over duplicate-bearing sparse COO tensors.

    ``coalesce_backward`` is a sparse op with no meaningful dense shapes, so the
    benchmark sweeps the sibling (sparse_shape, nnz) workload descriptors.
    """

    def set_shapes(self, shape_file_path=None):
        # (sparse_shape, nnz) pairs; nnz > numel(shape) guarantees duplicates.
        self.shapes = _COALESCE_BACKWARD_SHAPES

    def get_input_iter(self, cur_dtype):
        for sparse_shape, nnz, seed in self.shapes:
            inp = _make_uncoalesced_coo(sparse_shape, nnz, cur_dtype, self.device, seed)
            coalesced = torch.ops.aten._coalesce(inp)
            grad_output = torch.sparse_coo_tensor(
                coalesced._indices(),
                torch.randn(coalesced._nnz(), dtype=cur_dtype, device=self.device),
                coalesced.shape,
                device=self.device,
            )
            yield grad_output, inp


@pytest.mark.coalesce_backward
def test_coalesce_backward():
    bench = CoalesceBackwardBenchmark(
        op_name="coalesce_backward",
        torch_op=_native_coalesce_backward,
        gems_op=flag_gems.coalesce_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
