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

# (M, K, N) shapes for the sparse-dense matmul benchmark.
SMM_SHAPES = [
    (64, 64, 64),
    (128, 128, 128),
    (256, 256, 256),
    (512, 128, 256),
    (1024, 256, 512),
]


def _make_sparse_coo(M, K, dtype, device, density=0.3, seed=0):
    torch.manual_seed(seed)
    dense = torch.randn(M, K, dtype=dtype, device=device)
    mask = (torch.rand(M, K, device=device) < density).to(torch.bool)
    return (dense * mask).to_sparse().coalesce()


def _torch_smm_ref(self, mat):
    """A native-CUDA baseline for ``torch.smm``.

    ``torch.smm`` has no native CUDA kernel (it decomposes to ``sspaddmm``,
    which is NYI on CUDA), so the closest equivalent in native PyTorch is to
    materialize the sparse operand to dense, run the dense matmul, then
    re-sparsify (and coalesce) the result. This baseline performs the same
    logical work as the GEMS kernel and is the meaningful comparison target.
    """
    return (self.to_dense() @ mat).to_sparse().coalesce()


class SmmBenchmark(base.Benchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["tflops"]

    def set_shapes(self, shape_file_path=None):
        self.shapes = SMM_SHAPES

    def get_input_iter(self, cur_dtype):
        density = 0.3
        for M, K, N in self.shapes:
            sparse = _make_sparse_coo(M, K, cur_dtype, self.device, density=density)
            dense = torch.randn(K, N, dtype=cur_dtype, device=self.device)
            yield sparse, dense

    def get_tflops(self, op, *args, **kwargs):
        sparse, dense = args[0], args[1]
        # 2 * M * K * N flops for the (materialized) dense matmul.
        return 2 * sparse.shape[0] * sparse.shape[1] * dense.shape[1]


@pytest.mark.smm
def test_smm():
    bench = SmmBenchmark(
        op_name="smm",
        torch_op=_torch_smm_ref,
        gems_op=flag_gems.smm,
        # ``torch.smm`` is only implemented for float32 on the CUDA backend, so the
        # GEMS implementation mirrors that restriction; there is no
        # ``consts.FLOAT_DTYPES`` parametrization here.
        dtypes=[torch.float32],
    )
    bench.run()
