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

# Square system sizes. A sparse pattern with diagonal dominance is used so the
# systems are invertible and well conditioned.
SPSOLVE_SIZES = [64, 128, 256, 512, 1024]


def _spsolve_torch_dense(A_csr, b):
    """Dense baseline: torch has no runnable native _spsolve here (CPU lacks a
    kernel, CUDA needs cuDSS), so compare against the equivalent dense solve.

    The dense solver has no half-precision path, so half inputs are solved in
    fp32 (matching the FlagGems implementation) and cast back."""
    out_dtype = A_csr.dtype
    compute_dtype = (
        torch.float32 if out_dtype in (torch.float16, torch.bfloat16) else out_dtype
    )
    X = torch.linalg.solve(A_csr.to_dense().to(compute_dtype), b.to(compute_dtype))
    return X.to(out_dtype)


class SpsolveBenchmark(base.Benchmark):
    """Benchmark for ``_spsolve``.

    Native ``_spsolve`` only supports a 1-D right-hand side with ``left=True``,
    so the benchmark uses a vector RHS. Both baseline and FlagGems paths densify
    then solve; the FlagGems path uses a Triton scatter kernel for the
    CSR-to-dense conversion and the FlagGems Triton ``linalg_solve`` kernel.
    """

    def set_shapes(self, shape_file_path=None):
        self.shapes = SPSOLVE_SIZES

    def get_input_iter(self, cur_dtype):
        for n in self.shapes:
            dense = torch.randn((n, n), dtype=cur_dtype, device=self.device)
            mask = torch.rand((n, n), device=self.device) > 0.1
            dense = dense.masked_fill(mask, 0.0)
            diag = dense.abs().sum(dim=1) + 1.0
            dense = dense - torch.diag(torch.diagonal(dense)) + torch.diag(diag)
            A_csr = dense.to_sparse_csr()
            b = torch.randn((n,), dtype=cur_dtype, device=self.device)
            yield A_csr, b


@pytest.mark.spsolve
def test_spsolve():
    bench = SpsolveBenchmark(
        op_name="spsolve",
        torch_op=_spsolve_torch_dense,
        dtypes=consts.FLOAT_DTYPES,
        gems_op=flag_gems.spsolve,
    )
    bench.run()
