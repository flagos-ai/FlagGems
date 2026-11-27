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

from . import base


class LinalgEigvalsBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        # Eigenvalues require square matrices
        self.shapes = [(32, 32), (64, 64), (128, 128), (256, 256), (512, 512)]

    def get_input_iter(self, cur_dtype):
        with_out = self.op_name == "linalg_eigvals_out"
        for shape in self.shapes:
            x = torch.randn(shape, dtype=cur_dtype, device=self.device)
            if with_out:
                out = torch.empty(
                    shape[:-1],
                    dtype=torch.promote_types(cur_dtype, torch.complex64),
                    device=self.device,
                )
                yield x, {"out": out}
            else:
                yield x,


@pytest.mark.linalg_eigvals
def test_linalg_eigvals():
    bench = LinalgEigvalsBenchmark(
        op_name="linalg_eigvals",
        torch_op=torch.linalg.eigvals,
        # _linalg_eigvals requires float32 for cuSOLVER eigenvalue computation
        dtypes=[torch.float32],
    )
    bench.run()


@pytest.mark.linalg_eigvals_out
def test_linalg_eigvals_out():
    bench = LinalgEigvalsBenchmark(
        op_name="linalg_eigvals_out",
        torch_op=torch.linalg.eigvals,
        # _linalg_eigvals requires float32 for cuSOLVER eigenvalue computation
        dtypes=[torch.float32],
    )
    bench.set_gems(flag_gems.linalg_eigvals_out)
    bench.run()
