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

from . import base

VENDOR = flag_gems.vendor_name


_LINALG_DET_SHAPES = [
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
    (4, 32, 32),
    (8, 64, 64),
]

_LINALG_DET_DTYPES = [torch.float32]


class LinalgDetBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = _LINALG_DET_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            A = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield (A,)


@pytest.mark.underscore_linalg_det
def test_underscore_linalg_det():
    def _torch_linalg_det(A):
        return torch.ops.aten._linalg_det(A)

    def _gems_linalg_det(A):
        return flag_gems.ops._linalg_det(A)

    bench = LinalgDetBenchmark(
        op_name="_linalg_det",
        torch_op=_torch_linalg_det,
        dtypes=_LINALG_DET_DTYPES,
    )
    bench.set_gems(_gems_linalg_det)
    bench.run()


_LINALG_DET_BACKWARD_DTYPES = [torch.float32]


class LinalgDetBackwardBenchmark(base.Benchmark):
    """Backward benchmark for aten::_linalg_det.

    The forward emits the determinant plus the non-differentiable LU/pivot
    metadata, so the timed closure has to differentiate only the determinant.
    ``base.Benchmark`` assumes a single differentiable output, hence the
    explicit ``get_latency`` override: it times ``torch.autograd.grad`` of the
    determinant with respect to the input matrix.
    """

    def set_shapes(self, shape_file_path=None):
        self.shapes = _LINALG_DET_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            A = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield (A,)

    def get_latency(self, op, *args, **kwargs):
        A = args[0]
        work = A.detach().clone().requires_grad_(True)
        result = op(work)
        det = result[0] if isinstance(result, tuple) else result
        dout = torch.randn_like(det)

        def run():
            return torch.autograd.grad(
                (det,), (work,), grad_outputs=(dout,), retain_graph=True
            )

        return self._time_callable(run, None)


@pytest.mark.linalg_det_backward
def test_underscore_linalg_det_backward():
    def _torch_linalg_det(A):
        return torch.ops.aten._linalg_det(A)

    def _gems_linalg_det(A):
        return flag_gems.ops._linalg_det(A)

    bench = LinalgDetBackwardBenchmark(
        op_name="_linalg_det",
        torch_op=_torch_linalg_det,
        dtypes=_LINALG_DET_BACKWARD_DTYPES,
        is_backward=True,
    )
    bench.set_gems(_gems_linalg_det)
    bench.run()
