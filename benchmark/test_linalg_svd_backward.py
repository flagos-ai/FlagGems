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

"""Backward benchmark for ``aten::_linalg_svd``.

The forward returns three differentiable outputs, so ``base.Benchmark``'s
single-output ``is_backward`` path (which differentiates one tensor) cannot
express this operator. The class below therefore keeps the default shape
resolution of the forward benchmark and times the real backward closure:
``torch.autograd.grad`` over ``(U, S, Vh)`` with respect to the input matrix,
using three independent upstream gradients.

The candidate path calls ``flag_gems._linalg_svd`` directly, which wraps the
public forward entry in the custom autograd Function; the reference path calls
``torch.linalg.svd`` (the same internal ``aten::_linalg_svd`` primitive).
Timing covers only the backward pass: factor construction and the first
forward stay outside the timed closure.
"""

import math

import pytest
import torch

import flag_gems

from . import base

# Same shape set the forward benchmark uses. The Triton SVD kernels only cover
# float32 CUDA matrices and the full_matrices path needs max(M, N) <= 64.
LINALG_SVD_BACKWARD_SHAPES = [(8, 8), (16, 16), (32, 32), (64, 64)]
LINALG_SVD_BACKWARD_DTYPES = [torch.float32]


class LinalgSvdBackwardBenchmark(base.Benchmark):
    """Times the autograd backward of the SVD factors w.r.t. the input."""

    DEFAULT_SHAPE_DESC = "(*B), M, N"

    def set_shapes(self, shape_file_path=None):
        self.shapes = LINALG_SVD_BACKWARD_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield (inp,)

    def get_latency(self, op, *args, **kwargs):
        inp = args[0]
        work = inp.detach().clone().requires_grad_(True)
        u, s, vh = op(work)
        grad_u = torch.randn_like(u)
        grad_s = torch.randn_like(s)
        grad_vh = torch.randn_like(vh)

        def run():
            return torch.autograd.grad(
                (u, s, vh),
                (work,),
                (grad_u, grad_s, grad_vh),
                retain_graph=True,
            )

        # Correctness gate before timing: a stale or broken kernel must not be
        # timed as a fast candidate. The reference is float64 on the same
        # factors, so it is independent of the forward's own rounding.
        grad_a = run()[0]
        u64 = u.detach().double()
        vh64 = vh.detach().double()
        gu64 = grad_u.double()
        gvh64 = grad_vh.double()
        gs64 = grad_s.double()
        s64 = s.detach().double()
        sinv = 1.0 / s64
        gram_u = u64.mH @ gu64
        gram_v = vh64 @ gvh64.mH
        f = 1.0 / (s64.unsqueeze(-2) ** 2 - s64.unsqueeze(-1) ** 2)
        f.diagonal(0, -2, -1).zero_()
        t = (
            (f * (gram_u - gram_u.mH)) * s64.unsqueeze(-2)
            + s64.unsqueeze(-1) * (f * (gram_v - gram_v.mH))
            + torch.diag_embed(gs64)
        )
        c = t - gram_u * sinv.unsqueeze(-2) - sinv.unsqueeze(-1) * gram_v.mH
        expected = (
            u64 @ c @ vh64
            + (gu64 * sinv.unsqueeze(-2)) @ vh64
            + (u64 * sinv.unsqueeze(-2)) @ gvh64
        ).float()
        rel = float((grad_a - expected).norm() / expected.norm())
        if not math.isfinite(rel) or rel > 5e-4:
            pytest.fail(f"_linalg_svd backward relative error {rel:.3e} exceeds 5e-4")

        return self._time_callable(run, None)


@pytest.mark.linalg_svd_backward
def test_linalg_svd_backward():
    def _torch_svd(A):
        return torch.linalg.svd(A, full_matrices=False)

    def _gems_svd(A):
        return flag_gems._linalg_svd(A, full_matrices=False)

    bench = LinalgSvdBackwardBenchmark(
        op_name="linalg_svd_backward",
        torch_op=_torch_svd,
        dtypes=LINALG_SVD_BACKWARD_DTYPES,
    )
    bench.set_gems(_gems_svd)
    bench.run()
