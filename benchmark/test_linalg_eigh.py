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

# Benchmark shapes for linalg_eigh.
#
# The op has on-device Triton paths:
#   - n == 2       -> closed-form `_eig_2x2_kernel`.
#   - 3 <= n <= 64  -> register-resident Jacobi.
#   - n > 64       -> global-memory pair-wise Jacobi (correct but slower; kept
#                    small here to bound benchmark runtime from per-pair launches).

# n == 2: closed-form kernel.
EIG_BENCHMARK_2X2_SHAPES = [(2, 2)]

# n > 2: Jacobi path (register-resident up to 64, global above).
EIG_BENCHMARK_SHAPES = [
    (8, 8),
    (32, 32),
    (64, 64),
    (96, 96),
]


class LinalgEighBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = EIG_BENCHMARK_2X2_SHAPES + EIG_BENCHMARK_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            A = torch.randn(shape, dtype=cur_dtype, device=self.device)
            A = (A + A.transpose(-2, -1)) / 2
            yield A,


@pytest.mark.linalg_eigh
def test_linalg_eigh():
    bench = LinalgEighBenchmark(
        op_name="linalg_eigh",
        torch_op=torch.linalg.eigh,
        dtypes=[torch.float32],
    )
    bench.run()


class LinalgEighBackwardBenchmark(base.Benchmark):
    """Backward of ``linalg_eigh`` measured through the real autograd entry.

    A single input tensor is enough: the shared measurement path sets
    ``requires_grad`` on it, runs the forward once outside the timer, and then
    times ``torch.autograd.grad`` on the two differentiable outputs.  The
    upstream gradients come from that path's ``randn_like`` outputs, which are
    identical for the native baseline and the FlagGems candidate because both
    consume the same ``args``.

    Note on semantics: the eigenvector gradient produced by
    ``torch.autograd.grad`` is only meaningful for a loss that is invariant
    under a per-column sign flip, and two valid eigenvector branches differ by
    exactly that flip.  Benchmarking the raw ``randn_like`` gradient is still
    correct as a *timing* measurement (both sides run the same math on their own
    eigenpairs), but its numeric value is not a cross-implementation
    comparable; the accuracy tests use a sign-invariant loss for that.
    """

    MAX_BACKWARD_ELEMENTS = None

    def set_shapes(self, shape_file_path=None):
        # The forward class hard-codes its own list instead of reading
        # core_shapes.yaml; the backward keeps exactly those five shapes so the
        # two directions are measured on the same problems.
        self.shapes = EIG_BENCHMARK_2X2_SHAPES + EIG_BENCHMARK_SHAPES

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            # ``_case_from_plan``/``_case_id`` live on Benchmark and attach the
            # pytest nodeid, so go through them rather than yielding a raw plan.
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"A": list(shape)},
                    params={"uplo": "L"},
                    builder_args=(shape,),
                ),
            )

    def build_inputs(self, case):
        # ``case`` is a BenchmarkCaseSpec; ``builder_args`` carries the plan's
        # own builder tuple, whose single element is the (rows, cols) shape.
        plan = case.builder_args[0]
        (shape,) = plan.builder_args
        torch.manual_seed(0)
        a = torch.randn(shape, dtype=case.dtype, device=self.device)
        # eigh reads one triangle; make the stored triangle symmetric so the
        # native and Gems forwards decompose the same operator.
        a = (a + a.transpose(-2, -1)) / 2
        return a, {}

    def _benchmark_callable(self, op, *args, **kwargs):
        """Time only the eigen-decomposition backward.

        ``eigh`` returns two differentiable tensors, which the shared
        one-output ``is_backward`` path cannot express (it calls
        ``randn_like(out)``), so the upstream gradients are built here, once,
        outside the timed closure.  They are a pure function of ``A`` and are
        therefore byte-identical for the native baseline and the Gems
        candidate.  ``grad_V`` comes from a sign-invariant reconstruction loss,
        the only kind of eigenvector gradient that is well defined.
        """
        a = args[0]
        w, v = op(a)
        grad_w, grad_v = _upstream_gradients(w, v, a)
        fn = lambda: torch.autograd.grad(
            (w, v), (a,), (grad_w, grad_v), retain_graph=True
        )
        return fn, [a]


def _upstream_gradients(w, v, a):
    """Sign-invariant upstream gradients of the eigenpairs of ``a``.

    Built from ``(w, v)`` themselves so the same numbers are handed to the
    native baseline and the Gems candidate regardless of which eigenvector
    branch each forward picked: the loss reconstructs ``A`` from the eigenpairs,
    which is invariant under a per-column sign flip.
    """
    n = a.shape[-1]
    torch.manual_seed(1)
    target = torch.randn(a.shape, dtype=a.dtype, device=a.device)
    target = (target + target.transpose(-2, -1)) / 2
    coef = torch.randn(n, dtype=a.dtype, device=a.device)
    wd = w.detach().requires_grad_(True)
    vd = v.detach().requires_grad_(True)
    rebuilt = vd @ torch.diag_embed(wd) @ vd.transpose(-2, -1)
    loss = ((rebuilt - target) ** 2).sum() + (coef * wd).sum()
    grad_w, grad_v = torch.autograd.grad(loss, (wd, vd))
    return grad_w.detach().contiguous(), grad_v.detach().contiguous()


@pytest.mark.linalg_eigh_backward
def test_linalg_eigh_backward():
    """Backward of linalg_eigh on the forward benchmark's five core shapes.

    ``float32`` is the forward test's own dtype; ``float64`` is added because
    the implementation and the accuracy tests cover it as well.
    """
    bench = LinalgEighBackwardBenchmark(
        op_name="linalg_eigh",
        torch_op=torch.linalg.eigh,
        dtypes=[torch.float32, torch.float64],
        is_backward=True,
    )
    bench.set_gems(flag_gems.linalg_eigh)
    bench.run()
