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

"""Benchmark for the backward of aten::_sparse_log_softmax on sparse COO input.

There is no public ATen schema spelled ``_sparse_log_softmax_backward``: the
gradient reaches the input through the forward's autograd node
(SparseLogSoftmaxBackward0 natively, ``_SparseLogSoftmax.backward`` in
FlagGems).  The measured op is therefore the full backward call
``torch.autograd.grad(out, input, grad_outputs=<coalesced sparse grad on the
output's support>)`` on both sides, which is exactly the contract the
integration ships.

The upstream gradient is built once per case as a coalesced sparse COO tensor
on the forward output's own support — ``torch.randn_like`` has no SparseCUDA
kernel, so the framework's generic backward path cannot be used and the
subclass below binds the prebuilt gradient instead.

The case grid mirrors the forward benchmark
(benchmark/test__sparse_log_softmax.py): every default shape times one sparse
pooling axis (``dim < sparse_dim``) and the dense block axes
(``dim >= sparse_dim``), in both supported dtypes.
"""

import pytest
import torch

import flag_gems

from . import base
from .generated_operator_utils import OperatorBenchmark
from .test__sparse_log_softmax import (
    SPARSE_DTYPES,
    SPARSE_SHAPES,
    _sparse_input,
    _specs_for,
)


def _case_fn(shape, dtype):
    del dtype
    for sparse_dim, dim in _specs_for(shape):
        yield base.BenchmarkCasePlan(
            shape={"input": list(shape), "sparse_dim": sparse_dim},
            params={"dim": dim, "half_to_float": False},
            builder_args=(shape, sparse_dim, dim),
        )


def _build_inputs_fn(plan, dtype, device):
    """Builds (input, dim, kwargs) for one case.

    The input carries requires_grad in place per the framework's backward
    contract (no pre-clone), and half_to_float must travel as a keyword: a
    positional False would bind the ScalarType parameter of the .int overload.
    """
    shape, sparse_dim, dim = plan.builder_args
    inp = _sparse_input(shape, sparse_dim, dtype, device, dim)
    inp.requires_grad_(True)
    return inp, dim, {"half_to_float": False}


class SparseLogSoftmaxBackwardBenchmark(OperatorBenchmark):
    """Times the integrated autograd backward with a prebuilt sparse grad.

    ``_benchmark_callable`` runs the forward once per side, materializes the
    upstream gradient on the output's coalesced support outside the timed
    region, and then times only ``torch.autograd.grad`` — the same callable the
    framework's generic backward path times, with a sparse ``grad_outputs``
    because the SparseCUDA backend has no ``normal_`` for ``randn_like``.
    """

    def _benchmark_callable(self, op, *args, **kwargs):
        out = op(*args, **kwargs)
        grad_out = torch.sparse_coo_tensor(
            out._indices(),
            torch.randn(out._values().shape, dtype=out.dtype, device=out.device),
            out.shape,
            device=out.device,
            dtype=out.dtype,
            is_coalesced=True,
        )
        xs = [x for x in args if torch.is_tensor(x) and x.requires_grad]
        fn = lambda: torch.autograd.grad(  # noqa: E731
            (out,), tuple(xs), grad_outputs=(grad_out,), retain_graph=True
        )
        return fn, xs

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=SPARSE_SHAPES)


@pytest.mark.sparse_log_softmax_backward
def test__sparse_log_softmax_backward():
    bench = SparseLogSoftmaxBackwardBenchmark(
        op_name="sparse_log_softmax_backward",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._sparse_log_softmax.default,
        gems_op=getattr(flag_gems, "_sparse_log_softmax", None),
        dtypes=SPARSE_DTYPES,
    )
    bench.run()
