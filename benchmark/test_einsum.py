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

from functools import partial

import pytest
import torch

from . import base, consts
from .generated_operator_utils import OperatorBenchmark


class EinsumBenchmark(OperatorBenchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["tflops"]
    DEFAULT_SHAPES = [(1, 512, 512, 512), (1, 1024, 1024, 1024), (16, 512, 512, 512)]

    def __init__(self, *args, batched=False, **kwargs):
        self.batched = batched
        super().__init__(*args, **kwargs)

    def set_more_shapes(self):
        return []

    def set_shapes(self, *args, **kwargs):
        self.shapes = self.DEFAULT_SHAPES

    def get_tflops(self, op, *args, **kwargs):
        A, B = args[1], args[2]
        if self.batched:
            return A.shape[0] * A.shape[1] * B.shape[2] * A.shape[2] * 2
        return A.shape[0] * B.shape[1] * A.shape[1] * 2


class EinsumGenericBenchmark(OperatorBenchmark):
    def set_shapes(self, *args, **kwargs):
        pass  # keep shapes set by caller


def _case_fn(shape, dtype, *, equation):
    del dtype
    if equation in ("ij,jk->ik", "bij,bjk->bik"):
        b, m, n, k = shape
        operands = (
            ((b, m, k), (b, k, n))
            if equation == "bij,bjk->bik"
            else ((m, k), (k, n))
        )
    elif equation == "i,i->":
        operands = (tuple(shape), tuple(shape))
    elif equation == "i,j->ij":
        m, n = shape
        operands = ((m,), (n,))
    elif equation == "...ij,...jk->...ik":
        b, h, m, k, n = shape
        operands = ((b, h, m, k), (b, h, k, n))
    else:
        operands = (tuple(shape),)
    yield base.BenchmarkCasePlan(
        shape={"operands": [list(dims) for dims in operands]},
        params={"equation": equation},
        builder_args=operands,
    )


def _build_inputs_fn(plan, dtype, device):
    operands = [
        torch.randn(dims, dtype=dtype, device=device) for dims in plan.builder_args
    ]
    return (plan.params["equation"], *operands)


@pytest.mark.einsum
def test_einsum_matmul():
    bench = EinsumBenchmark(
        case_fn=partial(_case_fn, equation="ij,jk->ik"),
        build_inputs_fn=_build_inputs_fn,
        op_name="einsum",
        torch_op=torch.einsum,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.einsum
def test_einsum_bmm():
    bench = EinsumBenchmark(
        case_fn=partial(_case_fn, equation="bij,bjk->bik"),
        build_inputs_fn=_build_inputs_fn,
        op_name="einsum",
        torch_op=torch.einsum,
        dtypes=consts.FLOAT_DTYPES,
        batched=True,
    )
    bench.run()


@pytest.mark.einsum
def test_einsum_dot():
    bench = EinsumGenericBenchmark(
        case_fn=partial(_case_fn, equation="i,i->"),
        build_inputs_fn=_build_inputs_fn,
        op_name="einsum",
        torch_op=torch.einsum,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.shapes = [(1024,), (4096,), (65536,)]
    bench.run()


@pytest.mark.einsum
def test_einsum_outer():
    bench = EinsumGenericBenchmark(
        case_fn=partial(_case_fn, equation="i,j->ij"),
        build_inputs_fn=_build_inputs_fn,
        op_name="einsum",
        torch_op=torch.einsum,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.shapes = [(1024, 1024), (4096, 4096)]
    bench.run()


@pytest.mark.einsum
def test_einsum_trace():
    bench = EinsumGenericBenchmark(
        case_fn=partial(_case_fn, equation="ii->"),
        build_inputs_fn=_build_inputs_fn,
        op_name="einsum",
        torch_op=torch.einsum,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.shapes = [(1024, 1024), (4096, 4096)]
    bench.run()


@pytest.mark.einsum
def test_einsum_diagonal():
    bench = EinsumGenericBenchmark(
        case_fn=partial(_case_fn, equation="ii->i"),
        build_inputs_fn=_build_inputs_fn,
        op_name="einsum",
        torch_op=torch.einsum,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.shapes = [(1024, 1024), (4096, 4096)]
    bench.run()


@pytest.mark.einsum
def test_einsum_transpose():
    bench = EinsumGenericBenchmark(
        case_fn=partial(_case_fn, equation="ij->ji"),
        build_inputs_fn=_build_inputs_fn,
        op_name="einsum",
        torch_op=torch.einsum,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.shapes = [(1024, 1024), (4096, 4096)]
    bench.run()


@pytest.mark.einsum
def test_einsum_sum_all():
    bench = EinsumGenericBenchmark(
        case_fn=partial(_case_fn, equation="ijk->"),
        build_inputs_fn=_build_inputs_fn,
        op_name="einsum",
        torch_op=torch.einsum,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.shapes = [(64, 64, 64), (128, 128, 128)]
    bench.run()


@pytest.mark.einsum
def test_einsum_sum_dim():
    bench = EinsumGenericBenchmark(
        case_fn=partial(_case_fn, equation="ijk->j"),
        build_inputs_fn=_build_inputs_fn,
        op_name="einsum",
        torch_op=torch.einsum,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.shapes = [(64, 64, 64), (128, 128, 128)]
    bench.run()


@pytest.mark.einsum
def test_einsum_ellipsis():
    bench = EinsumGenericBenchmark(
        case_fn=partial(_case_fn, equation="...ij,...jk->...ik"),
        build_inputs_fn=_build_inputs_fn,
        op_name="einsum",
        torch_op=torch.einsum,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.shapes = [(2, 4, 64, 64, 128), (2, 8, 128, 128, 256)]
    bench.run()
