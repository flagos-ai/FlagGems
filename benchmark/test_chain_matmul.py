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

"""Benchmark for ``aten::chain_matmul`` / ``aten::chain_matmul.out``.

``chain_matmul`` multiplies a *list* of rank-2 matrices in the order that
minimizes scalar multiplications:

    aten::chain_matmul(Tensor[] matrices) -> Tensor
    aten::chain_matmul.out(Tensor[] matrices, *, Tensor(a!) out) -> Tensor(a!)

No public Benchmark family models a TensorList chain product (``BlasBenchmark``
is bound to the fixed ``(b, m, n, k)`` GEMM contract and takes a legacy
``input_fn``), so both tests use the two-phase ``GenericBenchmark``
(``case_fn`` + ``build_inputs_fn``, never a bare ``input_fn``): one Workload per
chain, where a chain is a tuple of 2-D matrix shapes.

``torch_op=torch.ops.aten.chain_matmul`` is only the perf comparison reference;
the candidate comes from ``flag_gems`` via KernelGen's
``--override`` under ``op_name="chain_matmul"`` for both
ordinary and ``out=`` calls.

The chains are sized so the largest is a few GFLOP: enough to hide launch
overhead and expose the parenthesization cost, while every intermediate stays
well under a gigabyte. ``set_shapes`` replaces the dense 2-D/3-D shape file
entries (meaningless as a *sequence* of matrices) and ``set_more_shapes``
contributes nothing.
"""

import pytest
import torch

import flag_gems

from .generated_operator_utils import OperatorBenchmark
from . import base, consts, utils

# Each entry is one chain: a tuple of rank-2 matrix shapes whose inner
# dimensions are consistent ((m0, k0), (k0, k1), ..., (k_{n-1}, m_out)).
_CHAIN_BENCH_SHAPES = (
    ((512, 512), (512, 512)),  # square two-matrix GEMM
    ((1024, 1024), (1024, 1024)),  # large square pair
    ((512, 1024), (1024, 512), (512, 256)),  # three matrices, non-square
    (
        (256, 1024),
        (1024, 256),
        (256, 1024),
        (1024, 256),
    ),  # four matrices, alternating shape
    ((1024, 64), (64, 1024), (1024, 64)),  # skinny inner dims
    (
        (128, 2048),
        (2048, 128),
        (128, 2048),
        (2048, 128),
        (128, 512),
    ),  # five matrices
    ((1024, 256), (256, 1024)),  # rectangular pair
)

# The .out overload measures the same work plus a write into the caller's
# buffer; a representative subset keeps the run time in line with the default
# benchmark without changing the measured semantics.
_OUT_CHAIN_BENCH_SHAPES = _CHAIN_BENCH_SHAPES[:3]


class ChainMatmulBenchmark(OperatorBenchmark):
    """GenericBenchmark whose shapes are chains of rank-2 matrices.

    The shape source is a list of chains rather than the dense shape file, so
    ``set_shapes`` substitutes the local case list and ``set_more_shapes``
    contributes nothing extra.
    """

    BENCH_SHAPES = _CHAIN_BENCH_SHAPES

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=list(self.BENCH_SHAPES))

    def set_more_shapes(self):
        return []


class ChainMatmulOutBenchmark(ChainMatmulBenchmark):
    BENCH_SHAPES = _OUT_CHAIN_BENCH_SHAPES


def _case_fn(chain, dtype):
    # One Workload per chain: the shape entry lists every matrix of the chain
    # (they are the benchmark's unit of input, not a single dense tensor).
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"matrices": [list(matrix) for matrix in chain]},
        params={"num_matrices": len(chain)},
        builder_args=(chain,),
    )


def _build_inputs_fn(plan, dtype, device):
    chain = plan.builder_args[0]
    matrices = [utils.generate_tensor_input(shape, dtype, device) for shape in chain]
    return matrices, {}


def _build_inputs_fn_out(plan, dtype, device):
    chain = plan.builder_args[0]
    matrices = [utils.generate_tensor_input(shape, dtype, device) for shape in chain]
    out = torch.empty((chain[0][0], chain[-1][1]), dtype=dtype, device=device)
    return matrices, {"out": out}


@pytest.mark.chain_matmul
@pytest.mark.chain_matmul_benchmark
def test_chain_matmul():
    bench = ChainMatmulBenchmark(
        op_name="chain_matmul",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.chain_matmul,
        # KernelGen injects the candidate via --override; the
        # direct module callable may not exist until the op is merged, in which
        # case the benchmark reports no candidate and falls back to the
        # dispatcher reference (torch_op).
        gems_op=getattr(flag_gems, "chain_matmul", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.chain_matmul_out
@pytest.mark.chain_matmul_benchmark
def test_chain_matmul_out():
    bench = ChainMatmulOutBenchmark(
        op_name="chain_matmul",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn_out,
        torch_op=torch.ops.aten.chain_matmul.out,
        gems_op=getattr(flag_gems, "chain_matmul", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
