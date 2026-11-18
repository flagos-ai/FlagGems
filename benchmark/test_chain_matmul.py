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

"""Benchmark for ``aten::chain_matmul`` / ``aten::chain_matmul.out``.

``chain_matmul`` multiplies a *list* of rank-2 matrices in the order that
minimizes scalar multiplications, so one benchmark case is a whole chain, not a
single tensor shape. No pointwise/BLAS Benchmark family models that
(``BlasBenchmark`` is bound to the fixed ``(b, m, n, k)`` GEMM contract), so
both tests use the two-phase ``GenericBenchmark`` (``case_fn`` +
``build_inputs_fn``) with one Workload per chain.

``set_shapes`` substitutes the local chain list: the dense 2-D/3-D entries of
``core_shapes.yaml`` are meaningless as a *sequence* of matrices, and the
catch-all ``Benchmark:`` entry in that file would otherwise win the MRO lookup
in ``base.Benchmark.set_shapes``. ``set_more_shapes`` contributes nothing for
the same reason.
"""

import pytest
import torch

import flag_gems

from . import base, consts, utils

# Chain dimensions (d0, d1, ..., dn): n matrices of shapes (d0, d1) ... and an
# (d0, dn) output. The narrow inner dimensions are what make the optimal
# parenthesization non-trivial. These were the comprehensive-level cases of the
# first revision of this benchmark; they are in core now so the optimization
# targets them (review request on PR #7028).
CORE_CHAINS = [
    (1024, 64, 1024),
    (2048, 32, 2048, 32),
    (512, 2048, 64, 1024),
    (1024, 128, 2048, 64, 512),
    (256, 1024, 64, 2048, 128, 512),
    (512, 128, 1024, 64, 2048, 32, 512),
]

# Each entry is one chain: a tuple of rank-2 matrix shapes whose inner
# dimensions are consistent ((m0, k0), (k0, k1), ..., (k_{n-1}, m_out)).
# The chains are sized so the largest is a few GFLOP: enough to hide launch
# overhead and expose the parenthesization cost, while every intermediate stays
# well under a gigabyte.
CHAIN_SHAPES = [
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
] + [tuple(zip(dims, dims[1:])) for dims in CORE_CHAINS]

# The .out overload measures the same work plus a write into the caller's
# buffer; a representative subset keeps the run time in line with the default
# benchmark without changing the measured semantics.
OUT_CHAIN_SHAPES = CHAIN_SHAPES[:3]


class ChainMatmulBenchmark(base.GenericBenchmark):
    """GenericBenchmark whose shapes are chains of rank-2 matrices."""

    BENCH_SHAPES = CHAIN_SHAPES
    DEFAULT_SHAPE_DESC = "chain of (M, K) matrix shapes"

    def set_shapes(self, shape_file_path=None):
        self.shapes = list(self.BENCH_SHAPES)
        self.shape_desc = self.DEFAULT_SHAPE_DESC

    def set_more_shapes(self):
        return []


class ChainMatmulOutBenchmark(ChainMatmulBenchmark):
    BENCH_SHAPES = OUT_CHAIN_SHAPES


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
def test_chain_matmul():
    bench = ChainMatmulBenchmark(
        op_name="chain_matmul",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.chain_matmul,
        gems_op=flag_gems.chain_matmul,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.chain_matmul_out
def test_chain_matmul_out():
    bench = ChainMatmulOutBenchmark(
        op_name="chain_matmul_out",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn_out,
        torch_op=torch.ops.aten.chain_matmul.out,
        gems_op=flag_gems.chain_matmul,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
