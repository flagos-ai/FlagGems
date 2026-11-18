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

"""NVIDIA baseline for cholesky_solve (torch native).

Self-contained module; depends only on torch. ``native()`` returns
``torch.ops.aten.cholesky_solve``, the ATen kernel the operator is a port of;
the two-``solve_triangular`` composition that the benchmark substitutes on
Ascend (no native NPU kernel) is not needed here, since baselines are
collected on NVIDIA. ``CholeskySolveBenchmark`` overrides ``set_shapes``, so
``core_shapes.yaml`` is not consulted: the 32 ``CHOLESKY_SOLVE_SHAPES`` entries
are each run with ``upper`` both ``False`` and ``True`` (64 cases). Dtypes
(``REAL_DTYPES + COMPLEX_DTYPES``: fp32/fp64/complex64/complex128 on NVIDIA)
and the ``A = B @ B.mH + 0.1 * I`` → ``torch.linalg.cholesky`` →
``(rhs, factor, upper)`` input construction are copied from
``benchmark/test_cholesky_solve.py``.
"""

import torch

OP_NAME = "cholesky_solve"
DTYPES = [torch.float32, torch.float64, torch.complex64, torch.complex128]
IS_INPLACE = False


# benchmark/test_cholesky_solve.py::CHOLESKY_SOLVE_SHAPES
# (32 shapes, each run with upper=False and upper=True)
_CASES = [
    # Single RHS: exposes the low-parallelism triangular-solve path.
    (8, 1),
    (16, 1),
    (32, 1),
    # Real small-gather single-RHS range (BLOCK_N=64).
    (33, 1),
    (48, 1),
    (63, 1),
    (64, 1),
    (128, 1),
    (256, 1),
    # Small-N small-RHS fused path coverage.
    (16, 2),
    (16, 4),
    (32, 4),
    # RHS sweep around BLOCK_RHS boundaries and tail cases.
    (64, 4),
    (64, 8),
    (64, 16),
    (64, 31),
    (64, 32),
    (64, 33),
    (64, 64),
    (64, 128),
    # Throughput-oriented larger systems.
    (128, 16),
    (128, 64),
    (256, 16),
    (256, 128),
    # Batched systems: important for occupancy with one batch per program tile.
    (16, 16, 1),
    (64, 16, 1),
    (256, 16, 1),
    (16, 16, 4),
    (16, 32, 4),
    (16, 32, 8),
    (32, 64, 16),
    (8, 128, 16),
]


def native():
    return torch.ops.aten.cholesky_solve


def grid():
    return [
        {"shape": shape, "upper": upper} for upper in (False, True) for shape in _CASES
    ]


def build_inputs(binding, dtype, device):
    # Mirrors benchmark/test_cholesky_solve.py::get_input_iter: factorize a
    # Hermitian positive-definite A = B @ B.mH + 0.1 * I, then pass L (lower)
    # or L.mH (upper) as the triangular factor.
    *batch_dims, n, nrhs = binding["shape"]
    B_mat = torch.randn(*batch_dims, n, n, dtype=dtype, device=device)
    eye = torch.eye(n, dtype=dtype, device=device)
    for _ in batch_dims:
        eye = eye.unsqueeze(0)
    A = B_mat @ B_mat.mH + eye * 0.1
    L = torch.linalg.cholesky(A)
    factor = L.mH.contiguous() if binding["upper"] else L
    rhs = torch.randn(*batch_dims, n, nrhs, dtype=dtype, device=device)
    return (rhs, factor), {"upper": binding["upper"]}


def key_shape(binding):
    shape = "x".join(str(dim) for dim in binding["shape"])
    return f"{shape}_upper{binding['upper']}"


def config(binding, dtype):
    # rhs/out keep the case shape (*batch, N, nrhs); factor is (*batch, N, N).
    shape = list(binding["shape"])
    factor_shape = shape[:-2] + [shape[-2], shape[-2]]
    return {
        "inputs": {
            "rhs": {"shape": shape, "dtype": str(dtype)},
            "factor": {"shape": factor_shape, "dtype": str(dtype)},
        },
        "outputs": {"out": {"shape": shape, "dtype": str(dtype)}},
        "dims": {"upper": binding["upper"]},
    }
