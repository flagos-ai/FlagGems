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

"""NVIDIA native baseline for `cholesky_solve_out`.

Mirrors benchmark/test_cholesky_solve.py (CholeskySolveOutBenchmark, a
base.Benchmark subclass):
    op_name  = "cholesky_solve_out"
    torch_op = torch.cholesky_solve
    dtypes   = REAL_DTYPES + COMPLEX_DTYPES

The test selects its callable and dtypes by vendor. On NVIDIA the branch taken
is the `else` one: torch_op is the real torch.cholesky_solve (the
`_composed_cholesky_solve_out` double triangular solve is Ascend-only), and
because the NVIDIA backend reports fp64_enabled=True the dtype list is
[float32, float64, complex64, complex128].

Shapes come from the subclass's CHOLESKY_SOLVE_OUT_CASES, a five-entry
((*batch, N, nrhs), upper) list -- not the 32-shape x 2-upper sweep used by the
non-out benchmark, and not core_shapes.yaml.

The Cholesky factor is built the way the benchmark builds it: a Hermitian
positive-definite A = B @ B^H + 0.1 * I, factored with torch.linalg.cholesky,
then transposed for the upper=True cases. The MUSA-only CPU setup path is not
reproduced.
"""

import torch

OP_NAME = "cholesky_solve_out"
DTYPES = [torch.float32, torch.float64, torch.complex64, torch.complex128]
IS_INPLACE = False
MUTATED_INPUTS = ("out",)
NATIVE_UNAVAILABLE_REASON = "torch.cholesky_solve is unavailable on this torch build"

# CholeskySolveOutBenchmark.set_shapes -- ((*batch, N, nrhs), upper)
_SHAPES = [
    ((16, 4), False),
    ((64, 1), True),
    ((64, 33), False),
    ((256, 128), True),
    ((8, 128, 16), False),
]


def _resolve(path):
    """Walk a dotted attribute path from the torch root; None when absent.

    torch.ops.aten.<name>.<overload> raises for an operator this build does not
    carry, so the lookup is guarded.
    """
    obj = torch
    try:
        for part in path.split(".")[1:]:
            obj = getattr(obj, part)
    except (AttributeError, RuntimeError):
        return None
    return obj


def native():
    op = _resolve("torch.cholesky_solve")
    return op if callable(op) else None


def grid():
    return [{"shape": list(shape), "upper": upper} for shape, upper in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = list(binding["shape"])
    upper = binding["upper"]
    *batch_dims, n, nrhs = shape
    b_mat = torch.randn(*batch_dims, n, n, dtype=dtype, device=device)
    eye = torch.eye(n, dtype=dtype, device=device)
    for _ in batch_dims:
        eye = eye.unsqueeze(0)
    a = b_mat @ b_mat.mH + eye * 0.1
    lower = torch.linalg.cholesky(a)
    factor = lower.mH.contiguous() if upper else lower
    rhs = torch.randn(*batch_dims, n, nrhs, dtype=dtype, device=device)
    return (rhs, factor, upper), {"out": torch.empty_like(rhs)}


def key_shape(binding):
    shape = "x".join(str(d) for d in binding["shape"])
    return f"{shape}_upper{int(binding['upper'])}"


def config(binding, dtype):
    shape = list(binding["shape"])
    *batch_dims, n, nrhs = shape
    rhs_shape = batch_dims + [n, nrhs]
    factor_shape = batch_dims + [n, n]
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": rhs_shape, "dtype": dt},
                {"shape": factor_shape, "dtype": dt},
                {"scalar": binding["upper"]},
            ],
            "kwargs": {"out": {"shape": rhs_shape, "dtype": dt}},
        },
        "outputs": {"out": {"shape": rhs_shape, "dtype": dt}},
        "dims": {
            "batch": batch_dims,
            "N": n,
            "nrhs": nrhs,
            "upper": binding["upper"],
        },
    }
