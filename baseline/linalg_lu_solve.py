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

"""NVIDIA native baseline for `linalg_lu_solve`.

Mirrors benchmark/test_linalg_lu_solve.py (LinalgLuSolveBenchmark, a
base.Benchmark subclass):
    op_name  = "linalg_lu_solve"
    torch_op = torch.linalg.lu_solve
    dtypes   = [torch.float32]

The class overrides set_shapes with LU_SOLVE_SHAPES -- read as (n, k) -- so
core_shapes.yaml is not consulted.

LU and pivots come from a real torch.linalg.lu_factor call on a
well-conditioned A = randn @ randn^T + n * I, which build_inputs performs
outside the timed region exactly as the benchmark's `_make_lu_inputs` does.

float64 is in the dtype list because the benchmark appends it when
flag_gems.runtime.device.support_fp64 is true, and the NVIDIA backend
reports fp64_enabled=True.
"""

import torch

OP_NAME = "linalg_lu_solve"
DTYPES = [torch.float32, torch.float64]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = "torch.linalg.lu_solve is unavailable on this torch build"

# test_linalg_lu_solve.py::LU_SOLVE_SHAPES -- (n, k)
_SHAPES = [
    (4, 4),
    (8, 8),
    (16, 16),
    (32, 32),
    (33, 16),
    (48, 32),
    (64, 1),
    (64, 64),
    (128, 128),
    (256, 64),
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
    op = _resolve("torch.linalg.lu_solve")
    return op if callable(op) else None


def grid():
    return [{"n": n, "k": k} for n, k in _SHAPES]


def build_inputs(binding, dtype, device):
    n, k = binding["n"], binding["k"]
    # benchmark::_make_lu_inputs -- A made well-conditioned with + n * I.
    a = torch.randn(n, n, dtype=dtype, device=device)
    a = a @ a.mT + torch.eye(n, dtype=dtype, device=device) * n
    lu, pivots = torch.linalg.lu_factor(a)
    b = torch.randn(n, k, dtype=dtype, device=device)
    return (lu, pivots, b), {}


def key_shape(binding):
    return [binding["n"], binding["k"]]


def config(binding, dtype):
    n, k = binding["n"], binding["k"]
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": [n, n], "dtype": dt, "note": "LU from lu_factor"},
                {"shape": [n], "dtype": "torch.int32"},
                {"shape": [n, k], "dtype": dt},
            ],
            "kwargs": {},
        },
        "outputs": {"out": {"shape": [n, k], "dtype": dt}},
        "dims": {"N": n, "K": k},
    }
