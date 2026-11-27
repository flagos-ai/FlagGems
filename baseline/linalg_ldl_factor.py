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

"""NVIDIA native baseline for `linalg_ldl_factor`.

Mirrors benchmark/test_linalg_ldl_factor.py (LdlFactorBenchmark, a
base.Benchmark subclass):
    op_name  = "linalg_ldl_factor"
    torch_op = torch.linalg.ldl_factor
    dtypes   = [float32] + [float64] when the backend has FP64

The class overrides set_shapes with its own square-matrix list, so
core_shapes.yaml is not consulted.

The input is made symmetric positive definite the way the benchmark makes it:
A @ A^T + n * I.
"""

import torch

OP_NAME = "linalg_ldl_factor"
DTYPES = [torch.float32, torch.float64]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = "torch.linalg.ldl_factor is unavailable on this torch build"

# LdlFactorBenchmark.set_shapes -- square matrices only
_SHAPES = [
    (4, 4),
    (8, 8),
    (16, 16),
    (32, 32),
    (64, 64),
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
    op = _resolve("torch.linalg.ldl_factor")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    n = shape[0]
    a = torch.randn(shape, dtype=dtype, device=device)
    a = a @ a.transpose(-2, -1) + torch.eye(n, dtype=dtype, device=device) * n
    return (a,), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    n = shape[0]
    dt = str(dtype)
    return {
        "inputs": {
            "args": [{"shape": shape, "dtype": dt, "note": "SPD"}],
            "kwargs": {},
        },
        "outputs": {
            "LD": {"shape": shape, "dtype": dt},
            "pivots": {"shape": [n], "dtype": "torch.int32"},
        },
        "dims": {"N": n},
    }
