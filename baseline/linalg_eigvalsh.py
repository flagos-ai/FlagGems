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

"""NVIDIA native baseline for `linalg_eigvalsh`.

Mirrors benchmark/test_linalg_eigvalsh.py (LinalgEigvalshBenchmark, a
base.Benchmark subclass):
    op_name  = "linalg_eigvalsh"
    torch_op = torch.ops.aten.linalg_eigvalsh.default
    dtypes   = [float32, float64] when the backend has FP64

The class overrides set_shapes with the module-level CASES list and declares
shape_desc "operator configuration", so core_shapes.yaml is not consulted.
get_input_iter branches on the operator name: the plain overload gets UPLO only.

The input is made symmetric the way the benchmark makes it:
raw + raw^T. UPLO is passed as a keyword.

float64 is included because the NVIDIA backend reports fp64_enabled=True.
"""

import torch

OP_NAME = "linalg_eigvalsh"
DTYPES = [torch.float32, torch.float64]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten.linalg_eigvalsh.default is unavailable on this torch build"
)

# test_linalg_eigvalsh.py::CASES -- (batch_shape, n, UPLO)
_CASES = [
    ((), 2, "L"),
    ((1,), 3, "U"),
    ((2,), 4, "L"),
    ((2, 3), 5, "U"),
    ((1,), 7, "L"),
    ((3,), 3, "U"),
    ((2, 2), 4, "L"),
    ((4,), 6, "U"),
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
    op = _resolve("torch.ops.aten.linalg_eigvalsh.default")
    return op if callable(op) else None


def grid():
    return [
        {"batch_shape": list(batch_shape), "n": n, "uplo": uplo}
        for batch_shape, n, uplo in _CASES
    ]


def build_inputs(binding, dtype, device):
    batch_shape = tuple(binding["batch_shape"])
    n = binding["n"]
    raw = torch.randn(*batch_shape, n, n, dtype=dtype, device=device)
    a = raw + raw.transpose(-1, -2)
    return (a,), {"UPLO": binding["uplo"]}


def key_shape(binding):
    batch = "x".join(str(d) for d in binding["batch_shape"]) or "scalar"
    return f"batch{batch}_n{binding['n']}_uplo{binding['uplo']}"


def config(binding, dtype):
    batch_shape = list(binding["batch_shape"])
    n = binding["n"]
    dt = str(dtype)
    a_shape = batch_shape + [n, n]
    eig_shape = batch_shape + [n]
    return {
        "inputs": {
            "args": [{"shape": a_shape, "dtype": dt, "note": "symmetric"}],
            "kwargs": {
                "UPLO": binding["uplo"],
            },
        },
        "outputs": {"eigenvalues": {"shape": eig_shape, "dtype": dt}},
        "dims": {
            "batch_shape": batch_shape,
            "n": n,
            "UPLO": binding["uplo"],
        },
    }
