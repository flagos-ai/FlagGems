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

"""NVIDIA native baseline for `linalg_cholesky_ex`.

Mirrors benchmark/test_linalg_cholesky_ex.py (LinalgCholeskyExBenchmark, a
base.Benchmark subclass):
    op_name  = "linalg_cholesky_ex"
    torch_op = torch.ops.aten.linalg_cholesky_ex.default
    dtypes   = [float32, float64] when the backend has FP64

The class overrides set_shapes with the module-level CASES list and declares
shape_desc "operator configuration", so core_shapes.yaml is not consulted.
Each case is (batch_shape, n, upper).

float64 is included because the NVIDIA backend reports fp64_enabled=True.

The input is made symmetric positive definite the way the benchmark makes it:
raw @ raw^T + 2.0 * I. upper and check_errors=False are passed as keywords.
"""

import torch

OP_NAME = "linalg_cholesky_ex"
DTYPES = [torch.float32, torch.float64]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten.linalg_cholesky_ex.default is unavailable on this build"
)

# test_linalg_cholesky_ex.py::CASES -- (batch_shape, n, upper)
_CASES = [
    ((), 2, False),
    ((1,), 3, True),
    ((2,), 4, False),
    ((2, 3), 5, True),
    ((1,), 7, False),
    ((3,), 3, True),
    ((2, 2), 4, False),
    ((4,), 6, True),
]

_CHECK_ERRORS = False


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
    op = _resolve("torch.ops.aten.linalg_cholesky_ex.default")
    return op if callable(op) else None


def grid():
    return [
        {"batch_shape": list(batch_shape), "n": n, "upper": upper}
        for batch_shape, n, upper in _CASES
    ]


def build_inputs(binding, dtype, device):
    batch_shape = tuple(binding["batch_shape"])
    n = binding["n"]
    raw = torch.randn(*batch_shape, n, n, dtype=dtype, device=device)
    eye = torch.eye(n, dtype=dtype, device=device).expand(*batch_shape, n, n)
    a = raw @ raw.transpose(-1, -2) + 2.0 * eye
    return (a,), {"upper": binding["upper"], "check_errors": _CHECK_ERRORS}


def key_shape(binding):
    batch = "x".join(str(d) for d in binding["batch_shape"]) or "scalar"
    return f"batch{batch}_n{binding['n']}_upper{int(binding['upper'])}"


def config(binding, dtype):
    batch_shape = list(binding["batch_shape"])
    n = binding["n"]
    dt = str(dtype)
    a_shape = batch_shape + [n, n]
    return {
        "inputs": {
            "args": [{"shape": a_shape, "dtype": dt, "note": "SPD"}],
            "kwargs": {
                "upper": {"scalar": binding["upper"]},
                "check_errors": {"scalar": _CHECK_ERRORS},
            },
        },
        "outputs": {
            "L": {"shape": a_shape, "dtype": dt},
            "info": {"shape": batch_shape, "dtype": "torch.int32"},
        },
        "dims": {
            "batch_shape": batch_shape,
            "n": n,
            "upper": binding["upper"],
        },
    }
