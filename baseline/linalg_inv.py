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

"""NVIDIA native baseline for `linalg_inv`.

Mirrors benchmark/test_linalg_inv.py (LinalgInvBenchmark, a base.Benchmark
subclass):
    op_name  = "linalg_inv"
    torch_op = torch.ops.aten.linalg_inv.default
    dtypes   = [float32, float64] when the backend has FP64

The class overrides set_shapes with the module-level CASES list, so
core_shapes.yaml is not consulted. Each case is (batch_shape, n).

float64 is included because the NVIDIA backend reports fp64_enabled=True.
"""

import torch

OP_NAME = "linalg_inv"
DTYPES = [torch.float32, torch.float64]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten.linalg_inv.default is unavailable on this torch build"
)

# test_linalg_inv.py::CASES -- (batch_shape, n)
_CASES = [
    ((), 2),
    ((1,), 3),
    ((2,), 4),
    ((2, 3), 5),
    ((1,), 7),
    ((3,), 3),
    ((2, 2), 4),
    ((4,), 6),
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
    op = _resolve("torch.ops.aten.linalg_inv.default")
    return op if callable(op) else None


def grid():
    return [{"batch_shape": list(batch_shape), "n": n} for batch_shape, n in _CASES]


def build_inputs(binding, dtype, device):
    batch_shape = tuple(binding["batch_shape"])
    n = binding["n"]
    raw = torch.randn(*batch_shape, n, n, dtype=dtype, device=device)
    eye = torch.eye(n, dtype=dtype, device=device).expand(*batch_shape, n, n)
    # Symmetric positive definite, exactly as the benchmark builds it.
    return (raw @ raw.transpose(-1, -2) + 3.0 * eye,), {}


def key_shape(binding):
    batch = "x".join(str(d) for d in binding["batch_shape"]) or "scalar"
    return f"batch{batch}_n{binding['n']}"


def config(binding, dtype):
    batch_shape = list(binding["batch_shape"])
    n = binding["n"]
    dt = str(dtype)
    a_shape = batch_shape + [n, n]
    return {
        "inputs": {
            "args": [{"shape": a_shape, "dtype": dt, "note": "SPD"}],
            "kwargs": {},
        },
        "outputs": {"out": {"shape": a_shape, "dtype": dt}},
        "dims": {"batch_shape": batch_shape, "n": n},
    }
