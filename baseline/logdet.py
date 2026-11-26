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

"""NVIDIA native baseline for `logdet`.

Mirrors benchmark/test_logdet.py (LogdetBenchmark, a base.Benchmark subclass):
    op_name  = "logdet"
    torch_op = torch.logdet
    dtypes   = [float32] + [float64] when the backend has FP64

The class overrides set_shapes with LOGDET_SHAPES and only extends it at
--level comprehensive, so core_shapes.yaml is not consulted and
LOGDET_MORE_SHAPES is not part of this grid.

The input is conditioned the way the benchmark conditions it: a randn matrix
scaled by 1/sqrt(n), then A @ A^T + I, which keeps the determinant positive so
logdet is finite.
"""

import math

import torch

OP_NAME = "logdet"
DTYPES = [torch.float32, torch.float64]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = "torch.logdet is unavailable on this torch build"

# test_logdet.py::LOGDET_SHAPES
_SHAPES = [
    (1, 1),
    (4, 4),
    (8, 8),
    (16, 16),
    (4096, 4, 4),
    (1024, 8, 8),
    (128, 16, 16),
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
    op = _resolve("torch.logdet")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    n = shape[-1]
    matrix = torch.randn(shape, dtype=dtype, device=device)
    matrix /= math.sqrt(n)
    inp = matrix @ matrix.transpose(-2, -1) + torch.eye(n, dtype=dtype, device=device)
    return (inp,), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    dt = str(dtype)
    return {
        "inputs": {
            "args": [{"shape": shape, "dtype": dt, "note": "SPD"}],
            "kwargs": {},
        },
        "outputs": {"out": {"shape": shape[:-2], "dtype": dt}},
        "dims": {"shape": shape, "N": shape[-1]},
    }
