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

"""NVIDIA native baseline for `linalg_tensorinv`.

Mirrors benchmark/test_linalg_tensorinv.py (LinalgTensorinvBenchmark, a
base.Benchmark subclass):
    op_name  = "linalg_tensorinv"
    torch_op = torch.linalg.tensorinv
    dtypes   = [torch.float32]

The class overrides set_shapes with TENSORINV_SHAPES, so core_shapes.yaml is
not consulted. The input is made invertible the way the benchmark makes it:
A @ A^T + 0.1 * I. ind=1 is passed positionally (matrix inverse on a 2-D
input).

float32 only: PyTorch has no half support here.
"""

import torch

OP_NAME = "linalg_tensorinv"
DTYPES = [torch.float32]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = "torch.linalg.tensorinv is unavailable on this torch build"

# test_linalg_tensorinv.py::TENSORINV_SHAPES
_SHAPES = [
    (2, 2),
    (4, 4),
    (8, 8),
    (16, 16),
    (24, 24),
]

_IND = 1


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
    op = _resolve("torch.linalg.tensorinv")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    a = torch.randn(shape, dtype=dtype, device=device)
    m = shape[0]
    a = a @ a.T + torch.eye(m, dtype=dtype, device=device) * 0.1
    return (a, _IND), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": shape, "dtype": dt, "note": "invertible"},
                {"scalar": _IND},
            ],
            "kwargs": {},
        },
        "outputs": {"out": {"shape": shape, "dtype": dt}},
        "dims": {"shape": shape, "ind": _IND},
    }
