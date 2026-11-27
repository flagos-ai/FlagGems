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

"""NVIDIA native baseline for `cdist_backward`.

Mirrors benchmark/test_cdist_backward.py (CdistBackwardBenchmark, a
base.Benchmark subclass):
    op_name  = "cdist_backward"
    torch_op = torch.ops.aten._cdist_backward
    dtypes   = [torch.float32]
    shapes   = yaml:cdist_backward (benchmark/core_shapes.yaml),
               shape_desc "batch, n1, dim"

n2 is derived as n1 // 2 + 1. The forward cdist (p=2) is computed in
build_inputs because the operator takes its result as an argument; on NVIDIA
the benchmark uses torch.cdist for it (the flag_gems.cdist branch is
kunlunxin-only, where the native p=2 kernel is missing).

float32 only: _cdist_backward accumulates in fp32 and lower precisions are not
numerically stable.
"""

import torch

OP_NAME = "cdist_backward"
DTYPES = [torch.float32]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten._cdist_backward is unavailable on this torch build"
)

# yaml:cdist_backward -- shape_desc "batch, n1, dim"
_SHAPES = [
    (2, 16, 32),
    (4, 32, 64),
    (8, 64, 128),
]

_P = 2.0


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
    op = _resolve("torch.ops.aten._cdist_backward")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    batch, n1, dim = binding["shape"]
    n2 = n1 // 2 + 1
    x1 = torch.randn((batch, n1, dim), dtype=dtype, device=device)
    x2 = torch.randn(batch, n2, dim, dtype=dtype, device=device)
    cdist = torch.cdist(x1, x2, p=_P)
    grad = torch.randn(batch, n1, n2, dtype=dtype, device=device)
    return (grad, x1, x2, _P, cdist), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    batch, n1, dim = binding["shape"]
    n2 = n1 // 2 + 1
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": [batch, n1, n2], "dtype": dt},
                {"shape": [batch, n1, dim], "dtype": dt},
                {"shape": [batch, n2, dim], "dtype": dt},
                {"scalar": _P},
                {"shape": [batch, n1, n2], "dtype": dt},
            ],
            "kwargs": {},
        },
        "outputs": {"grad_x1": {"shape": [batch, n1, dim], "dtype": dt}},
        "dims": {"batch": batch, "n1": n1, "n2": n2, "dim": dim, "p": _P},
    }
