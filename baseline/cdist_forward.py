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

"""NVIDIA native baseline for `cdist_forward`.

Mirrors benchmark/test_cdist_forward.py (CdistForwardBenchmark, a
base.Benchmark subclass):
    op_name  = "cdist_forward"
    torch_op = torch.cdist
    dtypes   = [torch.float32]

The class overrides set_shapes to assign CDIST_FORWARD_SHAPES, so
core_shapes.yaml is not consulted. Each entry is a (shape1, shape2) pair and
p is fixed at 2.0.

float32 only: torch.cdist has no float16 CUDA support and lower precisions are
not numerically stable here.
"""

import torch

OP_NAME = "cdist_forward"
DTYPES = [torch.float32]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = "torch.cdist is unavailable on this torch build"

# test_cdist_forward.py::CDIST_FORWARD_SHAPES -- (shape1, shape2)
_SHAPES = [
    ((4, 8), (6, 8)),
    ((8, 16), (8, 16)),
    ((16, 32), (16, 32)),
    ((32, 64), (32, 64)),
    ((64, 128), (64, 128)),
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
    op = _resolve("torch.cdist")
    return op if callable(op) else None


def grid():
    return [{"shape1": list(s1), "shape2": list(s2)} for s1, s2 in _SHAPES]


def build_inputs(binding, dtype, device):
    x1 = torch.randn(*binding["shape1"], dtype=dtype, device=device)
    x2 = torch.randn(*binding["shape2"], dtype=dtype, device=device)
    return (x1, x2, _P), {}


def key_shape(binding):
    s1 = "x".join(str(d) for d in binding["shape1"])
    s2 = "x".join(str(d) for d in binding["shape2"])
    return f"x1{s1}_x2{s2}"


def config(binding, dtype):
    s1 = list(binding["shape1"])
    s2 = list(binding["shape2"])
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": s1, "dtype": dt},
                {"shape": s2, "dtype": dt},
                {"scalar": _P},
            ],
            "kwargs": {},
        },
        "outputs": {"out": {"shape": [s1[-2], s2[-2]], "dtype": dt}},
        "dims": {"P": s1[-2], "R": s2[-2], "M": s1[-1], "p": _P},
    }
