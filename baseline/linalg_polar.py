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

"""NVIDIA native baseline for `linalg_polar`.

Mirrors benchmark/test_linalg_polar.py (LinalgPolarBenchmark, a base.Benchmark
subclass):
    op_name  = "linalg_polar"
    torch_op = TORCH_LINALG_POLAR = torch.ops.aten.linalg_polar.default
    dtypes   = [torch.float32]

LinalgPolarBenchmark.set_shapes uses its own CORE_SHAPES and only adds
COMPREHENSIVE_SHAPES at --level comprehensive, so core_shapes.yaml is not
consulted and (64, 8, 4) / (32, 16, 8) are not part of this grid.

The benchmark picks its callable at import time: torch.ops.aten.linalg_polar
when the build carries it, and a torch-composed reference otherwise. This
baseline resolves the ATen operator and reports it unavailable rather than
substituting the composite, so a missing operator is visible instead of being
silently measured as something else. See Q&A.md.
"""

import torch

OP_NAME = "linalg_polar"
DTYPES = [torch.float32]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten.linalg_polar.default is unavailable on this torch build"
)

# LinalgPolarBenchmark.CORE_SHAPES
_SHAPES = [
    (4, 2),
    (8, 4),
    (16, 8),
    (16, 8, 4),
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
    op = _resolve("torch.ops.aten.linalg_polar.default")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    inp = torch.randn(tuple(binding["shape"]), dtype=dtype, device=device)
    return (inp,), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    dt = str(dtype)
    h_shape = shape[:-2] + [shape[-1], shape[-1]]
    return {
        "inputs": {
            "args": [{"shape": shape, "dtype": dt}],
            "kwargs": {},
        },
        "outputs": {
            "U": {"shape": shape, "dtype": dt},
            "H": {"shape": h_shape, "dtype": dt},
        },
        "dims": {"shape": shape},
    }
