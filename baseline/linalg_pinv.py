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

"""NVIDIA native baseline for `linalg_pinv`.

Mirrors benchmark/test_linalg_pinv.py (LinalgPinvBenchmark, a base.Benchmark
subclass):
    op_name  = "linalg_pinv"
    torch_op = torch.linalg.pinv
    dtypes   = [torch.float32]

The class overrides set_shapes with PINV_SHAPES -- read as (batch, m, n) -- so
core_shapes.yaml is not consulted. The input is a plain randn tensor.
"""

import torch

OP_NAME = "linalg_pinv"
DTYPES = [torch.float32]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = "torch.linalg.pinv is unavailable on this torch build"

# test_linalg_pinv.py::PINV_SHAPES -- (batch, m, n)
_SHAPES = [
    (1, 4, 4),
    (1, 8, 8),
    (4, 4, 4),
    (4, 8, 8),
    (8, 4, 4),
    (8, 8, 8),
    (16, 4, 4),
    (16, 8, 8),
    (32, 4, 4),
    (32, 8, 8),
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
    op = _resolve("torch.linalg.pinv")
    return op if callable(op) else None


def grid():
    return [{"batch": b, "m": m, "n": n} for b, m, n in _SHAPES]


def build_inputs(binding, dtype, device):
    a = torch.randn(
        binding["batch"], binding["m"], binding["n"], dtype=dtype, device=device
    )
    return (a,), {}


def key_shape(binding):
    return [binding["batch"], binding["m"], binding["n"]]


def config(binding, dtype):
    b, m, n = binding["batch"], binding["m"], binding["n"]
    dt = str(dtype)
    return {
        "inputs": {
            "args": [{"shape": [b, m, n], "dtype": dt}],
            "kwargs": {},
        },
        "outputs": {"out": {"shape": [b, n, m], "dtype": dt}},
        "dims": {"batch": b, "M": m, "N": n},
    }
