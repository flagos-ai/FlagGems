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

"""NVIDIA native baseline for `logcumsumexp_out`.

Mirrors benchmark/test_logcumsumexp.py (base.GenericBenchmark2DOnly):
    op_name  = "logcumsumexp_out"
    torch_op = torch.ops.aten.logcumsumexp.out
    dtypes   = consts.FLOAT_DTYPES
    shapes   = yaml:GenericBenchmark2DOnly (benchmark/core_shapes.yaml)

The input_fn yields (inp, 1, out=empty_like(inp)).
"""

import torch

OP_NAME = "logcumsumexp_out"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
MUTATED_INPUTS = ("out",)
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten.logcumsumexp.out is unavailable on this torch build"
)

# yaml:GenericBenchmark2DOnly
_SHAPES = [
    (64, 64),
    (256, 256),
    (1024, 1024),
    (4096, 4096),
    (1024, 65536),
]

_DIM = 1


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
    op = _resolve("torch.ops.aten.logcumsumexp.out")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    inp = torch.randn(shape, dtype=dtype, device=device)
    out = torch.empty_like(inp)
    return (inp, _DIM), {"out": out}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": shape, "dtype": dt},
                {"scalar": _DIM},
            ],
            "kwargs": {"out": {"shape": shape, "dtype": dt}},
        },
        "outputs": {"out": {"shape": shape, "dtype": dt}},
        "dims": {"shape": shape, "dim": _DIM},
    }
