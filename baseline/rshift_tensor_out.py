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

"""NVIDIA native baseline for `rshift_tensor_out`.

Mirrors benchmark/test_rshift.py (RshiftOutBenchmark, a base.Benchmark
subclass):
    op_name  = "rshift_tensor_out"
    torch_op = torch.ops.aten.__rshift__.Tensor_out
    dtypes   = consts.INT_DTYPES + consts.EXTRA_INT_DTYPES

The class overrides set_shapes with its own three shapes, so core_shapes.yaml
is not consulted. The value tensor is randint(0, 100) and the shift amount
randint(0, 8), both in the parametrized integer dtype; `out` must be a keyword
for the .Tensor_out overload.

int64 is in the pool because the NVIDIA backend reports int64_enabled=True.
"""

import torch

OP_NAME = "rshift_tensor_out"
DTYPES = [torch.int16, torch.int32, torch.int8, torch.uint8, torch.int64]
IS_INPLACE = False
MUTATED_INPUTS = ("out",)
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten.__rshift__.Tensor_out is unavailable on this torch build"
)

# RshiftOutBenchmark.set_shapes -- shape_desc "SHAPE"
_SHAPES = [
    (1024,),
    (1024, 1024),
    (16, 512, 256),
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
    op = _resolve("torch.ops.aten.__rshift__.Tensor_out")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    value = torch.randint(0, 100, shape, dtype=dtype, device=device)
    shift = torch.randint(0, 8, shape, dtype=dtype, device=device)
    out = torch.empty_like(value)
    return (value, shift), {"out": out}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": shape, "dtype": dt},
                {"shape": shape, "dtype": dt, "note": "shift amount"},
            ],
            "kwargs": {"out": {"shape": shape, "dtype": dt}},
        },
        "outputs": {"out": {"shape": shape, "dtype": dt}},
        "dims": {"shape": shape},
    }
