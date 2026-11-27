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

"""NVIDIA native baseline for `ldexp`.

Mirrors benchmark/test_ldexp.py (LdexpBenchmark, a GenericBenchmark subclass):
    op_name  = "ldexp"
    torch_op = torch.ops.aten.ldexp.Tensor
    dtypes   = consts.FLOAT_DTYPES

The class overrides set_shapes with its own list and only extends it at
--level comprehensive, so core_shapes.yaml is not consulted and the
(2**24,) / (8192, 4096) shapes are not part of this grid.

The exponent operand is an int32 tensor drawn from randint(-8, 9), matching
ldexp_input_fn.
"""

import torch

OP_NAME = "ldexp"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten.ldexp.Tensor is unavailable on this torch build"
)

# LdexpBenchmark.set_shapes (core level)
_SHAPES = [
    (1,),
    (1024,),
    (1024, 1024),
    (4096, 4096),
    (16, 128, 64, 64),
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
    op = _resolve("torch.ops.aten.ldexp.Tensor")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    self_t = torch.randn(shape, dtype=dtype, device=device)
    other = torch.randint(-8, 9, shape, dtype=torch.int32, device=device)
    return (self_t, other), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": shape, "dtype": dt},
                {"shape": shape, "dtype": "torch.int32"},
            ],
            "kwargs": {},
        },
        "outputs": {"out": {"shape": shape, "dtype": dt}},
        "dims": {"shape": shape},
    }
