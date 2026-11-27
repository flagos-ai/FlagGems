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

"""NVIDIA native baseline for `pad_enum`.

Mirrors benchmark/test_pad_enum.py (base.GenericBenchmark):
    op_name  = "pad_enum"
    torch_op = torch.ops.aten._pad_enum
    dtypes   = consts.FLOAT_DTYPES
    shapes   = yaml:Benchmark (benchmark/core_shapes.yaml)

The pad widths are 3 on every side (2 * rank entries), mode is 3 and the
constant value is 1.5. Mode 3 is constant in the aten padding_mode enum
(0 reflect, 1 replicate, 2 circular, 3 constant); the benchmark picks it
because constant padding is valid for any rank and shape.
"""

import torch

OP_NAME = "pad_enum"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten._pad_enum is unavailable on this torch build"
)

# yaml:Benchmark -- the shared default shape list
_SHAPES = [
    (1073741824,),
    (64, 64),
    (4096, 4096),
    (64, 512, 512),
    (1024, 1024, 1024),
]

_PAD_WIDTH = 3
_MODE = 3  # constant
_VALUE = 1.5


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
    op = _resolve("torch.ops.aten._pad_enum")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    inp = torch.randn(shape, device=device, dtype=dtype)
    pad = [_PAD_WIDTH for _ in range(inp.ndim * 2)]
    return (inp, pad, _MODE, _VALUE), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    pad = [_PAD_WIDTH for _ in range(len(shape) * 2)]
    out_shape = [d + 2 * _PAD_WIDTH for d in shape]
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": shape, "dtype": dt},
                pad,
                {"scalar": _MODE},
                {"scalar": _VALUE},
            ],
            "kwargs": {},
        },
        "outputs": {"out": {"shape": out_shape, "dtype": dt}},
        "dims": {"shape": shape, "pad": pad, "mode": _MODE, "value": _VALUE},
    }
