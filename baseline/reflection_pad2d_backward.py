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

"""NVIDIA native baseline for `reflection_pad2d_backward`.

Mirrors benchmark/test_reflection_pad2d_backward.py
(ReflectionPad2dBackwardBenchmark, a base.Benchmark subclass):
    op_name  = "reflection_pad2d_backward"
    torch_op = torch.ops.aten.reflection_pad2d_backward
    dtypes   = consts.FLOAT_DTYPES

The class overrides set_shapes with its own ResNet-like 4-D list, so
core_shapes.yaml is not consulted; set_more_shapes only applies at --level
comprehensive and is therefore not part of this grid.

padding is [2, 2, 2, 2] and grad_output is randn_like of the reflect-padded
input, which build_inputs derives from a real F.pad call outside the timed
region, as the benchmark does.

This calls the ATen backward kernel directly; it is not an autograd benchmark.
"""

import torch

OP_NAME = "reflection_pad2d_backward"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten.reflection_pad2d_backward is unavailable on this build"
)

# ReflectionPad2dBackwardBenchmark.set_shapes -- ResNet-like NCHW
_SHAPES = [
    (4, 3, 224, 224),
    (16, 64, 56, 56),
    (32, 128, 28, 28),
    (64, 256, 14, 14),
    (128, 512, 7, 7),
]

_PADDING = [2, 2, 2, 2]


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
    op = _resolve("torch.ops.aten.reflection_pad2d_backward")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    inp = torch.randn(shape, dtype=dtype, device=device)
    padded = torch.nn.functional.pad(inp, _PADDING, mode="reflect")
    grad_output = torch.randn_like(padded)
    return (grad_output, inp, list(_PADDING)), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    n, c, h, w = shape
    # F.pad takes [left, right, top, bottom].
    padded = [n, c, h + _PADDING[2] + _PADDING[3], w + _PADDING[0] + _PADDING[1]]
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": padded, "dtype": dt, "note": "grad_output"},
                {"shape": shape, "dtype": dt},
                list(_PADDING),
            ],
            "kwargs": {},
        },
        "outputs": {"grad_input": {"shape": shape, "dtype": dt}},
        "dims": {
            "N": n,
            "C": c,
            "H": h,
            "W": w,
            "padding": list(_PADDING),
        },
    }
