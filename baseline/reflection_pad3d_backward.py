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

"""NVIDIA native baseline for `reflection_pad3d_backward`.

Mirrors benchmark/test_reflection_pad3d_backward.py
(ReflectionPad3dBackwardBenchmark, a base.Benchmark subclass):
    op_name  = "reflection_pad3d_backward"
    torch_op = torch.ops.aten.reflection_pad3d_backward
    dtypes   = consts.FLOAT_DTYPES

The class overrides set_shapes to cross REFLECTION_PAD3D_BACKWARD_SHAPES with
REFLECTION_PAD3D_PADDINGS, so core_shapes.yaml is not consulted and each shape
contributes three collection points.

grad_output is ones of the padded volume. The benchmark unpacks the padding
tuple as (pad_d0, pad_d1, pad_h0, pad_h1, pad_w0, pad_w1) when computing the
output dims; that order is reproduced here even though ATen's own convention
lists W first, because the padding values are what the benchmark passes and the
output shape it derives is what the collected record must describe.

This calls the ATen backward kernel directly; it is not an autograd benchmark.
"""

import torch

OP_NAME = "reflection_pad3d_backward"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten.reflection_pad3d_backward is unavailable on this build"
)

# test_reflection_pad3d_backward.py::REFLECTION_PAD3D_BACKWARD_SHAPES
_SHAPES = [
    (1, 1, 4, 4, 4),
    (2, 3, 8, 8, 8),
    (1, 1, 16, 16, 16),
    (2, 4, 8, 16, 32),
]

# REFLECTION_PAD3D_PADDINGS -- each value is strictly less than its dim size.
_PADDINGS = [
    (1, 1, 1, 1, 1, 1),
    (2, 2, 2, 2, 2, 2),
    (1, 2, 1, 2, 1, 2),
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
    op = _resolve("torch.ops.aten.reflection_pad3d_backward")
    return op if callable(op) else None


def grid():
    return [
        {"shape": list(shape), "padding": list(padding)}
        for shape in _SHAPES
        for padding in _PADDINGS
    ]


def _out_shape(shape, padding):
    n, c, d, h, w = shape
    pad_d0, pad_d1, pad_h0, pad_h1, pad_w0, pad_w1 = padding
    return [n, c, d + pad_d0 + pad_d1, h + pad_h0 + pad_h1, w + pad_w0 + pad_w1]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    padding = tuple(binding["padding"])
    out_shape = _out_shape(list(shape), list(padding))
    x = torch.randn(shape, dtype=dtype, device=device)
    grad_output = torch.ones(tuple(out_shape), dtype=dtype, device=device)
    return (grad_output, x, padding), {}


def key_shape(binding):
    shape = "x".join(str(d) for d in binding["shape"])
    pad = "".join(str(p) for p in binding["padding"])
    return f"{shape}_pad{pad}"


def config(binding, dtype):
    shape = list(binding["shape"])
    padding = list(binding["padding"])
    out_shape = _out_shape(shape, padding)
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": out_shape, "dtype": dt, "note": "grad_output"},
                {"shape": shape, "dtype": dt},
                padding,
            ],
            "kwargs": {},
        },
        "outputs": {"grad_input": {"shape": shape, "dtype": dt}},
        "dims": {"shape": shape, "padding": padding},
    }
