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

"""NVIDIA native baseline for `upsample_nearest3d_backward_grad_input`.

Mirrors benchmark/test_upsample_nearest3d_backward.py
(UpsampleNearest3dBackwardGradInputBenchmark, an
UpsampleNearest3dBackwardBenchmark subclass):
    op_name  = "upsample_nearest3d_backward_grad_input"
    torch_op = torch.ops.aten.upsample_nearest3d_backward.grad_input
    dtypes   = consts.FLOAT_DTYPES

The parent class overrides set_shapes with three volumetric feature-map sizes
and returns None from set_more_shapes, so core_shapes.yaml is not consulted.

The output volume is twice the input in each spatial dimension. grad_output is
ones of that shape, derived from a real torch.ops.aten.upsample_nearest3d
forward which build_inputs runs outside the timed region, as the benchmark
does. The .grad_input overload requires its output buffer as a keyword.

This calls the ATen backward kernel directly; it is not an autograd benchmark.
"""

import torch

OP_NAME = "upsample_nearest3d_backward_grad_input"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
MUTATED_INPUTS = ("grad_input",)
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten.upsample_nearest3d_backward.grad_input is unavailable"
)

# UpsampleNearest3dBackwardBenchmark.set_shapes -- volumetric feature maps
_SHAPES = [
    (2, 3, 8, 8, 8),
    (4, 8, 16, 16, 16),
    (4, 16, 24, 24, 24),
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
    op = _resolve("torch.ops.aten.upsample_nearest3d_backward.grad_input")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    x = torch.randn(shape, dtype=dtype, device=device)
    out_d, out_h, out_w = shape[2] * 2, shape[3] * 2, shape[4] * 2
    output_size = (out_d, out_h, out_w)
    out = torch.ops.aten.upsample_nearest3d(x, [out_d, out_h, out_w], None, None, None)
    grad_output = torch.ones_like(out)
    input_size = tuple(x.shape)
    grad_input = torch.empty(input_size, dtype=dtype, device=device)
    return (grad_output, output_size, input_size), {"grad_input": grad_input}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    n, c = shape[0], shape[1]
    output_size = [shape[2] * 2, shape[3] * 2, shape[4] * 2]
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {
                    "shape": [n, c] + output_size,
                    "dtype": dt,
                    "note": "grad_output (ones)",
                },
                output_size,
                shape,
            ],
            "kwargs": {"grad_input": {"shape": shape, "dtype": dt}},
        },
        "outputs": {"grad_input": {"shape": shape, "dtype": dt}},
        "dims": {
            "N": n,
            "C": c,
            "D": shape[2],
            "H": shape[3],
            "W": shape[4],
            "output_size": output_size,
        },
    }
