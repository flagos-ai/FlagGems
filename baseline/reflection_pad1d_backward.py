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

"""NVIDIA native baseline for `reflection_pad1d_backward`.

Mirrors benchmark/test_reflection_pad1d_backward.py
(ReflectionPad1dBackwardBenchmark, a base.Benchmark subclass):
    op_name  = "reflection_pad1d_backward"
    torch_op = torch.ops.aten.reflection_pad1d_backward
    dtypes   = consts.FLOAT_DTYPES

The class overrides set_shapes with REFLECTION_PAD1D_BACKWARD_SHAPES -- read as
(B, W) -- so core_shapes.yaml is not consulted. padding is (1, 2) and the
gradient is ones of the padded width, which build_inputs derives from a real
forward reflection_pad1d call outside the timed region, as the benchmark does.

This calls the ATen backward kernel directly; it is not an autograd benchmark.
"""

import torch

OP_NAME = "reflection_pad1d_backward"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten.reflection_pad1d_backward is unavailable on this build"
)

# test_reflection_pad1d_backward.py::REFLECTION_PAD1D_BACKWARD_SHAPES -- (B, W)
_SHAPES = [
    (2, 3),
    (4, 8),
    (8, 16),
    (1, 32),
    (4, 64),
    (8, 128),
]

_PADDING = (1, 2)


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
    op = _resolve("torch.ops.aten.reflection_pad1d_backward")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    b, w = binding["shape"]
    x = torch.randn(b, w, dtype=dtype, device=device)
    padded = torch.ops.aten.reflection_pad1d(x, _PADDING)
    w_out = padded.shape[-1]
    grad = torch.ones(b, w_out, dtype=dtype, device=device)
    return (grad, x, _PADDING), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    b, w = binding["shape"]
    w_out = w + _PADDING[0] + _PADDING[1]
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": [b, w_out], "dtype": dt, "note": "grad_output"},
                {"shape": [b, w], "dtype": dt},
                list(_PADDING),
            ],
            "kwargs": {},
        },
        "outputs": {"grad_input": {"shape": [b, w], "dtype": dt}},
        "dims": {"B": b, "W": w, "W_out": w_out, "padding": list(_PADDING)},
    }
