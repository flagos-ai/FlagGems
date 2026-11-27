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

"""NVIDIA native baseline for `prelu_kernel_backward`.

Mirrors benchmark/test_prelu_kernel_backward.py (PReluKernelBackwardBenchmark,
a base.Benchmark subclass):
    op_name  = "prelu_kernel_backward"
    torch_op = torch.ops.aten._prelu_kernel_backward
    dtypes   = consts.FLOAT_DTYPES

The class overrides set_shapes with PRELU_KERNEL_BACKWARD_SHAPES, so
core_shapes.yaml is not consulted: one large 4-D shape, one 2-D and one 5-D.

The weight is the single-element tensor [0.25] -- the shared-slope PReLU case.

This calls the ATen backward kernel directly; it is not an autograd benchmark.
"""

import torch

OP_NAME = "prelu_kernel_backward"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten._prelu_kernel_backward is unavailable on this torch build"
)

# test_prelu_kernel_backward.py::PRELU_KERNEL_BACKWARD_SHAPES
_SHAPES = [
    (16, 128, 64, 1280),  # large 4-D
    (1024, 1024),  # 2-D
    (16, 7, 57, 32, 29),  # 5-D
]

_SLOPE = 0.25


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
    op = _resolve("torch.ops.aten._prelu_kernel_backward")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    grad_output = torch.randn(shape, dtype=dtype, device=device)
    x = torch.randn(shape, dtype=dtype, device=device)
    weight = torch.tensor([_SLOPE], dtype=dtype, device=device)
    return (grad_output, x, weight), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": shape, "dtype": dt, "note": "grad_output"},
                {"shape": shape, "dtype": dt},
                {"shape": [1], "dtype": dt, "note": "shared slope"},
            ],
            "kwargs": {},
        },
        "outputs": {
            "grad_input": {"shape": shape, "dtype": dt},
            "grad_weight": {"shape": [1], "dtype": dt},
        },
        "dims": {"shape": shape, "slope": _SLOPE},
    }
