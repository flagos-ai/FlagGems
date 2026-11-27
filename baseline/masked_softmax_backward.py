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

"""NVIDIA native baseline for `masked_softmax_backward`.

Mirrors benchmark/test_masked_softmax_backward.py
(MaskedSoftmaxBackwardBenchmark, a base.Benchmark subclass):
    op_name  = "masked_softmax_backward"
    torch_op = torch.ops.aten._masked_softmax_backward
    dtypes   = consts.FLOAT_DTYPES

The class overrides set_shapes with its own four shapes and returns None from
set_more_shapes, so core_shapes.yaml is not consulted. The comment explains
why: the softmax reduction row must stay bounded, and the generic 1-D
1G-element default would exceed the per-block tensor limit.

The mask is `rand(shape) < 0.3`, and the forward output comes from a real
torch.ops.aten._masked_softmax(inp, mask, -1, 2) call with the masked
positions zeroed -- which build_inputs performs outside the timed region, as
the benchmark's get_input_iter does. dim is -1.

This calls the ATen backward kernel directly; it is not an autograd benchmark.
"""

import torch

OP_NAME = "masked_softmax_backward"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten._masked_softmax_backward is unavailable on this build"
)

# MaskedSoftmaxBackwardBenchmark.set_shapes
_SHAPES = [
    (64, 64),
    (4096, 4096),
    (64, 512, 512),
    (8, 32, 2048),
]

_DIM = -1
_MASK_TYPE = 2
_MASK_P = 0.3


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
    op = _resolve("torch.ops.aten._masked_softmax_backward")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    inp = torch.randn(shape, dtype=dtype, device=device)
    mask = torch.rand(shape, device=device) < _MASK_P
    output = torch.ops.aten._masked_softmax(inp, mask, _DIM, _MASK_TYPE)
    output = torch.where(mask, torch.zeros_like(output), output)
    grad_output = torch.randn(shape, dtype=dtype, device=device)
    return (grad_output, output, mask, _DIM), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": shape, "dtype": dt, "note": "grad_output"},
                {"shape": shape, "dtype": dt, "note": "forward output"},
                {"shape": shape, "dtype": "torch.bool"},
                {"scalar": _DIM},
            ],
            "kwargs": {},
        },
        "outputs": {"grad_input": {"shape": shape, "dtype": dt}},
        "dims": {"shape": shape, "dim": _DIM},
    }
