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

"""NVIDIA native baseline for `upsample_nearest_exact1d_backward_grad_input`.

Mirrors benchmark/test_upsample_nearest_exact1d_backward.py
(UpsampleNearestExact1dBackwardOutBenchmark, an
UpsampleNearestExact1dBackwardBenchmark subclass):
    op_name  = "upsample_nearest_exact1d_backward_grad_input"
    torch_op = torch.ops.aten._upsample_nearest_exact1d_backward.grad_input
    dtypes   = consts.FLOAT_DTYPES

The parent class overrides set_shapes with its own (N, C, IW, OW) list and
returns [] from set_more_shapes, so core_shapes.yaml is not consulted. The
list spans launch-bound through throughput-bound workloads and both resize
directions -- note (8, 16, 512, 256) is a downsample.

grad_output is randn of [N, C, OW]; the .grad_input overload takes its output
buffer as a keyword.

This calls the ATen backward kernel directly; it is not an autograd benchmark.
"""

import torch

OP_NAME = "upsample_nearest_exact1d_backward_grad_input"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
MUTATED_INPUTS = ("grad_input",)
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten._upsample_nearest_exact1d_backward.grad_input is unavailable"
)

# UpsampleNearestExact1dBackwardBenchmark.set_shapes -- (N, C, IW, OW)
_SHAPES = [
    (1, 1, 8, 16),
    (2, 3, 64, 96),
    (8, 16, 512, 256),
    (16, 64, 2048, 4096),
    (8, 128, 8192, 12288),
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
    op = _resolve("torch.ops.aten._upsample_nearest_exact1d_backward.grad_input")
    return op if callable(op) else None


def grid():
    return [
        {"batch": n, "channels": c, "input_w": iw, "output_w": ow}
        for n, c, iw, ow in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    batch = binding["batch"]
    channels = binding["channels"]
    input_w = binding["input_w"]
    output_w = binding["output_w"]
    grad_output = torch.randn((batch, channels, output_w), dtype=dtype, device=device)
    input_size = (batch, channels, input_w)
    grad_input = torch.empty(input_size, dtype=dtype, device=device)
    return (grad_output, (output_w,), input_size), {"grad_input": grad_input}


def key_shape(binding):
    return [
        binding["batch"],
        binding["channels"],
        binding["input_w"],
        binding["output_w"],
    ]


def config(binding, dtype):
    batch = binding["batch"]
    channels = binding["channels"]
    input_w = binding["input_w"]
    output_w = binding["output_w"]
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {
                    "shape": [batch, channels, output_w],
                    "dtype": dt,
                    "note": "grad_output",
                },
                [output_w],
                [batch, channels, input_w],
            ],
            "kwargs": {
                "grad_input": {
                    "shape": [batch, channels, input_w],
                    "dtype": dt,
                }
            },
        },
        "outputs": {
            "grad_input": {
                "shape": [batch, channels, input_w],
                "dtype": dt,
            }
        },
        "dims": {
            "N": batch,
            "C": channels,
            "IW": input_w,
            "OW": output_w,
        },
    }
