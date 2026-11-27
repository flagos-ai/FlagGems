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

"""NVIDIA native baseline for `conv_depthwise2d`.

Mirrors benchmark/test_conv_depthwise2d.py (ConvDepthwise2DBenchmark, a
GenericBenchmark subclass):
    op_name  = "conv_depthwise2d"
    torch_op = torch.ops.aten._conv_depthwise2d
    dtypes   = consts.FLOAT_DTYPES
    shapes   = yaml:ConvDepthwise2DBenchmark (benchmark/core_shapes.yaml),
               shape_desc "N, C, H, W, K_h, K_w, stride, padding, dilation"

The `_conv_depthwise2d_torch_op` conv2d composition in that file is
kunlunxin-only; on NVIDIA the benchmark calls the real ATen operator, which is
what this baseline resolves.

Arguments are positional because the operator's first parameter is named
`self`: (input, weight, kernel_size, bias=None, stride, padding, dilation).
The weight is depthwise-shaped [C, 1, K_h, K_w].

The benchmark sets torch.backends.cudnn.allow_tf32 = False before running; the
same flag is set here.
"""

import torch

OP_NAME = "conv_depthwise2d"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten._conv_depthwise2d is unavailable on this torch build"
)

# yaml:ConvDepthwise2DBenchmark
# (N, C, H, W, K_h, K_w, stride, padding, dilation)
_SHAPES = [
    (32, 32, 64, 64, 3, 3, 1, 1, 1),
    (16, 64, 128, 128, 3, 3, 2, 1, 1),
    (8, 128, 56, 56, 5, 5, 1, 2, 1),
    (4, 256, 28, 28, 3, 3, 1, 1, 2),
    (2, 512, 14, 14, 3, 3, 1, 1, 1),
]

_FIELDS = (
    "batch",
    "channels",
    "input_h",
    "input_w",
    "kernel_h",
    "kernel_w",
    "stride",
    "padding",
    "dilation",
)


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
    op = _resolve("torch.ops.aten._conv_depthwise2d")
    if not callable(op):
        return None
    # The benchmark pins this before running.
    torch.backends.cudnn.allow_tf32 = False
    return op


def grid():
    return [dict(zip(_FIELDS, shape)) for shape in _SHAPES]


def build_inputs(binding, dtype, device):
    input_shape = (
        binding["batch"],
        binding["channels"],
        binding["input_h"],
        binding["input_w"],
    )
    weight_shape = (binding["channels"], 1, binding["kernel_h"], binding["kernel_w"])
    input_tensor = torch.randn(size=input_shape, device=device, dtype=dtype)
    weight = torch.randn(size=weight_shape, device=device, dtype=dtype)
    stride, padding, dilation = (
        binding["stride"],
        binding["padding"],
        binding["dilation"],
    )
    # Positional: the aten op names its first parameter `self`.
    return (
        input_tensor,
        weight,
        [binding["kernel_h"], binding["kernel_w"]],
        None,  # bias
        [stride, stride],
        [padding, padding],
        [dilation, dilation],
    ), {}


def key_shape(binding):
    return [binding[key] for key in _FIELDS]


def config(binding, dtype):
    dt = str(dtype)
    input_shape = [
        binding["batch"],
        binding["channels"],
        binding["input_h"],
        binding["input_w"],
    ]
    weight_shape = [
        binding["channels"],
        1,
        binding["kernel_h"],
        binding["kernel_w"],
    ]
    stride, padding, dilation = (
        binding["stride"],
        binding["padding"],
        binding["dilation"],
    )
    return {
        "inputs": {
            "args": [
                {"shape": input_shape, "dtype": dt},
                {"shape": weight_shape, "dtype": dt},
                [binding["kernel_h"], binding["kernel_w"]],
                None,
                [stride, stride],
                [padding, padding],
                [dilation, dilation],
            ],
            "kwargs": {},
        },
        "outputs": {"out": {"dtype": dt}},
        "dims": {key: binding[key] for key in _FIELDS},
    }
