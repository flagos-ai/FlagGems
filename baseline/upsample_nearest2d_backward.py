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

"""NVIDIA native baseline for `upsample_nearest2d_backward`.

Mirrors benchmark/test_upsample_nearest2d_backward.py
(UpsampleNearest2dBackwardBenchmark, a base.Benchmark subclass):
    op_name  = "upsample_nearest2d_backward"
    torch_op = torch.ops.aten.upsample_nearest2d_backward.default
    dtypes   = [float32, float64, float16, bfloat16, uint8]
               (parametrized; every skip in DTYPES is
               mthreads/iluvatar/ascend-only, so all five run on NVIDIA)

The class overrides set_shapes with the named CASES list -- each entry is
(case, input_size, output_size, scales, input_layout, output_layout) -- and
declares a matching shape_desc, so core_shapes.yaml is not consulted.

grad_output is built in the case's input layout: "channels_last" permutes to
NHWC, makes it contiguous and permutes back; "strided" allocates twice the
width and takes [..., ::2]. uint8 inputs come from randint(0, 256) rather than
randn. The output_layout field only matters for the .grad_input overload, which
this operator name is not, so it is carried in the grid for identification
only.

This calls the ATen backward kernel directly; it is not an autograd benchmark.
"""

import torch

OP_NAME = "upsample_nearest2d_backward"
DTYPES = [
    torch.float32,
    torch.float64,
    torch.float16,
    torch.bfloat16,
    torch.uint8,
]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten.upsample_nearest2d_backward.default is unavailable"
)

# test_upsample_nearest2d_backward.py::CASES
# (case, input_size, output_size, scales, input_layout, output_layout)
_CASES = [
    ("identity-small", (1, 3, 7, 9), (7, 9), (None, None), "contiguous", "contiguous"),
    (
        "identity-large",
        (8, 32, 128, 128),
        (128, 128),
        (None, None),
        "contiguous",
        "contiguous",
    ),
    ("identity-explicit", (2, 8, 31, 47), (31, 47), (0.7, 2.2), "strided", "strided"),
    ("double-small", (2, 3, 7, 9), (14, 18), (None, None), "contiguous", "contiguous"),
    (
        "double-medium",
        (8, 16, 64, 64),
        (128, 128),
        (None, None),
        "contiguous",
        "contiguous",
    ),
    (
        "double-large",
        (2, 32, 128, 128),
        (256, 256),
        (2.0, 2.0),
        "contiguous",
        "contiguous",
    ),
    (
        "double-channels-last",
        (4, 32, 63, 65),
        (126, 130),
        (None, None),
        "channels_last",
        "channels_last",
    ),
    (
        "fractional-small",
        (3, 5, 7, 11),
        (19, 23),
        (None, None),
        "contiguous",
        "contiguous",
    ),
    (
        "fractional-large",
        (4, 16, 127, 129),
        (193, 257),
        (None, None),
        "contiguous",
        "contiguous",
    ),
    (
        "downsample",
        (4, 16, 127, 129),
        (53, 71),
        (None, None),
        "contiguous",
        "contiguous",
    ),
    (
        "strong-downsample",
        (2, 8, 127, 255),
        (3, 5),
        (None, None),
        "contiguous",
        "contiguous",
    ),
    ("output-one", (4, 8, 31, 47), (1, 1), (None, None), "contiguous", "contiguous"),
    ("input-one", (2, 3, 1, 1), (31, 47), (None, None), "contiguous", "contiguous"),
    ("strided-output", (2, 8, 31, 47), (67, 99), (None, None), "strided", "strided"),
    (
        "explicit-fractional",
        (2, 3, 31, 47),
        (67, 99),
        (2.2, 2.125),
        "contiguous",
        "contiguous",
    ),
    (
        "explicit-below-one",
        (2, 3, 7, 11),
        (19, 23),
        (0.7, 0.7),
        "contiguous",
        "contiguous",
    ),
    (
        "explicit-double",
        (2, 3, 7, 11),
        (14, 22),
        (0.7, 2.2),
        "contiguous",
        "contiguous",
    ),
    (
        "one-sided-scale",
        (2, 3, 7, 11),
        (19, 23),
        (None, 0.4),
        "contiguous",
        "contiguous",
    ),
    (
        "one-equal-axis",
        (2, 8, 31, 47),
        (31, 99),
        (0.7, 2.2),
        "channels_last",
        "strided",
    ),
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
    op = _resolve("torch.ops.aten.upsample_nearest2d_backward.default")
    return op if callable(op) else None


def grid():
    return [
        {
            "name": name,
            "shape": list(shape),
            "output_size": list(size),
            "scales": list(scales),
            "input_layout": layout,
            "output_layout": out_layout,
        }
        for name, shape, size, scales, layout, out_layout in _CASES
    ]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    size = tuple(binding["output_size"])
    layout = binding["input_layout"]
    output_shape = (*shape[:2], *size)
    if dtype == torch.uint8:
        grad_output = torch.randint(0, 256, output_shape, dtype=dtype, device=device)
    else:
        grad_output = torch.randn(output_shape, dtype=dtype, device=device)
    if layout == "channels_last":
        grad_output = grad_output.permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2)
    elif layout == "strided":
        buffer = torch.empty(
            (*output_shape[:-1], output_shape[-1] * 2), dtype=dtype, device=device
        )
        buffer[..., ::2].copy_(grad_output)
        grad_output = buffer[..., ::2]
    scales_h, scales_w = binding["scales"]
    return (grad_output, size, shape, scales_h, scales_w), {}


def key_shape(binding):
    return binding["name"]


def config(binding, dtype):
    shape = list(binding["shape"])
    size = list(binding["output_size"])
    n, c = shape[0], shape[1]
    dt = str(dtype)
    grad = {"shape": [n, c] + size, "dtype": dt}
    layout = binding["input_layout"]
    if layout == "channels_last":
        grad["stride"] = [c * size[0] * size[1], 1, c * size[1], c]
    elif layout == "strided":
        grad["stride"] = [
            c * size[0] * size[1] * 2,
            size[0] * size[1] * 2,
            size[1] * 2,
            2,
        ]
    scales_h, scales_w = binding["scales"]
    return {
        "inputs": {
            "args": [grad, size, shape, scales_h, scales_w],
            "kwargs": {},
        },
        "outputs": {"grad_input": {"shape": shape, "dtype": dt}},
        "dims": {
            "case": binding["name"],
            "input_size": shape,
            "output_size": size,
            "scales": list(binding["scales"]),
            "input_layout": layout,
            "output_layout": binding["output_layout"],
        },
    }
