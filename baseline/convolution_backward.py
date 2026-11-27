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

"""NVIDIA native baseline for `convolution_backward`.

Mirrors benchmark/test_convolution_backward.py (ConvolutionBackwardBenchmark, a
GenericBenchmark subclass):
    op_name  = "convolution_backward"
    torch_op = torch.ops.aten.convolution_backward.default
    dtypes   = [float16, bfloat16, float32, float64] (one parametrized run each)
    input_fn = partial(_grouped_inputs, out=False, masks=MASKS)

The class carries its own DEFAULT_SHAPES -- (batch, in_c, out_c, spatial,
groups) -- overrides get_input_iter to walk it and returns [] from
set_more_shapes, so core_shapes.yaml is not consulted. 1-D, 2-D and 3-D spatial
extents and grouped/depthwise cases are all represented.

_grouped_inputs crosses each shape with transposed in (False, True) and the
four output masks, so one shape yields eight collection points. The kernel is
k = (3,) * dim, stride/padding/dilation are all ones and output_padding is
zeros.

float64 is included because the NVIDIA backend reports fp64_enabled=True; the
benchmark skips that row only on backends without FP64.

This calls the ATen backward kernel directly -- it is not an autograd
benchmark (is_backward is not set).
"""

import torch

OP_NAME = "convolution_backward"
DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten.convolution_backward.default is unavailable on this build"
)

# ConvolutionBackwardBenchmark.DEFAULT_SHAPES
# (batch, input channels, output channels, spatial extent, groups)
_SHAPES = [
    (4, 16, 32, (128,), 1),
    (4, 16, 32, (32, 32), 1),
    (2, 8, 16, (12, 12, 12), 1),
    (4, 16, 32, (32, 32), 2),
    (4, 16, 32, (32, 32), 16),
    (2, 3, 5, (17, 19), 1),
]

# test_convolution_backward.py::MASKS
_MASKS = [
    (True, False, False),
    (False, True, False),
    (False, False, True),
    (True, True, True),
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
    op = _resolve("torch.ops.aten.convolution_backward.default")
    return op if callable(op) else None


def grid():
    return [
        {
            "batch": batch,
            "ci": ci,
            "co": co,
            "spatial": list(spatial),
            "groups": groups,
            "transposed": transposed,
            "mask": list(mask),
        }
        for batch, ci, co, spatial, groups in _SHAPES
        for transposed in (False, True)
        for mask in _MASKS
    ]


def build_inputs(binding, dtype, device):
    batch, ci, co = binding["batch"], binding["ci"], binding["co"]
    spatial = tuple(binding["spatial"])
    groups, transposed = binding["groups"], binding["transposed"]
    mask = tuple(binding["mask"])
    dim = len(spatial)
    k = (3,) * dim
    x = torch.randn((batch, ci, *spatial), device=device, dtype=dtype)
    ws = (ci, co // groups, *k) if transposed else (co, ci // groups, *k)
    w = torch.randn(ws, device=device, dtype=dtype)
    go = torch.randn((batch, co, *spatial), device=device, dtype=dtype)
    return (
        go,
        x,
        w,
        [co],
        [1] * dim,
        [1] * dim,
        [1] * dim,
        transposed,
        [0] * dim,
        groups,
        mask,
    ), {}


def key_shape(binding):
    spatial = "x".join(str(d) for d in binding["spatial"])
    mask = "".join(str(int(m)) for m in binding["mask"])
    return (
        f"b{binding['batch']}_ci{binding['ci']}_co{binding['co']}"
        f"_sp{spatial}_g{binding['groups']}"
        f"_t{int(binding['transposed'])}_mask{mask}"
    )


def config(binding, dtype):
    batch, ci, co = binding["batch"], binding["ci"], binding["co"]
    spatial = list(binding["spatial"])
    groups, transposed = binding["groups"], binding["transposed"]
    dim = len(spatial)
    k = [3] * dim
    ws = [ci, co // groups] + k if transposed else [co, ci // groups] + k
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": [batch, co] + spatial, "dtype": dt},
                {"shape": [batch, ci] + spatial, "dtype": dt},
                {"shape": ws, "dtype": dt},
                [co],
                [1] * dim,
                [1] * dim,
                [1] * dim,
                transposed,
                [0] * dim,
                groups,
                list(binding["mask"]),
            ],
            "kwargs": {},
        },
        "outputs": {
            "grad_input": {"shape": [batch, ci] + spatial, "dtype": dt},
            "grad_weight": {"shape": ws, "dtype": dt},
            "grad_bias": {"shape": [co], "dtype": dt},
        },
        "dims": {
            "batch": batch,
            "ci": ci,
            "co": co,
            "spatial": spatial,
            "groups": groups,
            "transposed": transposed,
            "mask": list(binding["mask"]),
        },
    }
