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

"""NVIDIA native baseline for `fake_quantize_per_channel_affine_cachemask_out`.

Mirrors benchmark/test_fake_quantize_per_channel_affine_cachemask.py
(BenchmarkFakeQuantizePerChannelAffineCachemaskOut, a base.Benchmark subclass):
    op_name  = "fake_quantize_per_channel_affine_cachemask_out"
    torch_op = torch.ops.aten.fake_quantize_per_channel_affine_cachemask.out
    dtypes   = consts.FLOAT_DTYPES

The parent class overrides set_shapes with its own DEFAULT_SHAPES, so
core_shapes.yaml is not consulted, and crosses each shape with axis in (0, 1),
so one shape yields two collection points. scale is float32 drawn
rand(channels) + 0.1 and zero_point is int32 zeros, where
channels = shape[axis]. quant_min/quant_max are 0/255.

The .out variant additionally allocates out0 (same dtype as input) and out1
(bool mask).
"""

import torch

OP_NAME = "fake_quantize_per_channel_affine_cachemask_out"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
MUTATED_INPUTS = ("out0", "out1")
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten.fake_quantize_per_channel_affine_cachemask.out is unavailable"
)

# BenchmarkFakeQuantizePerChannelAffineCachemask.DEFAULT_SHAPES
_SHAPES = [
    (4, 4),
    (64, 64),
    (128, 256),
    (512, 512),
    (1024, 1024),
    (2, 3, 128, 128),
    (8, 16, 64, 64),
]

_AXIS_CONFIGS = (0, 1)
_QUANT_MIN = 0
_QUANT_MAX = 255


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
    op = _resolve("torch.ops.aten.fake_quantize_per_channel_affine_cachemask.out")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s), "axis": axis} for s in _SHAPES for axis in _AXIS_CONFIGS]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    axis = binding["axis"]
    inp = torch.randn(shape, dtype=dtype, device=device)
    channels = shape[axis]
    scale = torch.rand(channels, dtype=torch.float32, device=device) + 0.1
    zero_point = torch.zeros(channels, dtype=torch.int32, device=device)
    out0 = torch.empty_like(inp)
    out1 = torch.empty_like(inp, dtype=torch.bool)
    return (inp, scale, zero_point, axis, _QUANT_MIN, _QUANT_MAX), {
        "out0": out0,
        "out1": out1,
    }


def key_shape(binding):
    shape = "x".join(str(d) for d in binding["shape"])
    return f"{shape}_axis{binding['axis']}"


def config(binding, dtype):
    shape = list(binding["shape"])
    axis = binding["axis"]
    channels = shape[axis]
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": shape, "dtype": dt},
                {"shape": [channels], "dtype": "torch.float32"},
                {"shape": [channels], "dtype": "torch.int32"},
                {"scalar": axis},
                {"scalar": _QUANT_MIN},
                {"scalar": _QUANT_MAX},
            ],
            "kwargs": {
                "out0": {"shape": shape, "dtype": dt},
                "out1": {"shape": shape, "dtype": "torch.bool"},
            },
        },
        "outputs": {
            "out0": {"shape": shape, "dtype": dt},
            "out1": {"shape": shape, "dtype": "torch.bool"},
        },
        "dims": {"shape": shape, "axis": axis, "channels": channels},
    }
