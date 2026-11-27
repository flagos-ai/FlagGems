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

"""NVIDIA native baseline for `pixel_unshuffle_out`.

Mirrors benchmark/test_pixel_unshuffle.py (PixelUnshuffleOutBenchmark, a
base.Benchmark subclass):
    op_name  = "pixel_unshuffle_out"
    torch_op = torch.ops.aten.pixel_unshuffle.out
    dtypes   = consts.FLOAT_DTYPES

The class builds PIXEL_UNSHUFFLE_SHAPES by reading the `pixel_unshuffle` entry
from core_shapes.yaml itself -- shape_desc "N, C, H, W, downscale_factor" --
and splitting each row into (shape, downscale_factor). It then overrides
set_shapes with that list and returns [] from set_more_shapes, so the operator
name's own yaml lookup never applies.

The out buffer is (N, C * r * r, H // r, W // r).
"""

import torch

OP_NAME = "pixel_unshuffle_out"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
MUTATED_INPUTS = ("out",)
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten.pixel_unshuffle.out is unavailable on this torch build"
)

# yaml:pixel_unshuffle, split into (shape, downscale_factor) as the class does.
_SHAPES = [
    ((1, 3, 8, 8), 2),
    ((2, 4, 12, 6), 3),
    ((4, 16, 64, 48), 4),
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
    op = _resolve("torch.ops.aten.pixel_unshuffle.out")
    return op if callable(op) else None


def grid():
    return [{"shape": list(shape), "downscale_factor": r} for shape, r in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    r = binding["downscale_factor"]
    x = torch.randn(shape, dtype=dtype, device=device)
    n, c, h, w = shape
    out = torch.empty((n, c * r * r, h // r, w // r), dtype=dtype, device=device)
    return (x, r), {"out": out}


def key_shape(binding):
    shape = "x".join(str(d) for d in binding["shape"])
    return f"{shape}_r{binding['downscale_factor']}"


def config(binding, dtype):
    shape = list(binding["shape"])
    r = binding["downscale_factor"]
    n, c, h, w = shape
    out_shape = [n, c * r * r, h // r, w // r]
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": shape, "dtype": dt},
                {"scalar": r},
            ],
            "kwargs": {"out": {"shape": out_shape, "dtype": dt}},
        },
        "outputs": {"out": {"shape": out_shape, "dtype": dt}},
        "dims": {
            "N": n,
            "C": c,
            "H": h,
            "W": w,
            "downscale_factor": r,
        },
    }
