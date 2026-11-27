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

"""NVIDIA native baseline for `grid_sampler`.

Mirrors benchmark/test_grid_sampler.py (GridSamplerBenchmark, a base.Benchmark
subclass):
    op_name  = "grid_sampler"
    torch_op = torch.ops.aten.grid_sampler
    dtypes   = consts.FLOAT_DTYPES

The class overrides set_shapes with its own (input_shape, grid_shape) pairs, so
core_shapes.yaml is not consulted: four 4-D cases plus one 5-D case, spanning
small thumbnails to HD-size grids.

The grid is drawn in float32 and mapped to [-1.1, 1.1] before the cast, so the
padding branches also run while the bulk of the samples stay in bounds -- the
benchmark's comment on stable latencies. The trailing arguments are
(0, 0, False): bilinear interpolation, zeros padding, align_corners=False.
"""

import torch

OP_NAME = "grid_sampler"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten.grid_sampler is unavailable on this torch build"
)

# GridSamplerBenchmark.set_shapes -- (input_shape, grid_shape)
_SHAPES = [
    ((1, 3, 112, 112), (1, 112, 112, 2)),
    ((2, 64, 32, 32), (2, 32, 32, 2)),
    ((4, 256, 16, 16), (4, 16, 16, 2)),
    ((8, 3, 64, 64), (8, 64, 64, 2)),
    ((1, 8, 16, 16, 16), (1, 16, 16, 16, 3)),
]

_INTERPOLATION_MODE = 0  # bilinear
_PADDING_MODE = 0  # zeros
_ALIGN_CORNERS = False


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
    op = _resolve("torch.ops.aten.grid_sampler")
    return op if callable(op) else None


def grid():
    return [{"input_shape": list(inp), "grid_shape": list(g)} for inp, g in _SHAPES]


def build_inputs(binding, dtype, device):
    inp = torch.randn(tuple(binding["input_shape"]), dtype=dtype, device=device)
    sample_grid = torch.rand(
        tuple(binding["grid_shape"]), dtype=torch.float32, device=device
    )
    sample_grid = (sample_grid * 2.2 - 1.1).to(dtype)
    return (
        inp,
        sample_grid,
        _INTERPOLATION_MODE,
        _PADDING_MODE,
        _ALIGN_CORNERS,
    ), {}


def key_shape(binding):
    inp = "x".join(str(d) for d in binding["input_shape"])
    g = "x".join(str(d) for d in binding["grid_shape"])
    return f"in{inp}_grid{g}"


def config(binding, dtype):
    inp = list(binding["input_shape"])
    g = list(binding["grid_shape"])
    dt = str(dtype)
    out_shape = [inp[0], inp[1]] + g[1:-1]
    return {
        "inputs": {
            "args": [
                {"shape": inp, "dtype": dt},
                {"shape": g, "dtype": dt},
                {"scalar": _INTERPOLATION_MODE},
                {"scalar": _PADDING_MODE},
                {"scalar": _ALIGN_CORNERS},
            ],
            "kwargs": {},
        },
        "outputs": {"out": {"shape": out_shape, "dtype": dt}},
        "dims": {
            "input_shape": inp,
            "grid_shape": g,
            "interpolation_mode": _INTERPOLATION_MODE,
            "padding_mode": _PADDING_MODE,
            "align_corners": _ALIGN_CORNERS,
        },
    }
