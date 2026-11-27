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

"""NVIDIA native baseline for `grid_sampler_3d_backward`.

Mirrors benchmark/test_grid_sampler_3d_backward.py
(GridSampler3dBackwardBenchmark, a base.Benchmark subclass):
    op_name  = "grid_sampler_3d_backward"
    torch_op = torch.ops.aten.grid_sampler_3d_backward.default
    dtypes   = [torch.float32]

The class overrides set_shapes with its own
(N, C, iD, iH, iW, oD, oH, oW) list, so core_shapes.yaml is not consulted.
The grid is drawn in [-1, 1], interpolation_mode 0 (bilinear), padding_mode 0
(zeros), align_corners True and output_mask [True, True].

float32 only: the CUDA kernel supports float32/float64 and the benchmark pins
float32.

This calls the ATen backward kernel directly; it is not an autograd benchmark.
"""

import torch

OP_NAME = "grid_sampler_3d_backward"
DTYPES = [torch.float32]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten.grid_sampler_3d_backward.default is unavailable on this build"
)

# GridSampler3dBackwardBenchmark.set_shapes -- (N, C, iD, iH, iW, oD, oH, oW)
_SHAPES = [
    (1, 3, 8, 8, 8, 4, 4, 4),
    (2, 16, 16, 16, 16, 8, 8, 8),
    (4, 32, 16, 16, 16, 8, 8, 8),
    (2, 64, 32, 32, 32, 16, 16, 16),
    (1, 128, 32, 32, 32, 16, 16, 16),
]

_FIELDS = ("N", "C", "iD", "iH", "iW", "oD", "oH", "oW")
_INTERPOLATION_MODE = 0  # bilinear
_PADDING_MODE = 0  # zeros
_ALIGN_CORNERS = True
_OUTPUT_MASK = [True, True]


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
    op = _resolve("torch.ops.aten.grid_sampler_3d_backward.default")
    return op if callable(op) else None


def grid():
    return [dict(zip(_FIELDS, shape)) for shape in _SHAPES]


def build_inputs(binding, dtype, device):
    n, c = binding["N"], binding["C"]
    i_d, i_h, i_w = binding["iD"], binding["iH"], binding["iW"]
    o_d, o_h, o_w = binding["oD"], binding["oH"], binding["oW"]
    grad_output = torch.randn(n, c, o_d, o_h, o_w, device=device, dtype=dtype)
    input_tensor = torch.randn(n, c, i_d, i_h, i_w, device=device, dtype=dtype)
    sample_grid = torch.rand(n, o_d, o_h, o_w, 3, device=device, dtype=dtype) * 2 - 1
    return (
        grad_output,
        input_tensor,
        sample_grid,
        _INTERPOLATION_MODE,
        _PADDING_MODE,
        _ALIGN_CORNERS,
        list(_OUTPUT_MASK),
    ), {}


def key_shape(binding):
    return [binding[key] for key in _FIELDS]


def config(binding, dtype):
    n, c = binding["N"], binding["C"]
    i_d, i_h, i_w = binding["iD"], binding["iH"], binding["iW"]
    o_d, o_h, o_w = binding["oD"], binding["oH"], binding["oW"]
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": [n, c, o_d, o_h, o_w], "dtype": dt},
                {"shape": [n, c, i_d, i_h, i_w], "dtype": dt},
                {"shape": [n, o_d, o_h, o_w, 3], "dtype": dt},
                {"scalar": _INTERPOLATION_MODE},
                {"scalar": _PADDING_MODE},
                {"scalar": _ALIGN_CORNERS},
                list(_OUTPUT_MASK),
            ],
            "kwargs": {},
        },
        "outputs": {
            "grad_input": {"shape": [n, c, i_d, i_h, i_w], "dtype": dt},
            "grad_grid": {"shape": [n, o_d, o_h, o_w, 3], "dtype": dt},
        },
        "dims": {key: binding[key] for key in _FIELDS},
    }
