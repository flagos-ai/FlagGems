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

"""NVIDIA native baseline for `grid_sampler_3d`.

Mirrors benchmark/test_grid_sampler_3d.py (GridSampler3DBenchmark, a
base.Benchmark subclass):
    op_name  = "grid_sampler_3d"
    torch_op = torch.ops.aten.grid_sampler_3d
    dtypes   = consts.FLOAT_DTYPES

The class overrides set_shapes with GRID_SAMPLER_3D_SHAPES -- read as
(N, C, iD, iH, iW) -- so core_shapes.yaml is not consulted. The output volume
is half the input in each spatial dimension.

The grid is randn * 1.5 (not a uniform in [-1, 1]), which deliberately covers
more than the valid spatial range so the padding branches run. The trailing
arguments are (0, 0, False): bilinear, zeros padding, align_corners=False.
"""

import torch

OP_NAME = "grid_sampler_3d"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten.grid_sampler_3d is unavailable on this torch build"
)

# test_grid_sampler_3d.py::GRID_SAMPLER_3D_SHAPES -- (N, C, iD, iH, iW)
_SHAPES = [
    (1, 3, 8, 8, 8),
    (2, 8, 16, 16, 16),
    (4, 16, 32, 32, 32),
    (8, 32, 64, 64, 64),
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
    op = _resolve("torch.ops.aten.grid_sampler_3d")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    n, _, i_d, i_h, i_w = shape
    o_d, o_h, o_w = i_d // 2, i_h // 2, i_w // 2
    inp = torch.randn(shape, dtype=dtype, device=device)
    sample_grid = torch.randn(n, o_d, o_h, o_w, 3, dtype=dtype, device=device)
    sample_grid = sample_grid * 1.5  # cover more spatial range
    return (
        inp,
        sample_grid,
        _INTERPOLATION_MODE,
        _PADDING_MODE,
        _ALIGN_CORNERS,
    ), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    n, c, i_d, i_h, i_w = shape
    o_d, o_h, o_w = i_d // 2, i_h // 2, i_w // 2
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": shape, "dtype": dt},
                {"shape": [n, o_d, o_h, o_w, 3], "dtype": dt},
                {"scalar": _INTERPOLATION_MODE},
                {"scalar": _PADDING_MODE},
                {"scalar": _ALIGN_CORNERS},
            ],
            "kwargs": {},
        },
        "outputs": {"out": {"shape": [n, c, o_d, o_h, o_w], "dtype": dt}},
        "dims": {
            "N": n,
            "C": c,
            "iD": i_d,
            "iH": i_h,
            "iW": i_w,
            "oD": o_d,
            "oH": o_h,
            "oW": o_w,
        },
    }
