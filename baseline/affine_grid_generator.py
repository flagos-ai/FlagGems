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

"""NVIDIA native baseline for `affine_grid_generator`.

Mirrors benchmark/test_affine_grid_generator.py (AffineGridBenchmark, a
base.Benchmark subclass):
    op_name  = "affine_grid_generator"
    torch_op = torch.affine_grid_generator
    dtypes   = [torch.float32]

The class overrides set_shapes to assign its module-level AFFINE_GRID_SHAPES
directly, so core_shapes.yaml is never consulted. Each (N, H, W) triple becomes
theta[N, 2, 3] with size [N, 3, H, W] and align_corners=False.

float32 only: the affine_grid kernel accumulates in float32 and Half inputs
mismatch the reference.
"""

import torch

OP_NAME = "affine_grid_generator"
DTYPES = [torch.float32]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.affine_grid_generator is unavailable on this torch build"
)

# test_affine_grid_generator.py::AFFINE_GRID_SHAPES -- read as (N, H, W)
_SHAPES = [
    (1, 4, 4),
    (1, 8, 8),
    (2, 8, 8),
    (4, 16, 16),
]

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
    op = _resolve("torch.affine_grid_generator")
    return op if callable(op) else None


def grid():
    return [{"N": n, "H": h, "W": w} for n, h, w in _SHAPES]


def build_inputs(binding, dtype, device):
    n, h, w = binding["N"], binding["H"], binding["W"]
    theta = torch.randn((n, 2, 3), dtype=dtype, device=device)
    size = [n, 3, h, w]
    return (theta, size, _ALIGN_CORNERS), {}


def key_shape(binding):
    return [binding["N"], binding["H"], binding["W"]]


def config(binding, dtype):
    n, h, w = binding["N"], binding["H"], binding["W"]
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": [n, 2, 3], "dtype": dt},
                [n, 3, h, w],
                _ALIGN_CORNERS,
            ],
            "kwargs": {},
        },
        "outputs": {"grid": {"shape": [n, h, w, 2], "dtype": dt}},
        "dims": {"N": n, "H": h, "W": w},
    }
