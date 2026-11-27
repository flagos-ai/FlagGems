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

"""NVIDIA native baseline for `cudnn_affine_grid_generator`.

Mirrors benchmark/test_cudnn_affine_grid_generator.py (CudnnAffineGridBenchmark,
a base.Benchmark subclass):
    op_name  = "cudnn_affine_grid_generator"
    torch_op = torch.cudnn_affine_grid_generator
    dtypes   = [torch.float16, torch.float32, torch.float64]

The class overrides set_shapes with CUDNN_AFFINE_GRID_SHAPES and returns [] from
set_more_shapes, so core_shapes.yaml is not consulted. Each (N, H, W, C) entry
builds theta[N, 2, 3] and calls op(theta, N, C, H, W) -- note the argument order
puts C before H and W.

bfloat16 is excluded: cuDNN rejects it for this operator with
CUDNN_STATUS_BAD_PARAM.
"""

import torch

OP_NAME = "cudnn_affine_grid_generator"
DTYPES = [torch.float16, torch.float32, torch.float64]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.cudnn_affine_grid_generator is unavailable on this torch build"
)

# test_cudnn_affine_grid_generator.py::CUDNN_AFFINE_GRID_SHAPES -- (N, H, W, C)
_SHAPES = [
    (1, 16, 16, 3),
    (2, 32, 32, 3),
    (4, 64, 64, 3),
    (8, 128, 128, 3),
    (16, 256, 256, 3),
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
    op = _resolve("torch.cudnn_affine_grid_generator")
    return op if callable(op) else None


def grid():
    return [{"N": n, "H": h, "W": w, "C": c} for n, h, w, c in _SHAPES]


def build_inputs(binding, dtype, device):
    n, h, w, c = binding["N"], binding["H"], binding["W"], binding["C"]
    theta = torch.randn((n, 2, 3), dtype=dtype, device=device)
    return (theta, n, c, h, w), {}


def key_shape(binding):
    return [binding["N"], binding["H"], binding["W"], binding["C"]]


def config(binding, dtype):
    n, h, w, c = binding["N"], binding["H"], binding["W"], binding["C"]
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": [n, 2, 3], "dtype": dt},
                {"scalar": n},
                {"scalar": c},
                {"scalar": h},
                {"scalar": w},
            ],
            "kwargs": {},
        },
        "outputs": {"grid": {"shape": [n, h, w, 2], "dtype": dt}},
        "dims": {"N": n, "C": c, "H": h, "W": w},
    }
