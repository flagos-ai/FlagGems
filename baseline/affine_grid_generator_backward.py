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

"""NVIDIA native baseline for `affine_grid_generator_backward`.

Mirrors benchmark/test_affine_grid_generator_backward.py
(AffineGridBackwardBenchmark, a base.Benchmark subclass):
    op_name  = "affine_grid_generator_backward"
    torch_op = torch.ops.aten.affine_grid_generator_backward
    dtypes   = list(consts.FLOAT_DTYPES)

The class overrides set_shapes to assign AFFINE_GRID_BACKWARD_SHAPES and sets
shape_desc "N, H, W", so core_shapes.yaml is not consulted. Each triple becomes
grad_output[N, H, W, 2] with size [N, 1, H, W] and align_corners=True.

This calls the ATen backward kernel directly; it is not an autograd benchmark
(is_backward is not set), so the whole call is the measured region.

float64 is in the dtype list because the benchmark appends it when
flag_gems.runtime.device.support_fp64 is true, and the NVIDIA backend
reports fp64_enabled=True.
"""

import torch

OP_NAME = "affine_grid_generator_backward"
DTYPES = [torch.float16, torch.float32, torch.bfloat16, torch.float64]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten.affine_grid_generator_backward is unavailable on this torch build"
)

# test_affine_grid_generator_backward.py::AFFINE_GRID_BACKWARD_SHAPES -- "N, H, W"
_SHAPES = [
    (1, 64, 64),
    (4, 128, 128),
    (8, 256, 256),
    (16, 64, 64),
    (32, 32, 32),
    (2, 512, 512),
]

_ALIGN_CORNERS = True


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
    op = _resolve("torch.ops.aten.affine_grid_generator_backward")
    return op if callable(op) else None


def grid():
    return [{"N": n, "H": h, "W": w} for n, h, w in _SHAPES]


def build_inputs(binding, dtype, device):
    n, h, w = binding["N"], binding["H"], binding["W"]
    grad_output = torch.randn((n, h, w, 2), dtype=dtype, device=device)
    size = [n, 1, h, w]
    return (grad_output, size, _ALIGN_CORNERS), {}


def key_shape(binding):
    return [binding["N"], binding["H"], binding["W"]]


def config(binding, dtype):
    n, h, w = binding["N"], binding["H"], binding["W"]
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": [n, h, w, 2], "dtype": dt},
                [n, 1, h, w],
                _ALIGN_CORNERS,
            ],
            "kwargs": {},
        },
        "outputs": {"grad_theta": {"shape": [n, 2, 3], "dtype": dt}},
        "dims": {"N": n, "H": h, "W": w},
    }
