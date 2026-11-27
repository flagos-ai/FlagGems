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

"""NVIDIA native baseline for `cudnn_grid_sampler`.

Mirrors benchmark/test_cudnn_grid_sampler.py (GridSamplerBenchmark, a
base.Benchmark subclass):
    op_name  = "cudnn_grid_sampler"
    torch_op = torch.ops.aten.cudnn_grid_sampler
    dtypes   = [torch.float16, torch.float32]

The class overrides set_shapes with GRID_SAMPLER_SHAPES and shape_desc
"N, C, H_in, W_in, H_out, W_out" and returns [] from set_more_shapes, so
core_shapes.yaml is not consulted.

bfloat16 is excluded: cuDNN rejects it for this operator.
"""

import torch

OP_NAME = "cudnn_grid_sampler"
DTYPES = [torch.float16, torch.float32]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten.cudnn_grid_sampler is unavailable on this torch build"
)

# test_cudnn_grid_sampler.py::GRID_SAMPLER_SHAPES
# "N, C, H_in, W_in, H_out, W_out"
_SHAPES = [
    (1, 3, 32, 32, 32, 32),
    (2, 32, 64, 64, 64, 64),
    (4, 16, 128, 128, 64, 64),
    (2, 64, 64, 64, 128, 128),
    (8, 32, 256, 256, 64, 64),
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
    op = _resolve("torch.ops.aten.cudnn_grid_sampler")
    return op if callable(op) else None


def grid():
    return [
        {
            "N": n,
            "C": c,
            "H_in": h_in,
            "W_in": w_in,
            "H_out": h_out,
            "W_out": w_out,
        }
        for n, c, h_in, w_in, h_out, w_out in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    n, c = binding["N"], binding["C"]
    h_in, w_in = binding["H_in"], binding["W_in"]
    h_out, w_out = binding["H_out"], binding["W_out"]
    inp = torch.randn((n, c, h_in, w_in), dtype=dtype, device=device)
    sample_grid = torch.rand((n, h_out, w_out, 2), dtype=dtype, device=device) * 2 - 1
    return (inp, sample_grid), {}


def key_shape(binding):
    return [
        binding["N"],
        binding["C"],
        binding["H_in"],
        binding["W_in"],
        binding["H_out"],
        binding["W_out"],
    ]


def config(binding, dtype):
    n, c = binding["N"], binding["C"]
    h_in, w_in = binding["H_in"], binding["W_in"]
    h_out, w_out = binding["H_out"], binding["W_out"]
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": [n, c, h_in, w_in], "dtype": dt},
                {"shape": [n, h_out, w_out, 2], "dtype": dt},
            ],
            "kwargs": {},
        },
        "outputs": {"out": {"shape": [n, c, h_out, w_out], "dtype": dt}},
        "dims": {
            "N": n,
            "C": c,
            "H_in": h_in,
            "W_in": w_in,
            "H_out": h_out,
            "W_out": w_out,
        },
    }
