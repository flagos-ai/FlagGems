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

"""NVIDIA native baseline for `linalg_det_out`.

Mirrors benchmark/test_linalg_det.py (DetOutBenchmark, a base.Benchmark
subclass):
    op_name  = "linalg_det_out"
    torch_op = _torch_det_out (the module-local reference below)
    dtypes   = [float32] + [float64] when the backend has FP64

The class overrides set_shapes with DET_SHAPES, so core_shapes.yaml is not
consulted.

`_torch_det_out` has an NPU branch that composes an LU factorization out of
elementwise torch ops; on CUDA it calls torch.linalg.det(A, out=out) directly,
which is what this baseline resolves.
"""

import torch

OP_NAME = "linalg_det_out"
DTYPES = [torch.float32, torch.float64]
IS_INPLACE = False
MUTATED_INPUTS = ("out",)
NATIVE_UNAVAILABLE_REASON = "torch.linalg.det is unavailable on this torch build"

# test_linalg_det.py::DET_SHAPES
_SHAPES = [
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
    (256, 256),
    (4096, 4, 4),
    (1024, 8, 8),
    (1024, 16, 16),
    (128, 16, 16),
    (4, 32, 32),
    (512, 32, 32),
    (256, 64, 64),
    (32, 128, 128),
    (8, 256, 256),
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


def _torch_det_out(a, *, out):
    """benchmark::_torch_det_out, CUDA branch (the npu path is not reproduced)."""
    return torch.linalg.det(a, out=out)


def native():
    if _resolve("torch.linalg.det") is None:
        return None
    return _torch_det_out


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    a = torch.randn(shape, dtype=dtype, device=device)
    out = torch.empty(shape[:-2], dtype=dtype, device=device)
    return (a,), {"out": out}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    dt = str(dtype)
    out_shape = shape[:-2]
    return {
        "inputs": {
            "args": [{"shape": shape, "dtype": dt}],
            "kwargs": {"out": {"shape": out_shape, "dtype": dt}},
        },
        "outputs": {"out": {"shape": out_shape, "dtype": dt}},
        "dims": {"shape": shape, "N": shape[-1]},
    }
