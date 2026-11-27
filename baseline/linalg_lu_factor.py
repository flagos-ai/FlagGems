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

"""NVIDIA native baseline for `linalg_lu_factor`.

Mirrors benchmark/test_linalg_lu_factor.py (LinalgLuFactorBenchmark, a
base.Benchmark subclass):
    op_name  = "linalg_lu_factor"
    torch_op = _torch_lu_factor
    dtypes   = _TEST_DTYPES = [float32, float64] when the backend has FP64

The class overrides get_input_iter to walk LINALG_LU_FACTOR_SHAPE directly and
declares shape_desc "input shape, pivot", so core_shapes.yaml is not consulted.
Each shape is crossed with pivot in _PIVOT_VALUES, which is [True, False] on
CUDA (pivot=False is CUDA-only).

`_torch_lu_factor` is defined twice in that file: a Python LU composite only
when VENDOR == "ascend", and otherwise a thin wrapper over
torch.linalg.lu_factor. This baseline takes the non-Ascend definition.
"""

import torch

OP_NAME = "linalg_lu_factor"
DTYPES = [torch.float32, torch.float64]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = "torch.linalg.lu_factor is unavailable on this torch build"

# test_linalg_lu_factor.py::LINALG_LU_FACTOR_SHAPE
_SHAPES = [
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
    (256, 256),
    (1024, 512),
    (32, 16),
    (16, 32),
    (128, 64),
    (64, 128),
    (4, 32, 32),
    (128, 16, 16),
    (1024, 512, 512),
]

# _PIVOT_VALUES on CUDA; pivot=False is only supported there.
_PIVOT_VALUES = (True, False)


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


def _torch_lu_factor(inp, *, pivot=True):
    """benchmark::_torch_lu_factor, the non-Ascend definition."""
    return torch.linalg.lu_factor(inp, pivot=pivot)


def native():
    if _resolve("torch.linalg.lu_factor") is None:
        return None
    return _torch_lu_factor


def grid():
    return [
        {"shape": list(s), "pivot": pivot} for s in _SHAPES for pivot in _PIVOT_VALUES
    ]


def build_inputs(binding, dtype, device):
    inp = torch.randn(tuple(binding["shape"]), dtype=dtype, device=device)
    return (inp,), {"pivot": binding["pivot"]}


def key_shape(binding):
    shape = "x".join(str(d) for d in binding["shape"])
    return f"{shape}_pivot{int(binding['pivot'])}"


def config(binding, dtype):
    shape = list(binding["shape"])
    m, n = shape[-2], shape[-1]
    batch = shape[:-2]
    k = min(m, n)
    dt = str(dtype)
    return {
        "inputs": {
            "args": [{"shape": shape, "dtype": dt}],
            "kwargs": {"pivot": {"scalar": binding["pivot"]}},
        },
        "outputs": {
            "LU": {"shape": shape, "dtype": dt},
            "pivots": {"shape": batch + [k], "dtype": "torch.int32"},
        },
        "dims": {"shape": shape, "M": m, "N": n, "pivot": binding["pivot"]},
    }
