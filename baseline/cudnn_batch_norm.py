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

"""NVIDIA native baseline for `cudnn_batch_norm`.

Mirrors benchmark/test_cudnn_batch_norm.py (CudnnBatchNormBenchmark, a
GenericBenchmark subclass):
    op_name  = "cudnn_batch_norm"
    torch_op = torch.ops.aten.cudnn_batch_norm.default
    dtypes   = [float16, float32] + [float64] when the backend has FP64

The class overrides set_shapes with its own DEFAULT_SHAPES and shape_desc
"N, C, spatial dimensions", so core_shapes.yaml is not consulted.

_inputs crosses each shape with training in (True, False), so one shape yields
two collection points. Parameters (weight/bias/running stats) are float32 when
the input is float16 and follow the input dtype otherwise. momentum 0.1,
eps 1e-5.

float64 is listed because the NVIDIA backend reports fp64_enabled=True.
"""

import torch

OP_NAME = "cudnn_batch_norm"
DTYPES = [torch.float16, torch.float32, torch.float64]
IS_INPLACE = True
MUTATED_INPUTS = (3, 4)
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten.cudnn_batch_norm.default is unavailable on this torch build"
)

# CudnnBatchNormBenchmark.DEFAULT_SHAPES -- "N, C, spatial dimensions"
_SHAPES = [
    (16, 32),
    (16, 16, 64),
    (16, 16, 1024),
    (16, 16, 4098),
    (1, 8, 4, 4),
    (16, 8, 128, 128),
    (2, 16, 7, 9, 11),
]

_MOMENTUM = 0.1
_EPS = 1e-5


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
    op = _resolve("torch.ops.aten.cudnn_batch_norm.default")
    return op if callable(op) else None


def grid():
    return [
        {"shape": list(s), "training": training}
        for s in _SHAPES
        for training in (True, False)
    ]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    param_dtype = torch.float32 if dtype == torch.float16 else dtype
    x = torch.randn(shape, dtype=dtype, device=device)
    w = torch.randn(shape[1], dtype=param_dtype, device=device)
    b = torch.randn_like(w)
    rm = torch.zeros_like(w)
    rv = torch.ones_like(w)
    return (x, w, b, rm, rv, binding["training"], _MOMENTUM, _EPS), {}


def key_shape(binding):
    shape = "x".join(str(d) for d in binding["shape"])
    return f"{shape}_train{int(binding['training'])}"


def config(binding, dtype):
    shape = list(binding["shape"])
    c = shape[1]
    training = binding["training"]
    dt = str(dtype)
    pdt = "torch.float32" if dtype == torch.float16 else dt
    return {
        "inputs": {
            "args": [
                {"shape": shape, "dtype": dt},
                {"shape": [c], "dtype": pdt},
                {"shape": [c], "dtype": pdt},
                {"shape": [c], "dtype": pdt, "note": "updated in place"},
                {"shape": [c], "dtype": pdt, "note": "updated in place"},
                {"scalar": training},
                {"scalar": _MOMENTUM},
                {"scalar": _EPS},
            ],
            "kwargs": {},
        },
        "outputs": {
            "out": {"shape": shape, "dtype": dt},
            "save_mean": {"shape": [c if training else 0], "dtype": pdt},
            "save_invstd": {"shape": [c if training else 0], "dtype": pdt},
            "reserve": {"dtype": "torch.uint8"},
        },
        "dims": {"N": shape[0], "C": c, "shape": shape, "training": training},
    }
