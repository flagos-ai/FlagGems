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

"""NVIDIA native baseline for `cudnn_batch_norm_backward`.

Mirrors benchmark/test_cudnn_batch_norm_backward.py
(CudnnBatchNormBackwardBenchmark, a GenericBenchmark subclass):
    op_name  = "cudnn_batch_norm_backward"
    torch_op = torch.ops.aten.cudnn_batch_norm_backward.default
    dtypes   = [float16, float32] + [float64] when the backend has FP64

The class overrides set_shapes with its own DEFAULT_SHAPES and shape_desc
"N, C, spatial dimensions", so core_shapes.yaml is not consulted.

At --level core `_inputs` runs with use_running_stats=False only, so
running_mean/running_var are None; the True variant is COMPREHENSIVE-only.
save_mean/save_invstd/reserve come from a real forward
torch.ops.aten.cudnn_batch_norm.default call, which build_inputs performs
outside the timed region exactly as the benchmark's input_fn does.

This calls the ATen backward kernel directly; it is not an autograd benchmark.
"""

import torch

OP_NAME = "cudnn_batch_norm_backward"
DTYPES = [torch.float16, torch.float32, torch.float64]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten.cudnn_batch_norm_backward.default is unavailable on this build"
)

# CudnnBatchNormBackwardBenchmark.DEFAULT_SHAPES -- "N, C, spatial dimensions"
_SHAPES = [
    (16, 16, 64),
    (16, 16, 1024),
    (16, 16, 4098),
    (1, 8, 4, 4),
    (16, 8, 128, 128),
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
    op = _resolve("torch.ops.aten.cudnn_batch_norm_backward.default")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    param_dtype = torch.float32 if dtype == torch.float16 else dtype
    inp = torch.randn(shape, dtype=dtype, device=device)
    weight = torch.randn(shape[1], dtype=param_dtype, device=device)
    bias = torch.randn_like(weight)
    # --level core runs use_running_stats=False only.
    running_mean = None
    running_var = None
    grad_output = torch.randn_like(inp)
    _, save_mean, save_invstd, reserve = torch.ops.aten.cudnn_batch_norm.default(
        inp, weight, bias, running_mean, running_var, True, _MOMENTUM, _EPS
    )
    return (
        inp,
        grad_output,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_invstd,
        _EPS,
        reserve,
    ), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    c = shape[1]
    dt = str(dtype)
    pdt = "torch.float32" if dtype == torch.float16 else dt
    return {
        "inputs": {
            "args": [
                {"shape": shape, "dtype": dt},
                {"shape": shape, "dtype": dt},
                {"shape": [c], "dtype": pdt},
                None,
                None,
                {"shape": [c], "dtype": pdt},
                {"shape": [c], "dtype": pdt},
                {"scalar": _EPS},
                {"dtype": "torch.uint8", "note": "cuDNN reserve from forward"},
            ],
            "kwargs": {},
        },
        "outputs": {
            "grad_input": {"shape": shape, "dtype": dt},
            "grad_weight": {"shape": [c], "dtype": pdt},
            "grad_bias": {"shape": [c], "dtype": pdt},
        },
        "dims": {"N": shape[0], "C": c, "shape": shape},
    }
