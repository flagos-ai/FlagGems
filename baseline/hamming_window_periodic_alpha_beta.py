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

"""NVIDIA native baseline for `hamming_window_periodic_alpha_beta`.

Mirrors benchmark/test_hamming_window_periodic_alpha.py (base.GenericBenchmark):
    op_name  = "hamming_window_periodic_alpha_beta"
    torch_op = torch.ops.aten.hamming_window.periodic_alpha_beta
    dtypes   = consts.FLOAT_DTYPES
    shapes   = yaml:Benchmark (benchmark/core_shapes.yaml)

This is a factory operator: it takes no input tensor. window_length is
math.prod(shape) of the core shape, so the shape list is used only as an
element-count source. At --level core periodic is True; the periodic=False
variant is COMPREHENSIVE-only. alpha 0.54 / beta 0.46 are the classic Hamming
coefficients.

All arguments are passed as keywords, as the input_fn yields them.
"""

import math

import torch

OP_NAME = "hamming_window_periodic_alpha_beta"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten.hamming_window.periodic_alpha_beta is unavailable on this build"
)

# yaml:Benchmark -- the shared default shape list
_SHAPES = [
    (1073741824,),
    (64, 64),
    (4096, 4096),
    (64, 512, 512),
    (1024, 1024, 1024),
]

_PERIODIC = True
_ALPHA = 0.54
_BETA = 0.46


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
    op = _resolve("torch.ops.aten.hamming_window.periodic_alpha_beta")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s), "window_length": math.prod(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    return (), {
        "window_length": binding["window_length"],
        "periodic": _PERIODIC,
        "alpha": _ALPHA,
        "beta": _BETA,
        "dtype": dtype,
        "device": device,
    }


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    length = binding["window_length"]
    dt = str(dtype)
    return {
        "inputs": {
            "args": [],
            "kwargs": {
                "window_length": {"scalar": length},
                "periodic": {"scalar": _PERIODIC},
                "alpha": {"scalar": _ALPHA},
                "beta": {"scalar": _BETA},
                "dtype": dt,
                "device": "cuda:0",
            },
        },
        "outputs": {"out": {"shape": [length], "dtype": dt}},
        "dims": {
            "shape": list(binding["shape"]),
            "window_length": length,
            "periodic": _PERIODIC,
            "alpha": _ALPHA,
            "beta": _BETA,
        },
    }
