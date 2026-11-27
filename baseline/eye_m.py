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

"""NVIDIA native baseline for `eye_m`.

Mirrors benchmark/test_eye.py (base.GenericBenchmark):
    op_name  = "eye_m"
    torch_op = torch.eye
    dtypes   = not passed, so Benchmark.DEFAULT_DTYPES = consts.FLOAT_DTYPES
    shapes   = yaml:Benchmark (benchmark/core_shapes.yaml)

This is the two-argument `eye.m` overload: n and m are read off the core shape
(shape[:2] for rank >= 2, or shape[0] twice for a 1-D entry) and passed as
keywords together with dtype and device, exactly as _input_fn_eye_m yields
them.

_input_fn_eye_m returns None -- yielding no input at all -- when
shape[0] >= 819200, because an n x m identity matrix of that side length cannot
be allocated. That guard removes the (1073741824,) core shape from the grid,
which is why four bindings remain. unsupported_reason() keeps the rule visible.

The COMPREHENSIVE-only 2**8..2**12 square sweep is not part of this grid.
"""

import torch

OP_NAME = "eye_m"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = "torch.eye is unavailable on this torch build"

# yaml:Benchmark, minus the shapes the input_fn refuses (shape[0] >= 819200)
_MAX_LEADING_DIM = 819200
_CORE_SHAPES = [
    (1073741824,),
    (64, 64),
    (4096, 4096),
    (64, 512, 512),
    (1024, 1024, 1024),
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


def _n_m(shape):
    """_input_fn_eye_m: (n, n) for a 1-D entry, shape[:2] otherwise."""
    if len(shape) == 1:
        return shape[0], shape[0]
    return shape[0], shape[1]


def native():
    op = _resolve("torch.eye")
    return op if callable(op) else None


def grid():
    bindings = []
    for shape in _CORE_SHAPES:
        if shape[0] >= _MAX_LEADING_DIM:
            # _input_fn_eye_m yields nothing for these.
            continue
        n, m = _n_m(shape)
        bindings.append({"shape": list(shape), "n": n, "m": m})
    return bindings


def build_inputs(binding, dtype, device):
    return (), {
        "n": binding["n"],
        "m": binding["m"],
        "dtype": dtype,
        "device": device,
    }


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    n, m = binding["n"], binding["m"]
    dt = str(dtype)
    return {
        "inputs": {
            "args": [],
            "kwargs": {
                "n": {"scalar": n},
                "m": {"scalar": m},
                "dtype": dt,
                "device": "cuda:0",
            },
        },
        "outputs": {"out": {"shape": [n, m], "dtype": dt}},
        "dims": {"n": n, "m": m, "shape": list(binding["shape"])},
    }
