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

"""NVIDIA native baseline for `trace_backward`.

Mirrors benchmark/test_trace_backward.py (base.GenericBenchmark):
    op_name  = "trace_backward"
    torch_op = torch.ops.aten.trace_backward
    dtypes   = consts.FLOAT_DTYPES
    shapes   = yaml:Benchmark (benchmark/core_shapes.yaml)

The operator takes a 0-d gradient and the target [n, m] size, so the core shape
is read only for its first two dimensions (or doubled for a 1-D entry) and no
input tensor of that shape is built.

The input_fn returns without yielding when n * m exceeds 64M elements, because
trace_backward materializes the full matrix; that removes three of the five
core shapes, leaving (64, 64) and (4096, 4096). unsupported_reason() keeps the
rule visible.

This calls the ATen backward kernel directly; it is not an autograd benchmark.
"""

import torch

OP_NAME = "trace_backward"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten.trace_backward is unavailable on this torch build"
)

_MAX_ELEMENTS = 64 * 1024 * 1024
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
    """_input_fn: (n, n) for a 1-D entry, shape[:2] otherwise."""
    if len(shape) == 1:
        return shape[0], shape[0]
    return shape[0], shape[1]


def native():
    op = _resolve("torch.ops.aten.trace_backward")
    return op if callable(op) else None


def grid():
    bindings = []
    for shape in _CORE_SHAPES:
        n, m = _n_m(shape)
        if n * m > _MAX_ELEMENTS:
            # The input_fn yields nothing for these.
            continue
        bindings.append({"shape": list(shape), "n": n, "m": m})
    return bindings


def build_inputs(binding, dtype, device):
    grad = torch.randn((), dtype=dtype, device=device)
    return (grad, [binding["n"], binding["m"]]), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    n, m = binding["n"], binding["m"]
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": [], "dtype": dt, "note": "0-d grad"},
                [n, m],
            ],
            "kwargs": {},
        },
        "outputs": {"grad_input": {"shape": [n, m], "dtype": dt}},
        "dims": {"n": n, "m": m, "shape": list(binding["shape"])},
    }
