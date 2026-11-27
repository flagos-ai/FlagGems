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

"""NVIDIA native baseline for `is_any_true`.

Mirrors benchmark/test_is_any_true.py (AnyTrueBenchmark, a base.Benchmark
subclass):
    op_name  = "is_any_true"
    torch_op = torch.ops.aten._is_any_true.default
    dtypes   = consts.BOOL_DTYPES

The class overrides set_shapes with its own list -- chosen to cover the launch
floor, the one-program boundary and streaming inputs -- so core_shapes.yaml is
not consulted. Each size is crossed with strided in (False, True): the strided
case allocates twice the elements and takes `[::2]`, giving a non-contiguous
input. The last element is set True where the tensor is non-empty, so the
reduction's early-exit path is not hit at element 0.

_is_any_true accepts bool tensors only.
"""

import torch

OP_NAME = "is_any_true"
DTYPES = [torch.bool]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten._is_any_true.default is unavailable on this torch build"
)

# AnyTrueBenchmark.set_shapes -- launch floor, one-program boundary, streaming
_SHAPES = [
    (0,),
    (1,),
    (257,),
    (65536,),
    (65537,),
    (1048576,),
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
    op = _resolve("torch.ops.aten._is_any_true.default")
    return op if callable(op) else None


def grid():
    return [
        {"shape": list(s), "strided": strided}
        for s in _SHAPES
        for strided in (False, True)
    ]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    strided = binding["strided"]
    inp = torch.zeros(shape[0] * (2 if strided else 1), dtype=dtype, device=device)
    if strided:
        inp = inp[::2]
    if inp.numel():
        inp[-1] = True
    return (inp,), {}


def key_shape(binding):
    n = binding["shape"][0]
    return f"n{n}_strided{int(binding['strided'])}"


def config(binding, dtype):
    n = binding["shape"][0]
    stride = [2] if binding["strided"] else [1]
    return {
        "inputs": {
            "args": [{"shape": [n], "dtype": "torch.bool", "stride": stride}],
            "kwargs": {},
        },
        "outputs": {"out": {"shape": [], "dtype": "torch.bool"}},
        "dims": {"n": n, "strided": binding["strided"]},
    }
