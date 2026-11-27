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

"""NVIDIA native baseline for `reshape_from_tensor`.

Mirrors benchmark/test_reshape_from_tensor.py (ReshapeFromTensorBenchmark, a
base.Benchmark subclass):
    op_name  = "reshape_from_tensor"
    torch_op = torch.ops.aten._reshape_from_tensor.default
    dtypes   = consts.FLOAT_DTYPES

The class overrides set_shapes with its own (size, transpose) pairs, so
core_shapes.yaml is not consulted. The input is built as (size // 16, 16) and
transposed for the `transpose` cases, which turns the reshape from a
metadata-only view into a materializing copy -- the two regimes the list is
chosen to cover.

The target shape is a CPU int64 tensor [size], not a Python list; that is what
this overload takes.
"""

import torch

OP_NAME = "reshape_from_tensor"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten._reshape_from_tensor.default is unavailable on this build"
)

# ReshapeFromTensorBenchmark.set_shapes -- (size, transpose)
_SHAPES = [
    (64, False),
    (4096, False),
    (4096, True),
    (65536, True),
    (1048576, True),
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
    op = _resolve("torch.ops.aten._reshape_from_tensor.default")
    return op if callable(op) else None


def grid():
    return [{"size": size, "transpose": transpose} for size, transpose in _SHAPES]


def build_inputs(binding, dtype, device):
    size = binding["size"]
    inp = torch.randn((size // 16, 16), dtype=dtype, device=device)
    if binding["transpose"]:
        inp = inp.t()
    shape = torch.tensor([size], dtype=torch.int64)
    return (inp, shape), {}


def key_shape(binding):
    return f"size{binding['size']}_t{int(binding['transpose'])}"


def config(binding, dtype):
    size = binding["size"]
    dt = str(dtype)
    rows = size // 16
    if binding["transpose"]:
        in_shape, stride = [16, rows], [1, 16]
    else:
        in_shape, stride = [rows, 16], [16, 1]
    return {
        "inputs": {
            "args": [
                {"shape": in_shape, "dtype": dt, "stride": stride},
                {"shape": [1], "dtype": "torch.int64", "note": "CPU shape tensor"},
            ],
            "kwargs": {},
        },
        "outputs": {"out": {"shape": [size], "dtype": dt}},
        "dims": {"size": size, "transpose": binding["transpose"]},
    }
