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

"""NVIDIA native baseline for `cummax_helper`.

Mirrors benchmark/test_cummax_helper.py (base.GenericBenchmark2DOnly):
    op_name  = "cummax_helper"
    torch_op = torch.ops.aten._cummax_helper
    dtypes   = consts.FLOAT_DTYPES + consts.INT_DTYPES
    shapes   = yaml:GenericBenchmark2DOnly (benchmark/core_shapes.yaml)

input_fn yields (inp, values, indices, 1): the operator writes its results into
the preallocated `values` and `indices` buffers, so both are mutated.
"""

import torch

OP_NAME = "cummax_helper"
DTYPES = [torch.float16, torch.float32, torch.bfloat16, torch.int16, torch.int32]
IS_INPLACE = True
MUTATED_INPUTS = (1, 2)
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten._cummax_helper is unavailable on this torch build"
)

# yaml:GenericBenchmark2DOnly
_SHAPES = [
    (64, 64),
    (256, 256),
    (1024, 1024),
    (4096, 4096),
    (1024, 65536),
]

_DIM = 1

_FLOAT_DTYPES = (torch.float16, torch.float32, torch.bfloat16)
_INT_DTYPES = (torch.int16, torch.int32)
_BOOL_DTYPES = (torch.bool,)
_COMPLEX_DTYPES = (torch.complex64,)


def _generate_tensor_input(shape, dtype, device):
    """benchmark/utils.py::generate_tensor_input, verbatim semantics."""
    if dtype in _FLOAT_DTYPES:
        return torch.randn(shape, dtype=dtype, device=device)
    if dtype in _INT_DTYPES:
        return torch.randint(
            torch.iinfo(dtype).min,
            torch.iinfo(dtype).max,
            shape,
            dtype=dtype,
            device="cpu",
        ).to(device)
    if dtype in _BOOL_DTYPES:
        return torch.randint(0, 2, size=shape, dtype=dtype, device="cpu").to(device)
    if dtype in _COMPLEX_DTYPES:
        return torch.randn(shape, dtype=dtype, device=device)
    raise TypeError(f"unsupported dtype: {dtype}")


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
    op = _resolve("torch.ops.aten._cummax_helper")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    inp = _generate_tensor_input(shape, dtype, device)
    values = torch.empty_like(inp)
    indices = torch.empty(inp.shape, dtype=torch.int64, device=device)
    return (inp, values, indices, _DIM), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": shape, "dtype": dt},
                {"shape": shape, "dtype": dt, "note": "written in place"},
                {"shape": shape, "dtype": "torch.int64", "note": "written in place"},
                {"scalar": _DIM},
            ],
            "kwargs": {},
        },
        "outputs": {
            "values": {"shape": shape, "dtype": dt},
            "indices": {"shape": shape, "dtype": "torch.int64"},
        },
        "dims": {"shape": shape, "dim": _DIM},
    }
