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

"""NVIDIA native baseline for `tril_out`.

Mirrors benchmark/test_tril.py (base.GenericBenchmarkExcluse1D):
    op_name  = "tril_out"
    torch_op = torch.tril
    dtypes   = consts.FLOAT_DTYPES
    shapes   = yaml:GenericBenchmarkExcluse1D (benchmark/core_shapes.yaml)

The out buffer is deliberately non-contiguous: `_tril_out_transposed_input_fn`
allocates it with the last two dimensions swapped and then transposes them
back, so the operator writes into a transposed view. That is the point of this
benchmark variant. diagonal is 0.
"""

import torch

OP_NAME = "tril_out"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
MUTATED_INPUTS = ("out",)
NATIVE_UNAVAILABLE_REASON = "torch.tril is unavailable on this torch build"

# yaml:GenericBenchmarkExcluse1D
_SHAPES = [
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (64, 512, 512),
    (1024, 1024, 1024),
]

_DIAGONAL = 0

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
    op = _resolve("torch.tril")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    inp = _generate_tensor_input(shape, dtype, device)
    out_shape = (*shape[:-2], shape[-1], shape[-2])
    out = torch.empty(out_shape, dtype=dtype, device=device).transpose(-2, -1)
    return (inp, _DIAGONAL), {"out": out}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": shape, "dtype": dt},
                {"scalar": _DIAGONAL},
            ],
            "kwargs": {
                "out": {
                    "shape": shape,
                    "dtype": dt,
                    "note": "transposed view, non-contiguous",
                }
            },
        },
        "outputs": {"out": {"shape": shape, "dtype": dt}},
        "dims": {"shape": shape, "diagonal": _DIAGONAL},
    }
