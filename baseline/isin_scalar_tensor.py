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

"""NVIDIA native baseline for `isin_scalar_tensor`.

Mirrors benchmark/test_isin.py (IsinBenchmark, a GenericBenchmark2DOnly
subclass):
    op_name  = "isin_scalar_tensor"
    torch_op = torch.isin
    dtypes   = consts.INT_DTYPES
    shapes   = yaml:GenericBenchmark2DOnly (benchmark/core_shapes.yaml),
               then filtered by the class to <= 2**29 elements

This is the (Scalar, Tensor) overload: the first argument is a Python int taken
from the first element of the generated tensor, and the second is the tensor of
test elements. The assume_unique=True variant is COMPREHENSIVE-only.

All five core shapes are <= 2**29 elements, so the class's filter removes none
of them.
"""

import torch

OP_NAME = "isin_scalar_tensor"
DTYPES = [torch.int16, torch.int32]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = "torch.isin is unavailable on this torch build"

# yaml:GenericBenchmark2DOnly, all within the class's 2**29-element cap
_MAX_ELEMENTS = 2**29
_SHAPES = [
    (64, 64),
    (256, 256),
    (1024, 1024),
    (4096, 4096),
    (1024, 65536),
]

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
    op = _resolve("torch.isin")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    test_elements = _generate_tensor_input(shape, dtype, device)
    scalar_val = int(test_elements.ravel()[0].item())
    return (scalar_val, test_elements), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"note": "python int taken from test_elements[0]"},
                {"shape": shape, "dtype": dt},
            ],
            "kwargs": {},
        },
        "outputs": {"out": {"shape": [], "dtype": "torch.bool"}},
        "dims": {"shape": shape},
    }
