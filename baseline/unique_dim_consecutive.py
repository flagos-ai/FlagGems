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

"""NVIDIA native baseline for `unique_dim_consecutive`.

Mirrors benchmark/test_unique_dim_consecutive.py (base.GenericBenchmark2DOnly):
    op_name  = "unique_dim_consecutive"
    torch_op = torch.ops.aten.unique_dim_consecutive.default
    dtypes   = consts.FLOAT_DTYPES
    shapes   = yaml:GenericBenchmark2DOnly (benchmark/core_shapes.yaml)

The input_fn yields (inp, dim=0, return_inverse=True, return_counts=False),
all three as keywords.
"""

import torch

OP_NAME = "unique_dim_consecutive"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten.unique_dim_consecutive.default is unavailable on this build"
)

# yaml:GenericBenchmark2DOnly
_SHAPES = [
    (64, 64),
    (256, 256),
    (1024, 1024),
    (4096, 4096),
    (1024, 65536),
]

_DIM = 0
_RETURN_INVERSE = True
_RETURN_COUNTS = False

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
    op = _resolve("torch.ops.aten.unique_dim_consecutive.default")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    inp = _generate_tensor_input(tuple(binding["shape"]), dtype, device)
    return (inp,), {
        "dim": _DIM,
        "return_inverse": _RETURN_INVERSE,
        "return_counts": _RETURN_COUNTS,
    }


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    dt = str(dtype)
    return {
        "inputs": {
            "args": [{"shape": shape, "dtype": dt}],
            "kwargs": {
                "dim": {"scalar": _DIM},
                "return_inverse": {"scalar": _RETURN_INVERSE},
                "return_counts": {"scalar": _RETURN_COUNTS},
            },
        },
        "outputs": {
            "out": {"dtype": dt, "note": "data-dependent row count"},
            "inverse": {"shape": [shape[_DIM]], "dtype": "torch.int64"},
            "counts": {"dtype": "torch.int64"},
        },
        "dims": {"shape": shape, "dim": _DIM},
    }
