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

"""NVIDIA native baseline for `bitwise_right_shift_`.

Mirrors benchmark/test_bitwise_right_shift_.py
(BitwiseRightShiftInplaceBenchmark, a base.Benchmark subclass):
    op_name    = "bitwise_right_shift_"
    torch_op   = lambda a, b: a.bitwise_right_shift_(b)
    dtypes     = consts.INT_DTYPES
    is_inplace = True
    shapes     = yaml:Benchmark (benchmark/core_shapes.yaml)

The shift amount is a full tensor of the same shape and dtype drawn from
randint(0, 8) on the CPU and moved to the device, matching the benchmark's
get_input_iter.
"""

import torch

OP_NAME = "bitwise_right_shift_"
DTYPES = [torch.int16, torch.int32]
IS_INPLACE = True
MUTATED_INPUTS = (0,)
NATIVE_UNAVAILABLE_REASON = (
    "torch.Tensor.bitwise_right_shift_ is unavailable on this torch build"
)

# yaml:Benchmark -- the shared default shape list
_SHAPES = [
    (1073741824,),
    (64, 64),
    (4096, 4096),
    (64, 512, 512),
    (1024, 1024, 1024),
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


def _bitwise_right_shift_(a, b):
    """benchmark torch_op: lambda a, b: a.bitwise_right_shift_(b)"""
    return a.bitwise_right_shift_(b)


def native():
    if _resolve("torch.Tensor.bitwise_right_shift_") is None:
        return None
    return _bitwise_right_shift_


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    inp1 = _generate_tensor_input(shape, dtype, device)
    shift_amount = torch.randint(0, 8, shape, dtype=dtype, device="cpu").to(device)
    return (inp1, shift_amount), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": shape, "dtype": dt, "note": "written in place"},
                {"shape": shape, "dtype": dt},
            ],
            "kwargs": {},
        },
        "outputs": {"self": {"shape": shape, "dtype": dt}},
        "dims": {"shape": shape},
    }
