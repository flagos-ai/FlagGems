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

"""NVIDIA native baseline for `as_strided_copy_out`.

Mirrors benchmark/test_as_strided_copy.py (AsStridedCopyOutBenchmark, a
base.Benchmark subclass):
    op_name  = "as_strided_copy_out"
    torch_op = torch.ops.aten.as_strided_copy
    dtypes   = consts.FLOAT_DTYPES + consts.INT_DTYPES + consts.BOOL_DTYPES

The parent class overrides set_shapes to assign AS_STRIDED_COPY_SHAPES, so
core_shapes.yaml is not consulted. Each entry is
(input_shape, size, stride, storage_offset) and the subclass additionally
allocates out=empty(size).

int64 is deliberately absent: consts.EXTRA_INT_DTYPES is not part of this
operator's dtype list, so only int16/int32 are collected alongside the float
and bool sets.
"""

import torch

OP_NAME = "as_strided_copy_out"
DTYPES = [
    torch.float16,
    torch.float32,
    torch.bfloat16,
    torch.int16,
    torch.int32,
    torch.bool,
]
IS_INPLACE = False
MUTATED_INPUTS = ("out",)
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten.as_strided_copy is unavailable on this torch build"
)

# test_as_strided_copy.py::AS_STRIDED_COPY_SHAPES
# (input shape, size, stride, storage_offset)
_SHAPES = [
    ((64, 64), (64, 64), (64, 1), 0),
    ((256, 256), (128, 128), (1, 256), 0),
    ((1024, 1024), (512, 512), (1024, 1), 0),
    ((64, 128, 64), (32, 64, 32), (8192, 64, 2), 0),
    ((1024 * 1024,), (512 * 1024,), (2,), 0),
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
    op = _resolve("torch.ops.aten.as_strided_copy")
    return op if callable(op) else None


def grid():
    return [
        {
            "shape": list(input_shape),
            "size": list(size),
            "stride": list(stride),
            "storage_offset": storage_offset,
        }
        for input_shape, size, stride, storage_offset in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    inp = _generate_tensor_input(tuple(binding["shape"]), dtype, device)
    size = list(binding["size"])
    stride = list(binding["stride"])
    out = torch.empty(size, dtype=dtype, device=device)
    return (inp, size, stride, binding["storage_offset"]), {"out": out}


def key_shape(binding):
    shape = "x".join(str(d) for d in binding["shape"])
    size = "x".join(str(d) for d in binding["size"])
    stride = "x".join(str(d) for d in binding["stride"])
    return f"in{shape}_size{size}_stride{stride}_off{binding['storage_offset']}"


def config(binding, dtype):
    shape = list(binding["shape"])
    size = list(binding["size"])
    stride = list(binding["stride"])
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": shape, "dtype": dt},
                size,
                stride,
                binding["storage_offset"],
            ],
            "kwargs": {"out": {"shape": size, "dtype": dt}},
        },
        "outputs": {"out": {"shape": size, "dtype": dt}},
        "dims": {
            "input_shape": shape,
            "size": size,
            "stride": stride,
            "storage_offset": binding["storage_offset"],
        },
    }
