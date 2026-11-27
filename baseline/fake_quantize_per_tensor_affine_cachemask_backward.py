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

"""NVIDIA native baseline for
`fake_quantize_per_tensor_affine_cachemask_backward`.

Mirrors benchmark/test_fake_quantize_per_tensor_affine_cachemask_backward.py
(FakeQuantizePerTensorAffineCachemaskBackwardBenchmark, a GenericBenchmark
subclass):
    op_name  = "fake_quantize_per_tensor_affine_cachemask_backward"
    torch_op = torch.ops.aten.fake_quantize_per_tensor_affine_cachemask_backward
    dtypes   = consts.FLOAT_DTYPES

The class overrides set_shapes with its own eleven-entry list, so
core_shapes.yaml is not consulted. The input is (grad, mask) where mask is a
bool tensor of the same shape drawn from randint(0, 2).

This calls the ATen backward kernel directly; it is not an autograd benchmark.
"""

import torch

OP_NAME = "fake_quantize_per_tensor_affine_cachemask_backward"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten.fake_quantize_per_tensor_affine_cachemask_backward "
    "is unavailable on this torch build"
)

# FakeQuantizePerTensorAffineCachemaskBackwardBenchmark.set_shapes
_SHAPES = [
    (4, 4),
    (64, 64),
    (128, 256),
    (512, 512),
    (1024, 1024),
    (2048, 2048),
    (4096, 4096),
    (8192, 8192),
    (2, 3, 128, 128),
    (8, 16, 64, 64),
    (32, 64, 128, 128),
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
    op = _resolve("torch.ops.aten.fake_quantize_per_tensor_affine_cachemask_backward")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    grad = _generate_tensor_input(shape, dtype, device)
    mask = torch.randint(0, 2, shape, dtype=torch.bool, device=device)
    return (grad, mask), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": shape, "dtype": dt},
                {"shape": shape, "dtype": "torch.bool"},
            ],
            "kwargs": {},
        },
        "outputs": {"grad_input": {"shape": shape, "dtype": dt}},
        "dims": {"shape": shape},
    }
