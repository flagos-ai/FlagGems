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

"""NVIDIA native baseline for `fractional_max_pool3d`.

Mirrors benchmark/test_fractional_max_pool3d.py (FractionalMaxPool3dBenchmark,
a GenericBenchmark subclass):
    op_name  = "fractional_max_pool3d"
    torch_op = torch.nn.functional.fractional_max_pool3d
    dtypes   = consts.FLOAT_DTYPES

The class overrides set_shapes with FRACTIONAL_MAXPOOL3D_SHAPES, so
core_shapes.yaml is not consulted. At --level core the input_fn yields one
collection point per shape with output_size = (D//2, H//2, W//2) and
kernel_size 2; the quarter-size variant is COMPREHENSIVE-only.

kernel_size/output_size are passed as keywords, as the input_fn yields them.
torch.nn.functional.fractional_max_pool3d draws its own random samples when
none are supplied, which is what the benchmark relies on.
"""

import torch

OP_NAME = "fractional_max_pool3d"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.nn.functional.fractional_max_pool3d is unavailable on this build"
)

# test_fractional_max_pool3d.py::FRACTIONAL_MAXPOOL3D_SHAPES
_SHAPES = [
    (4, 3, 8, 32, 32),
    (2, 16, 8, 28, 28),
    (1, 64, 8, 56, 56),
    (2, 8, 4, 16, 16),
    (8, 16, 4, 8, 8),
]

_KERNEL_SIZE = 2

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
    op = _resolve("torch.nn.functional.fractional_max_pool3d")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    inp = _generate_tensor_input(shape, dtype, device)
    output_size = (shape[2] // 2, shape[3] // 2, shape[4] // 2)
    return (inp,), {"kernel_size": _KERNEL_SIZE, "output_size": output_size}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    output_size = [shape[2] // 2, shape[3] // 2, shape[4] // 2]
    out_shape = [shape[0], shape[1]] + output_size
    dt = str(dtype)
    return {
        "inputs": {
            "args": [{"shape": shape, "dtype": dt}],
            "kwargs": {
                "kernel_size": {"scalar": _KERNEL_SIZE},
                "output_size": output_size,
            },
        },
        "outputs": {"out": {"shape": out_shape, "dtype": dt}},
        "dims": {
            "N": shape[0],
            "C": shape[1],
            "D": shape[2],
            "H": shape[3],
            "W": shape[4],
            "kernel_size": _KERNEL_SIZE,
            "output_size": output_size,
        },
    }
