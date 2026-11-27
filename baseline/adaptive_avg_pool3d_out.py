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

"""NVIDIA native baseline for `adaptive_avg_pool3d_out`.

Mirrors benchmark/test__adaptive_avg_pool3d.py
(AdaptiveAvgPool3dOutBenchmark, a GenericBenchmark subclass):
    op_name  = "adaptive_avg_pool3d_out"
    torch_op = torch.ops.aten._adaptive_avg_pool3d.out
    dtypes   = consts.FLOAT_DTYPES

The benchmark class overrides get_input_iter and iterates its own hard-coded
5-D list, so core_shapes.yaml is not consulted at all; those five shapes are
reproduced verbatim below. At --level core the input_fn yields a single
output_size of (7, 7, 7); the (1,1,1)/(2,2,2)/(4,4,4) sizes are
COMPREHENSIVE-only and are therefore not part of this grid.
"""

import torch

OP_NAME = "adaptive_avg_pool3d_out"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten._adaptive_avg_pool3d.out is unavailable on this torch build"
)

# AdaptiveAvgPool3dOutBenchmark.get_input_iter::shapes_5d
_SHAPES = [
    (2, 3, 8, 8, 8),  # Small input
    (4, 16, 16, 16, 16),  # Medium input
    (2, 64, 32, 32, 32),  # Medium input
    (1, 3, 64, 64, 64),  # Typical video clip
    (4, 128, 16, 16, 16),  # Typical intermediate feature map
]

_OUTPUT_SIZE = (7, 7, 7)

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
    op = _resolve("torch.ops.aten._adaptive_avg_pool3d.out")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    inp = _generate_tensor_input(shape, dtype, device)
    out = torch.empty((shape[0], shape[1], *_OUTPUT_SIZE), dtype=dtype, device=device)
    return (inp, _OUTPUT_SIZE), {"out": out}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    out_shape = [shape[0], shape[1], *_OUTPUT_SIZE]
    dt = str(dtype)
    return {
        "inputs": {
            "args": [{"shape": shape, "dtype": dt}, list(_OUTPUT_SIZE)],
            "kwargs": {"out": {"shape": out_shape, "dtype": dt}},
        },
        "outputs": {"out": {"shape": out_shape, "dtype": dt}},
        "dims": {
            "N": shape[0],
            "C": shape[1],
            "D": shape[2],
            "H": shape[3],
            "W": shape[4],
            "output_size": list(_OUTPUT_SIZE),
        },
    }
