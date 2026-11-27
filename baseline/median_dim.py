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

"""NVIDIA native baseline for `median_dim`.

Mirrors benchmark/test_median.py (MedianReductionBenchmark, a base.Benchmark
subclass):
    op_name  = "median_dim"
    torch_op = torch.median
    dtypes   = consts.FLOAT_DTYPES + consts.INT_DTYPES
    shapes   = yaml:median_dim (benchmark/core_shapes.yaml),
               shape_desc "input shape or [input shape, dim, keepdim]"

MedianReductionBenchmark reads each yaml entry two ways. An entry whose first
element is itself a list is an explicit [shape, dim, keepdim] triple; a plain
shape instead derives both from its position in the list:

    keepdim = (case_id % 3 == 0)
    dim     = 0                 if the shape is 1-D
              len(shape) - 1    if case_id % 2 == 0
              0                 otherwise

case_id is the index into the shape list, so the grid below hard-codes the
resolved (shape, dim, keepdim) for each of the seven yaml entries in order.
keepdim is passed as a keyword.
"""

import torch

OP_NAME = "median_dim"
DTYPES = [
    torch.float16,
    torch.float32,
    torch.bfloat16,
    torch.int16,
    torch.int32,
]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = "torch.median is unavailable on this torch build"

# yaml:median_dim, with dim/keepdim resolved per the class's rules.
# (shape, dim, keepdim)
_CASES = [
    ((1536, 640), 1, True),  # case 0: plain shape, 0 % 3 == 0, 0 % 2 == 0
    ((144, 8192), 0, False),  # case 1: plain shape, 1 % 3 != 0, 1 % 2 != 0
    ((768, 1536), 1, False),  # case 2: plain shape, 2 % 3 != 0, 2 % 2 == 0
    ((32, 384, 128), 0, True),  # case 3: plain shape, 3 % 3 == 0, 3 % 2 != 0
    ((32, 384, 128), 1, False),  # case 4: explicit triple
    ((640, 1536), 0, False),  # case 5: explicit triple
    ((8, 64, 384, 64), 2, False),  # case 6: explicit triple
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
    op = _resolve("torch.median")
    return op if callable(op) else None


def grid():
    return [
        {"shape": list(shape), "dim": dim, "keepdim": keepdim}
        for shape, dim, keepdim in _CASES
    ]


def _out_shape(shape, dim, keepdim):
    out_shape = list(shape)
    if keepdim:
        out_shape[dim] = 1
    else:
        del out_shape[dim]
    return out_shape


def build_inputs(binding, dtype, device):
    inp = _generate_tensor_input(tuple(binding["shape"]), dtype, device)
    dim, keepdim = binding["dim"], binding["keepdim"]
    return (inp, dim), {"keepdim": keepdim}


def key_shape(binding):
    shape = "x".join(str(d) for d in binding["shape"])
    return f"{shape}_dim{binding['dim']}_keepdim{int(binding['keepdim'])}"


def config(binding, dtype):
    shape = list(binding["shape"])
    dim, keepdim = binding["dim"], binding["keepdim"]
    out_shape = _out_shape(shape, dim, keepdim)
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": shape, "dtype": dt},
                {"scalar": dim},
            ],
            "kwargs": {"keepdim": {"scalar": keepdim}},
        },
        "outputs": {
            "values": {"shape": out_shape, "dtype": dt},
            "indices": {"shape": out_shape, "dtype": "torch.int64"},
        },
        "dims": {"shape": shape, "dim": dim, "keepdim": keepdim},
    }
