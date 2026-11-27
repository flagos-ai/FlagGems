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

"""NVIDIA native baseline for `nanmedian_dim_values`.

Mirrors benchmark/test_nanmedian.py (base.GenericBenchmark):
    op_name  = "nanmedian_dim_values"
    torch_op = torch.nanmedian
    dtypes   = consts.FLOAT_DTYPES + [int8, int16, int32, uint8]
    input_fn = _dim_values_input_fn
    shapes   = yaml:nanmedian_dim_values (benchmark/core_shapes.yaml),
               shape_desc "M, N"

The benchmark's torch_op is torch.nanmedian itself; passing the (values,
indices) tuple under `out` routes it to the nanmedian.dim_values overload.

`_make_input` is dtype-dependent: uint8 from randint(0, 101), the signed
integers from randint(-100, 101), and the float dtypes from randn with every
17th flattened element set to NaN -- which is the point of nanmedian.

`_dim_values_input_fn` yields nothing for a 1-D shape; every yaml shape here is
2-D, so all four are in the grid. dim is -1 and the outputs are shape[:-1].
"""

import torch

OP_NAME = "nanmedian_dim_values"
DTYPES = [
    torch.float16,
    torch.float32,
    torch.bfloat16,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.uint8,
]
IS_INPLACE = False
MUTATED_INPUTS = ("out",)
NATIVE_UNAVAILABLE_REASON = "torch.nanmedian is unavailable on this torch build"

# yaml:nanmedian_dim_values -- shape_desc "M, N"
_SHAPES = [
    (64, 64),
    (1024, 1024),
    (256, 4096),
    (16, 131072),
]

_DIM = -1


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


def _make_input(shape, dtype, device):
    """benchmark/test_nanmedian.py::_make_input."""
    if dtype is torch.uint8:
        return torch.randint(0, 101, shape, dtype=dtype, device="cpu").to(device)
    if dtype in (torch.int8, torch.int16, torch.int32):
        return torch.randint(-100, 101, shape, dtype=dtype, device="cpu").to(device)
    inp = torch.randn(shape, dtype=dtype, device=device)
    if inp.numel() > 0:
        inp.reshape(-1)[::17] = float("nan")
    return inp


def native():
    op = _resolve("torch.nanmedian")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    inp = _make_input(shape, dtype, device)
    out_shape = shape[:-1]
    out_values = torch.empty(out_shape, dtype=dtype, device=device)
    out_indices = torch.empty(out_shape, dtype=torch.long, device=device)
    return (inp,), {"dim": _DIM, "out": (out_values, out_indices)}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    out_shape = shape[:-1]
    dt = str(dtype)
    return {
        "inputs": {
            "args": [{"shape": shape, "dtype": dt}],
            "kwargs": {
                "dim": {"scalar": _DIM},
                "out": [
                    {"shape": out_shape, "dtype": dt},
                    {"shape": out_shape, "dtype": "torch.int64"},
                ],
            },
        },
        "outputs": {
            "values": {"shape": out_shape, "dtype": dt},
            "indices": {"shape": out_shape, "dtype": "torch.int64"},
        },
        "dims": {"shape": shape, "dim": _DIM},
    }
