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

"""NVIDIA native baseline for `cumprod`.

Mirrors benchmark/test_cumprod.py (base.GenericBenchmark2DOnly):
    op_name  = "cumprod"
    torch_op = torch_cumprod (the module-local reference below)
    dtypes   = FLOAT + BOOL + INT + EXTRA_INT
    shapes   = yaml:GenericBenchmark2DOnly (benchmark/core_shapes.yaml)

The benchmark's `torch_cumprod` adds an Ascend-only bool upcast; on NVIDIA it
is a plain torch.cumprod(inp, dim) call, which is what this baseline resolves.

The input ranges matter here and come from `_make_input`: floats are drawn
uniform(0.99, 1.01) so a long running product neither overflows nor underflows,
uint8 from randint(0, 4), other integers from randint(-3, 4), and bool through
an int8 buffer. int64 is in the pool because the NVIDIA backend reports
int64_enabled=True.
"""

import torch

OP_NAME = "cumprod"
DTYPES = [
    torch.float16,
    torch.float32,
    torch.bfloat16,
    torch.bool,
    torch.int16,
    torch.int32,
    torch.int8,
    torch.uint8,
    torch.int64,
]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = "torch.cumprod is unavailable on this torch build"

# yaml:GenericBenchmark2DOnly
_SHAPES = [
    (64, 64),
    (256, 256),
    (1024, 1024),
    (4096, 4096),
    (1024, 65536),
]

_DIM = 1
_FLOAT_DTYPES = (torch.float16, torch.float32, torch.bfloat16)


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
    """benchmark/test_cumprod.py::_make_input -- range-controlled per dtype."""
    if dtype in _FLOAT_DTYPES:
        return torch.empty(shape, dtype=dtype, device=device).uniform_(0.99, 1.01)
    if dtype is torch.bool:
        return torch.randint(0, 2, shape, dtype=torch.int8, device="cpu").to(
            device, dtype=dtype
        )
    if dtype is torch.uint8:
        return torch.randint(0, 4, shape, dtype=dtype, device="cpu").to(device)
    return torch.randint(-3, 4, shape, dtype=dtype, device="cpu").to(device)


def native():
    op = _resolve("torch.cumprod")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    inp = _make_input(tuple(binding["shape"]), dtype, device)
    return (inp, _DIM), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    dt = str(dtype)
    return {
        "inputs": {
            "args": [{"shape": shape, "dtype": dt}, {"scalar": _DIM}],
            "kwargs": {},
        },
        "outputs": {"out": {"shape": shape, "dtype": dt}},
        "dims": {"shape": shape, "dim": _DIM},
    }
