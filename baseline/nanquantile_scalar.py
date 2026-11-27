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

"""NVIDIA native baseline for `nanquantile_scalar`.

Mirrors benchmark/test_nanquantile.py (NanquantileBenchmark, a
GenericBenchmark subclass; the test dispatches through the local `_run`
helper):
    op_name  = "nanquantile_scalar"
    torch_op = torch.ops.aten.nanquantile.scalar
    dtypes   = [torch.float32]

The class overrides set_shapes with its own list and only extends it at
--level comprehensive, so core_shapes.yaml is not consulted and the
(33, 65) / (4096,) / (256, 1024) / (4096, 1024) shapes are not in this grid.

`_make_input` sets every fifth flattened element to NaN -- the point of
nanquantile. The scalar overload takes q as a Python float (0.37) and dim -1.
"""

import torch

OP_NAME = "nanquantile_scalar"
DTYPES = [torch.float32]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten.nanquantile.scalar is unavailable on this torch build"
)

# NanquantileBenchmark.set_shapes (core level)
_SHAPES = [
    (1,),
    (17,),
    (1024,),
    (1025,),
    (128, 128),
    (512, 512),
    (1024, 1024),
]

_Q = 0.37
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
    """benchmark::_make_input -- every fifth flattened element is NaN."""
    inp = torch.randn(shape, dtype=dtype, device=device)
    inp.reshape(-1)[::5] = float("nan")
    return inp


def native():
    op = _resolve("torch.ops.aten.nanquantile.scalar")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    inp = _make_input(tuple(binding["shape"]), dtype, device)
    return (inp, _Q, _DIM), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": shape, "dtype": dt, "note": "every 5th element NaN"},
                {"scalar": _Q},
                {"scalar": _DIM},
            ],
            "kwargs": {},
        },
        "outputs": {"out": {"shape": shape[:-1], "dtype": dt}},
        "dims": {"shape": shape, "q": _Q, "dim": _DIM},
    }
